"""Session recorder -- writes video, chat, errors, and summary to disk.

Creates a per-session folder under ``recordings/{date}_cam_{N}/{HH-MM-SS}/`` containing:
  - recording.mp4  -- H.264 fragmented MP4 (black frames on capture failure)
  - chat.txt       -- timestamped user/agent messages
  - summary.txt    -- user/protocol events (commands, wake words, tool calls)
  - log.txt        -- system/infrastructure events (connect, FFmpeg, frames)
  - session.log    -- raw timestamped log of ALL events

Frames are pushed externally via :meth:`push_frame` so the recorder never
creates its own XR service connection (avoids dual-connection conflicts).
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger


class SessionRecorder:
    """Records video frames, chat, errors, and summary for a single session.

    Frames must be pushed from the outside via :meth:`push_frame`.  The
    internal writer thread drains the queue at the target frame rate and
    substitutes black frames when no data is available.
    """

    def __init__(
        self,
        camera_index: int,
        recordings_root: str = "/app/recordings",
        width: int = 1280,
        height: int = 720,
        framerate: int = 15,
    ):
        self._camera_index = camera_index
        self._recordings_root = recordings_root
        self._width = width
        self._height = height
        self._framerate = framerate

        self._session_dir: Optional[Path] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        self._frame_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None

        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=30)

        self._chat_file = None
        self._log_file = None
        self._chat_lock = threading.Lock()
        self._log_lock = threading.Lock()

        self._start_time: Optional[float] = None
        self._message_counts: dict[str, int] = {}

        self._user_error_count = 0
        self._user_data_count = 0
        self._user_events: list[str] = []

        self._sys_error_count = 0
        self._sys_data_count = 0
        self._sys_events: list[str] = []

        self._MAX_NOTABLE_EVENTS = 200
        self._summary_lock = threading.Lock()
        self._last_summary_write = 0.0
        self._SUMMARY_WRITE_INTERVAL = 2.0

        self._black_frame = np.zeros((height, width, 3), dtype=np.uint8)

        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Create session folder and begin recording.  Returns True on success."""
        if self._running:
            logger.warning("[Recorder] Already running")
            return True

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        self._session_dir = (
            Path(self._recordings_root) / f"{date_str}_cam_{self._camera_index}" / time_str
        )
        self._session_dir.mkdir(parents=True, exist_ok=True)

        video_path = self._session_dir / "recording.mp4"
        chat_path = self._session_dir / "chat.txt"
        log_path = self._session_dir / "session.log"

        try:
            self._chat_file = open(chat_path, "a", encoding="utf-8")
            self._log_file = open(log_path, "a", encoding="utf-8")
        except OSError as exc:
            logger.error(f"[Recorder] Failed to open log files: {exc}")
            return False

        header = f"=== Session started {now.strftime('%Y-%m-%d %H:%M:%S')} | cam {self._camera_index} ===\n"
        self._chat_file.write(header)
        self._chat_file.flush()
        self._log_file.write(header)
        self._log_file.flush()

        if not self._start_ffmpeg(str(video_path)):
            logger.error("[Recorder] Failed to start FFmpeg")
            self._close_files()
            return False

        self._start_time = time.time()
        self._message_counts = {}
        self._user_error_count = 0
        self._user_data_count = 0
        self._user_events = []
        self._sys_error_count = 0
        self._sys_data_count = 0
        self._sys_events = []
        self._stop_event.clear()
        # Drain any stale frames from a previous session
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        self._running = True

        self._frame_thread = threading.Thread(
            target=self._frame_writer_loop, name="rec-frame-writer", daemon=True,
        )
        self._frame_thread.start()

        logger.info(f"[Recorder] Recording to {self._session_dir}")
        self.log_data(f"Recording started -> {video_path.name}", user_facing=False)
        self._write_files(final=False)
        return True

    def stop(self) -> None:
        """Stop recording and write the final summary file."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        if self._frame_thread and self._frame_thread.is_alive():
            self._frame_thread.join(timeout=5.0)

        self._stop_ffmpeg()
        self._write_log("DATA", "Recording stopped")
        self._sys_data_count += 1
        self._add_event("DATA", "Recording stopped", user_facing=False)
        self._write_files(final=True)
        self._close_files()
        logger.info("[Recorder] Stopped")

    # ------------------------------------------------------------------
    # Public frame push
    # ------------------------------------------------------------------

    def push_frame(self, frame: np.ndarray) -> None:
        """Enqueue a BGR frame for recording.  Drops silently if full."""
        if not self._running:
            return
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            frame = cv2.resize(frame, (self._width, self._height))
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Public logging helpers
    # ------------------------------------------------------------------

    def log_chat(self, role: str, text: str) -> None:
        """Append a timestamped chat line.  ``role`` is e.g. 'User' or 'Agent'."""
        if not self._running or self._chat_file is None:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rec_ts = self._rec_elapsed()
        line = f"[{ts}] [{rec_ts}] {role}: {text}\n"
        with self._chat_lock:
            try:
                self._chat_file.write(line)
                self._chat_file.flush()
            except OSError:
                pass
        self._message_counts[role] = self._message_counts.get(role, 0) + 1
        self._flush_files()

    def log_error(self, text: str, *, user_facing: bool = False) -> None:
        """Append a timestamped ERROR entry to session.log.

        Set ``user_facing=True`` for protocol-level errors that belong in
        summary.txt.  System/infra errors go to log.txt (default).
        """
        if user_facing:
            self._user_error_count += 1
        else:
            self._sys_error_count += 1
        self._add_event("ERROR", text, user_facing=user_facing)
        self._write_log("ERROR", text)
        self._flush_files()

    def log_data(self, text: str, *, user_facing: bool = False) -> None:
        """Append a timestamped DATA entry to session.log.

        Set ``user_facing=True`` for protocol-level events that belong in
        summary.txt.  System/infra events go to log.txt (default).
        """
        if user_facing:
            self._user_data_count += 1
        else:
            self._sys_data_count += 1
        self._add_event("DATA", text, user_facing=user_facing)
        self._write_log("DATA", text)
        self._flush_files()

    def _add_event(self, level: str, text: str, *, user_facing: bool) -> None:
        target = self._user_events if user_facing else self._sys_events
        if len(target) < self._MAX_NOTABLE_EVENTS:
            ts = datetime.now().strftime("%H:%M:%S")
            rec_ts = self._rec_elapsed()
            target.append(f"[{ts}] [{rec_ts}] {level}: {text}")

    def _flush_files(self) -> None:
        """Rewrite summary.txt and log.txt if enough time has passed."""
        now = time.monotonic()
        if now - self._last_summary_write < self._SUMMARY_WRITE_INTERVAL:
            return
        with self._summary_lock:
            if now - self._last_summary_write < self._SUMMARY_WRITE_INTERVAL:
                return
            self._write_files(final=False)
            self._last_summary_write = now

    # ------------------------------------------------------------------
    # FFmpeg management
    # ------------------------------------------------------------------

    def _start_ffmpeg(self, output_path: str) -> bool:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._framerate),
            "-i", "pipe:0",
            "-vf", "format=yuv420p",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-movflags", "frag_keyframe+empty_moov",
            output_path,
        ]
        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, bufsize=0,
            )
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, name="rec-ffmpeg-stderr", daemon=True,
            )
            self._stderr_thread.start()
            return True
        except Exception as exc:
            logger.error(f"[Recorder] FFmpeg launch failed: {exc}")
            self._ffmpeg_proc = None
            return False

    def _stop_ffmpeg(self) -> None:
        proc = self._ffmpeg_proc
        self._ffmpeg_proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                try:
                    proc.stdin.flush()
                    proc.stdin.close()
                except OSError:
                    pass
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("[Recorder] FFmpeg did not exit in time, terminating")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _drain_stderr(self) -> None:
        proc = self._ffmpeg_proc
        if not proc or not proc.stderr:
            return
        try:
            for raw in iter(proc.stderr.readline, b""):
                if not raw:
                    break
                line = raw.decode(errors="ignore").strip()
                low = line.lower()
                if any(w in low for w in ("error", "fail", "invalid")):
                    logger.error(f"[Recorder] ffmpeg: {line}")
                    self.log_error(f"ffmpeg: {line}")
                elif any(w in low for w in ("warning", "discont")):
                    logger.warning(f"[Recorder] ffmpeg: {line}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Frame writer thread
    # ------------------------------------------------------------------

    def _frame_writer_loop(self) -> None:
        """Drain the frame queue and pipe to FFmpeg at target FPS.

        When the queue is empty the last successfully received frame is
        repeated so that minor async scheduling jitter doesn't produce
        black-frame flicker.  Black frames only appear when no real
        frame has ever been received.
        """
        interval = 1.0 / self._framerate
        last_good_frame = self._black_frame
        consecutive_stale = 0

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            # Drain the queue and keep only the newest frame so we
            # never fall behind.
            frame = None
            try:
                while True:
                    frame = self._frame_queue.get_nowait()
            except queue.Empty:
                pass

            if frame is not None:
                last_good_frame = frame
                consecutive_stale = 0
            else:
                consecutive_stale += 1
                if consecutive_stale == self._framerate * 5:
                    self.log_error("Sustained frame capture failure (5s+ without new frames)")

            self._write_frame(last_good_frame)

            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def _write_frame(self, frame: np.ndarray) -> None:
        proc = self._ffmpeg_proc
        if proc is None or proc.poll() is not None:
            return
        try:
            raw_bytes = np.ascontiguousarray(frame).tobytes()
            proc.stdin.write(raw_bytes)
        except BrokenPipeError:
            logger.error("[Recorder] FFmpeg pipe broken")
            self.log_error("FFmpeg pipe broken during recording")
            self._running = False
            self._stop_event.set()
        except Exception as exc:
            logger.error(f"[Recorder] Frame write error: {exc}")

    # ------------------------------------------------------------------
    # Summary + Log file writers
    # ------------------------------------------------------------------

    def _session_header(self, *, final: bool) -> list[str]:
        now = datetime.now()
        duration = time.time() - self._start_time if self._start_time else 0
        mins, secs = divmod(int(duration), 60)
        hours, mins = divmod(mins, 60)
        status = "COMPLETED" if final else "IN PROGRESS"
        return [
            f"Status:     {status}",
            f"Camera:     {self._camera_index}",
            f"Started:    {datetime.fromtimestamp(self._start_time).strftime('%Y-%m-%d %H:%M:%S') if self._start_time else 'N/A'}",
            f"Ended:      {now.strftime('%Y-%m-%d %H:%M:%S') if final else '(still recording)'}",
            f"Duration:   {hours:02d}:{mins:02d}:{secs:02d}",
            f"Resolution: {self._width}x{self._height} @ {self._framerate} fps",
        ]

    def _write_files(self, *, final: bool = True) -> None:
        if self._session_dir is None:
            return
        self._write_summary_txt(final=final)
        self._write_log_txt(final=final)

    def _write_summary_txt(self, *, final: bool) -> None:
        """Write summary.txt -- user/protocol events only."""
        path = self._session_dir / "summary.txt"
        header = self._session_header(final=final)

        user_msgs = self._message_counts.get("User", 0)
        agent_msgs = self._message_counts.get("Agent", 0)

        lines = [
            "Session Summary",
            "===============",
            *header,
            "",
            "Totals",
            "------",
            f"  User messages:  {user_msgs}",
            f"  Agent messages: {agent_msgs}",
            f"  User errors:    {self._user_error_count}",
            f"  User events:    {self._user_data_count}",
            "",
        ]

        other_roles = {r: c for r, c in self._message_counts.items() if r not in ("User", "Agent")}
        if other_roles:
            lines.append("Other message sources:")
            for role, count in sorted(other_roles.items()):
                lines.append(f"  {role}: {count}")
            lines.append("")

        if self._user_events:
            lines.append("Notable Events")
            lines.append("--------------")
            for event in self._user_events:
                lines.append(f"  {event}")
            if len(self._user_events) >= self._MAX_NOTABLE_EVENTS:
                lines.append(f"  ... (truncated at {self._MAX_NOTABLE_EVENTS} events)")
            lines.append("")
        else:
            lines.append("Notable Events")
            lines.append("--------------")
            lines.append("  (none)")
            lines.append("")

        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            logger.error(f"[Recorder] Failed to write summary.txt: {exc}")

    def _write_log_txt(self, *, final: bool) -> None:
        """Write log.txt -- system/infrastructure events only."""
        path = self._session_dir / "log.txt"
        header = self._session_header(final=final)

        lines = [
            "System Log",
            "==========",
            *header,
            "",
            "Totals",
            "------",
            f"  System errors:  {self._sys_error_count}",
            f"  System events:  {self._sys_data_count}",
            "",
        ]

        if self._sys_events:
            lines.append("Events")
            lines.append("------")
            for event in self._sys_events:
                lines.append(f"  {event}")
            if len(self._sys_events) >= self._MAX_NOTABLE_EVENTS:
                lines.append(f"  ... (truncated at {self._MAX_NOTABLE_EVENTS} events)")
            lines.append("")
        else:
            lines.append("Events")
            lines.append("------")
            lines.append("  (none)")
            lines.append("")

        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            logger.error(f"[Recorder] Failed to write log.txt: {exc}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rec_elapsed(self) -> str:
        """Return elapsed recording time as ``HH:MM:SS``."""
        if self._start_time is None:
            return "00:00:00"
        elapsed = time.time() - self._start_time
        mins, secs = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    def _write_log(self, level: str, text: str) -> None:
        if self._log_file is None:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rec_ts = self._rec_elapsed()
        line = f"[{ts}] [{rec_ts}] {level}: {text}\n"
        with self._log_lock:
            try:
                self._log_file.write(line)
                self._log_file.flush()
            except OSError:
                pass

    def _close_files(self) -> None:
        for fh in (self._chat_file, self._log_file):
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
        self._chat_file = None
        self._log_file = None
