"""Resilient WebSocket client for connecting the voice bridge to the NAT server.

Auto-reconnects with exponential backoff. Dispatches received messages
to registered callbacks. Fires on_connect / on_disconnect hooks.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Coroutine, Dict, Optional

from loguru import logger


MessageCallback = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]
LifecycleHook = Callable[[], Any]


class NATWebSocketClient:
    """WebSocket client that connects to the NAT server."""

    def __init__(
        self,
        url: str,
        session_id: str,
        camera_index: int,
        rtsp_base: str = "rtsp://mediamtx:8554",
        on_connect: Optional[LifecycleHook] = None,
        on_disconnect: Optional[LifecycleHook] = None,
    ):
        self._url = f"{url}?session_id={session_id}"
        self._session_id = session_id
        self._camera_index = camera_index
        self._rtsp_base = rtsp_base
        self._ws = None
        self._running = False
        self._callbacks: Dict[str, MessageCallback] = {}
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._last_protocol_signature: Optional[str] = None
        self._protocol_sync_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def on(self, message_type: str, callback: MessageCallback):
        self._callbacks[message_type] = callback

    async def send(self, message: dict) -> bool:
        if self._ws is None:
            return False
        try:
            await self._ws.send(json.dumps(message))
            return True
        except Exception as exc:
            logger.warning(f"[WS Client] Send failed: {exc}")
            return False

    async def run(self):
        import websockets

        self._running = True
        while self._running:
            protocol_sync_task: Optional[asyncio.Task] = None
            try:
                logger.info(f"[WS Client] Connecting to {self._url}")
                async with websockets.connect(self._url) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1.0
                    logger.info(f"[WS Client] Connected (session={self._session_id})")
                    protocol_sync_task = asyncio.create_task(self._protocol_sync_loop())

                    if self._on_connect:
                        self._on_connect()

                    await self._send_stream_info()

                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            await self._dispatch(msg)
                        except json.JSONDecodeError:
                            logger.warning("[WS Client] Invalid JSON from NAT")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    f"[WS Client] Connection lost: {exc}. "
                    f"Reconnecting in {self._reconnect_delay:.0f}s"
                )
            finally:
                if protocol_sync_task is not None:
                    try:
                        protocol_sync_task.cancel()
                        await protocol_sync_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                was_connected = self._ws is not None
                self._ws = None
                if was_connected and self._on_disconnect:
                    self._on_disconnect()

            if not self._running:
                break
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * 2, self._max_reconnect_delay
            )

    async def reset_session(self):
        """Close the current WS connection; auto-reconnect will create a fresh session."""
        if self._ws:
            logger.info("[WS Client] Resetting session (closing connection)")
            await self._ws.close()

    async def stop(self):
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _send_stream_info(self):
        idx = f"{self._camera_index:04d}"
        sent = await self.send({
            "type": "stream_info",
            "camera_index": self._camera_index,
            "rtsp_base": self._rtsp_base,
            "paths": {
                "video": f"NB_{idx}_TX_CAM_RGB",
                "audio": f"NB_{idx}_TX_MIC_p6S",
                "merged": f"NB_{idx}_TX_CAM_RGB_MIC_p6S",
            },
        })
        if not sent:
            logger.warning("[WS Client] Failed to send stream_info; will retry on next reconnect")
        await self._send_local_protocols(force=True)

    def _collect_local_protocols(self) -> list[dict[str, str]]:
        """Scan local protocols/ directory and return payload-ready protocol data."""
        from pathlib import Path

        proto_dir = Path("protocols")
        if not proto_dir.is_dir():
            return []

        exts = {".txt", ".md", ".csv", ".json", ".yaml", ".yml"}
        protocols = []
        for f in sorted(proto_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                try:
                    content = f.read_text(encoding="utf-8")
                    if content.strip():
                        protocols.append({"name": f.name, "content": content})
                except Exception as exc:
                    logger.warning(f"[WS Client] Failed to read protocol {f.name}: {exc}")
        return protocols

    async def _send_local_protocols(self, force: bool = False):
        """Push protocols to NAT (all filenames preserved exactly)."""
        async with self._protocol_sync_lock:
            protocols = self._collect_local_protocols()
            signature = json.dumps(protocols, sort_keys=True, ensure_ascii=False)
            if not force and signature == self._last_protocol_signature:
                return True

            sent = await self.send({"type": "protocol_push", "protocols": protocols})
            if not sent:
                logger.warning("[WS Client] Protocol push failed; keeping pending state for retry")
                return False

            self._last_protocol_signature = signature
            names = ", ".join(p["name"] for p in protocols) if protocols else "(none)"
            logger.info(f"[WS Client] Pushed {len(protocols)} protocol(s) to NAT: {names}")
            return True

    async def _protocol_sync_loop(self):
        """Periodically sync local protocol files to NAT while connected."""
        while self._ws is not None:
            try:
                sent = await self._send_local_protocols(force=False)
            except Exception as exc:
                logger.warning(f"[WS Client] Protocol sync failed: {exc}")
                sent = False
            # Retry faster when a push did not go through.
            await asyncio.sleep(5.0 if sent else 2.0)

    async def _dispatch(self, msg: dict):
        msg_type = msg.get("type", "")
        callback = self._callbacks.get(msg_type)
        if callback:
            try:
                await callback(msg)
            except Exception as exc:
                logger.error(f"[WS Client] Callback error for {msg_type}: {exc}")
        elif msg_type == "pong":
            pass
        else:
            logger.debug(f"[WS Client] Unhandled message type: {msg_type}")
