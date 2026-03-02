"""Simple wake-word filter state machine.

Extracted and simplified from the old runtime_connector/filters/wakeword.py.
No Pipecat dependency -- pure function interface.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import List, Optional


class State(str, Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"


class WakeWordFilter:
    def __init__(
        self,
        wake_words: Optional[List[str]] = None,
        timeout_seconds: float = 10.0,
        sleep_commands: Optional[List[str]] = None,
    ):
        self._wake_words = [w.lower() for w in (wake_words or ["stella", "hey stella"])]
        self._sleep_commands = [c.lower() for c in (sleep_commands or ["thanks", "goodbye", "go to sleep"])]
        self._timeout = timeout_seconds
        self._state = State.IDLE
        self._last_activity = 0.0

    @property
    def state(self) -> State:
        if self._state == State.ACTIVE and self._is_timed_out():
            self._state = State.IDLE
        return self._state

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    @timeout_seconds.setter
    def timeout_seconds(self, value: float):
        self._timeout = max(1.0, value)

    def process(self, transcription: str) -> Optional[str]:
        """Process a transcription. Returns cleaned text if active, None if filtered.

        If a wake word is detected, strips it and activates.
        If already active, resets the timeout timer and passes text through.
        If a sleep command is detected while active, deactivates and returns None.
        """
        text = transcription.strip()
        if not text:
            return None

        text_lower = text.lower()

        # Check for wake word activation
        for ww in self._wake_words:
            if text_lower.startswith(ww):
                self._activate()
                cleaned = text[len(ww):].strip()
                cleaned = cleaned.lstrip(",").lstrip(".").strip()
                return cleaned if cleaned else None

        # If not active (or timed out), filter out
        if self.state != State.ACTIVE:
            return None

        # Active: check for sleep commands
        for cmd in self._sleep_commands:
            if text_lower.strip().rstrip(".!") == cmd:
                self._state = State.IDLE
                return None

        # Active: pass through and reset timer
        self._last_activity = time.monotonic()
        return text

    def _activate(self):
        self._state = State.ACTIVE
        self._last_activity = time.monotonic()

    def _is_timed_out(self) -> bool:
        if self._last_activity == 0.0:
            return False
        return (time.monotonic() - self._last_activity) > self._timeout
