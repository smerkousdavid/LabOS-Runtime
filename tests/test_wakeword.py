"""Unit tests for the WakeWordFilter state machine."""

from __future__ import annotations

import time
from unittest.mock import patch

from wakeword import WakeWordFilter, State


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_filter(**kwargs) -> WakeWordFilter:
    defaults = dict(
        wake_words=["stella", "hey stella", "hi stella"],
        timeout_seconds=10.0,
        sleep_commands=["thanks", "goodbye", "go to sleep"],
    )
    defaults.update(kwargs)
    return WakeWordFilter(**defaults)


# ---------------------------------------------------------------------------
# Wake word detection
# ---------------------------------------------------------------------------

class TestWakeWordDetection:
    def test_simple_wake_word(self):
        f = _make_filter()
        result = f.process("stella look up a picture of a cat")
        assert result == "look up a picture of a cat"
        assert f.state == State.ACTIVE

    def test_hey_stella(self):
        f = _make_filter()
        result = f.process("hey stella what is the weather")
        assert result == "what is the weather"
        assert f.state == State.ACTIVE

    def test_hi_stella(self):
        f = _make_filter()
        result = f.process("hi stella list some protocols")
        assert result == "list some protocols"
        assert f.state == State.ACTIVE

    def test_wake_word_only_returns_none(self):
        """Wake word alone with no follow-up text returns None."""
        f = _make_filter()
        result = f.process("stella")
        assert result is None
        assert f.state == State.ACTIVE

    def test_wake_word_with_leading_comma(self):
        f = _make_filter()
        result = f.process("stella, open the door")
        assert result == "open the door"

    def test_wake_word_with_leading_period(self):
        f = _make_filter()
        result = f.process("stella. tell me a joke")
        assert result == "tell me a joke"

    def test_case_insensitive(self):
        f = _make_filter()
        result = f.process("Stella tell me something")
        assert result == "tell me something"
        assert f.state == State.ACTIVE


# ---------------------------------------------------------------------------
# Filtering when IDLE
# ---------------------------------------------------------------------------

class TestIdleFiltering:
    def test_no_wake_word_returns_none(self):
        f = _make_filter()
        assert f.state == State.IDLE
        result = f.process("just some random text")
        assert result is None

    def test_empty_string_returns_none(self):
        f = _make_filter()
        result = f.process("   ")
        assert result is None


# ---------------------------------------------------------------------------
# Active pass-through
# ---------------------------------------------------------------------------

class TestActivePassthrough:
    def test_passthrough_while_active(self):
        f = _make_filter()
        f.process("stella start")
        assert f.state == State.ACTIVE

        result = f.process("now do something else")
        assert result == "now do something else"

    def test_passthrough_resets_timer(self):
        f = _make_filter(timeout_seconds=5.0)
        f.process("stella begin")
        first_activity = f._last_activity

        time.sleep(0.05)
        f.process("keep going")
        assert f._last_activity > first_activity


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_returns_to_idle(self):
        f = _make_filter(timeout_seconds=0.1)
        f.process("stella start")
        assert f.state == State.ACTIVE

        time.sleep(0.15)
        assert f.state == State.IDLE

    def test_text_filtered_after_timeout(self):
        f = _make_filter(timeout_seconds=0.1)
        f.process("stella go")
        time.sleep(0.15)

        result = f.process("more text without wake word")
        assert result is None

    def test_timeout_setter_clamps(self):
        f = _make_filter()
        f.timeout_seconds = 0.5
        assert f.timeout_seconds == 1.0  # clamped to minimum


# ---------------------------------------------------------------------------
# Sleep commands
# ---------------------------------------------------------------------------

class TestSleepCommands:
    def test_thanks_deactivates(self):
        f = _make_filter()
        f.process("stella begin")
        assert f.state == State.ACTIVE

        result = f.process("thanks")
        assert result is None
        assert f.state == State.IDLE

    def test_goodbye_deactivates(self):
        f = _make_filter()
        f.process("stella begin")
        result = f.process("goodbye")
        assert result is None
        assert f.state == State.IDLE

    def test_go_to_sleep_deactivates(self):
        f = _make_filter()
        f.process("stella begin")
        result = f.process("go to sleep")
        assert result is None
        assert f.state == State.IDLE

    def test_sleep_command_with_trailing_punctuation(self):
        f = _make_filter()
        f.process("stella begin")
        result = f.process("thanks.")
        assert result is None
        assert f.state == State.IDLE

    def test_sleep_command_while_idle_ignored(self):
        f = _make_filter()
        result = f.process("thanks")
        assert result is None
        assert f.state == State.IDLE


# ---------------------------------------------------------------------------
# State transitions sequence
# ---------------------------------------------------------------------------

class TestStateTransitions:
    def test_full_lifecycle(self):
        f = _make_filter()
        assert f.state == State.IDLE

        f.process("hey stella what is up")
        assert f.state == State.ACTIVE

        f.process("tell me more")
        assert f.state == State.ACTIVE

        f.process("goodbye")
        assert f.state == State.IDLE

        result = f.process("nobody is listening")
        assert result is None
        assert f.state == State.IDLE

        f.process("stella restart")
        assert f.state == State.ACTIVE
