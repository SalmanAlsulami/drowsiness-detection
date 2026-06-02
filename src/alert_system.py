"""
Audio alert system for the drowsiness demo.
Three alert behaviors:
  1. warning  - single beep at 880 Hz
  2. danger   - repeating beep at 1200 Hz every 2 seconds
  3. none     - stop all sound

Usage in demo.py:
    from alert_system import AlertManager
    alert_mgr = AlertManager(PROJECT_ROOT)
    alert_mgr.update("danger")   # or "warning" or "none"
"""

import math
import struct
import threading
import time
import wave
from pathlib import Path

import pygame

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _generate_beep_wav(path: Path, freq: float, duration: float,
                       volume: float = 0.9, sample_rate: int = 44100) -> None:
    # generates a simple sine-wave WAV beep with fade in/out to avoid clicks
    n_samples = int(sample_rate * duration)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            t    = i / sample_rate
            fade = min(i, n_samples - i, int(n_samples * 0.1))
            amp  = volume * (fade / (n_samples * 0.1)) if fade < n_samples * 0.1 else volume
            wf.writeframes(struct.pack("<h", int(amp * 32767 * math.sin(2 * math.pi * freq * t))))


def _ensure_alert_sounds(root: Path) -> dict:
    # makes sure both alert WAV files exist, generating them if needed
    sounds = {
        "warning": root / "alert_warning.wav",
        "danger":  root / "alert_danger.wav",
    }
    if not sounds["warning"].exists():
        print("  [Alert] Generating warning beep ...")
        _generate_beep_wav(sounds["warning"], freq=880,  duration=0.4)
    if not sounds["danger"].exists():
        print("  [Alert] Generating danger beep ...")
        _generate_beep_wav(sounds["danger"],  freq=1200, duration=0.8, volume=1.0)

    # if the old alert.wav exists, use it for danger
    legacy = root / "alert.wav"
    if legacy.exists():
        sounds["danger"] = legacy

    return sounds


class AlertManager:
    """
    Manages audio alerts in a background thread so model inference is not blocked.
    - warning: plays once when the state first becomes warning
    - danger:  plays every REPEAT_INTERVAL seconds while state is danger
    - none:    stops any playing sound
    """

    REPEAT_INTERVAL = 2.0

    def __init__(self, root: Path):
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=1, buffer=512)
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        self._sounds = _ensure_alert_sounds(root)
        self._state  = "none"
        self._lock   = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, state: str) -> None:
        with self._lock:
            self._state = state

    def _play(self, key: str) -> None:
        try:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.load(str(self._sounds[key]))
                pygame.mixer.music.play()
        except Exception as e:
            print(f"  [Alert] Sound error: {e}")

    def _loop(self) -> None:
        last_state     = "none"
        last_play_time = 0.0

        while True:
            with self._lock:
                state = self._state

            now = time.time()

            if state == "danger":
                if last_state != "danger" or now - last_play_time >= self.REPEAT_INTERVAL:
                    self._play("danger")
                    last_play_time = now
            elif state == "warning":
                if last_state != "warning":
                    self._play("warning")
                    last_play_time = now
            else:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()

            last_state = state
            time.sleep(0.05)


if __name__ == "__main__":
    print("Testing alert system ...")
    mgr = AlertManager(PROJECT_ROOT)

    print("Warning for 3 seconds ...")
    mgr.update("warning")
    time.sleep(3)

    print("Danger for 6 seconds (repeats every 2s) ...")
    mgr.update("danger")
    time.sleep(6)

    print("Stopping ...")
    mgr.update("none")
    time.sleep(1)
    print("Done.")
