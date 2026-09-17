"""Thread-safe MuJoCo keyboard input."""

from __future__ import annotations

import queue
import threading
from enum import Enum, auto


class KeyCommand(Enum):
    DAMPING = auto()
    FIXEDPOSE = auto()
    LOCO = auto()
    TRACKING = auto()
    EXIT = auto()


_KEY_MAP = {
    "p": KeyCommand.DAMPING,
    "r": KeyCommand.FIXEDPOSE,
    "l": KeyCommand.LOCO,
    "m": KeyCommand.TRACKING,
}

_MOVEMENT_KEYS = (
    ("up", "down"),
    ("left", "right"),
    ("q", "e"),
)


class Keyboard:
    def __init__(self, enabled: bool = True) -> None:
        self._events: queue.SimpleQueue[KeyCommand] = queue.SimpleQueue()
        self._pressed: set[str] = set()
        self._lock = threading.Lock()
        self._listener = None
        self._backend = None
        if enabled:
            from pynput import keyboard

            self._backend = keyboard
            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.start()

    def _key_name(self, key: object) -> str:
        char = getattr(key, "char", None)
        if char is not None:
            return str(char).lower()
        if self._backend is None:
            return ""
        special_keys = {
            self._backend.Key.esc: "esc",
            self._backend.Key.up: "up",
            self._backend.Key.down: "down",
            self._backend.Key.left: "left",
            self._backend.Key.right: "right",
        }
        return special_keys.get(key, "")

    def _on_press(self, key: object) -> None:
        name = self._key_name(key)
        if name == "esc":
            self._events.put(KeyCommand.EXIT)
            return
        if name:
            with self._lock:
                self._pressed.add(name)

    def _on_release(self, key: object) -> None:
        name = self._key_name(key)
        with self._lock:
            self._pressed.discard(name)
        command = _KEY_MAP.get(name)
        if command is not None:
            self._events.put(command)

    def poll(self) -> list[KeyCommand]:
        commands = []
        while not self._events.empty():
            commands.append(self._events.get())
        return commands

    def movement(self) -> tuple[int, int, int]:
        with self._lock:
            return tuple(
                int(positive in self._pressed) - int(negative in self._pressed) for positive, negative in _MOVEMENT_KEYS
            )

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
