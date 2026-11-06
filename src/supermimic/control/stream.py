"""Validated localhost UDP targets and Linux joystick input for simulation tests."""
from __future__ import annotations

import json
import math
import os
import socket
import struct
import time

from supermimic.control.targets import INTERFACE_DIMS


class TargetReceiver:
    """Packet: {interface, sequence, timestamp (Unix seconds), target: [...]}.

    A timeout aborts simulation rather than inventing a neutral pose. Senders must
    use increasing sequence numbers and synchronized clocks; restart the receiver
    when restarting the sender. Invalid packets never refresh freshness.
    """
    def __init__(self, interface: str, port: int, max_age: float = 0.25):
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("max_age must be finite and positive")
        self.interface, self.max_age = interface, max_age
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.sequence = -1
        self.target = None
        self.timestamp = -math.inf

    def accept(self, packet: dict, now: float) -> bool:
        try:
            seq, stamp, target = packet["sequence"], packet["timestamp"], packet["target"]
            if (packet["interface"] != self.interface or type(seq) is not int
                    or seq <= self.sequence or type(stamp) not in (float, int)
                    or not math.isfinite(stamp) or not 0 <= now - stamp <= self.max_age
                    or not isinstance(target, list) or len(target) != INTERFACE_DIMS[self.interface]
                    or not all(type(v) in (float, int) and math.isfinite(v) for v in target)):
                return False
        except (KeyError, TypeError):
            return False
        self.sequence, self.timestamp, self.target = seq, stamp, target
        return True

    def read(self) -> list[float]:
        now = time.time()
        # Bound work per control step even if an input producer floods the port.
        for _ in range(128):
            try:
                data = self.sock.recv(16384)
            except BlockingIOError:
                break
            try:
                self.accept(json.loads(data), now)
            except (ValueError, UnicodeError):
                pass
        if self.target is None or not 0 <= now - self.timestamp <= self.max_age:
            raise TimeoutError("control target missing or stale; stopping simulation")
        return self.target

    def close(self):
        self.sock.close()


class LinuxJoystick:
    """Linux /dev/input/js* axes; map/sign must be calibrated for the actual device."""
    def __init__(self, path: str):
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.axes = {}

    def read(self, indices: tuple[int, int, int], signs: tuple[float, float, float]):
        for _ in range(128):
            try:
                event = os.read(self.fd, 8)
            except BlockingIOError:
                break
            if len(event) != 8:
                raise ConnectionError("joystick disconnected")
            _, value, kind, number = struct.unpack("IhBB", event)
            if kind & 0x02:
                self.axes[number] = max(-1.0, value / 32767.0)
        if not all(index in self.axes for index in indices):
            raise ConnectionError("joystick axes not initialized; check axis mapping")
        return [self.axes[index] * sign for index, sign in zip(indices, signs)]

    def close(self):
        os.close(self.fd)
