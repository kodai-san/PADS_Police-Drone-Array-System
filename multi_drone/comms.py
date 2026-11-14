"""
Communication model: ideal, or degraded (packet drops + latency).
"""
import numpy as np
from collections import deque
from typing import Optional, Dict, Any


class CommChannel:
    """
    Simulates inter-drone communication with optional packet drops and latency.

    Usage:
        ch = CommChannel(drop_rate=0.3, latency_ms=100.0, dt=0.05)
        ch.send(sender_id=0, data=state_dict, t=current_time)
        received = ch.receive(receiver_id=1, t=current_time)
    """

    def __init__(self, drop_rate: float = 0.0, latency_ms: float = 0.0,
                 dt: float = 0.05, rng: Optional[np.random.Generator] = None):
        self.drop_rate = drop_rate
        self.latency_s = latency_ms / 1000.0
        self.dt = dt
        self.rng = rng or np.random.default_rng()
        self._buffer: deque = deque()  # (deliver_time, sender_id, data)
        self.bytes_sent = 0
        self.bytes_dropped = 0
        self.packets_sent = 0
        self.packets_dropped = 0
        self.msg_size = 64  # bytes per message

    def send(self, sender_id: int, data: dict, t: float):
        """Broadcast a message from sender_id."""
        self.packets_sent += 1
        self.bytes_sent += self.msg_size

        # Packet drop
        if self.rng.random() < self.drop_rate:
            self.packets_dropped += 1
            self.bytes_dropped += self.msg_size
            return

        deliver_time = t + self.latency_s
        self._buffer.append((deliver_time, sender_id, data.copy()))

    def receive(self, receiver_id: int, t: float) -> Dict[int, dict]:
        """
        Receive all messages available at time t (delivered and not from self).
        Returns {sender_id: data} for the latest message from each sender.
        """
        ready = {}
        remaining = deque()
        for item in self._buffer:
            deliver_time, sender_id, data = item
            if deliver_time <= t and sender_id != receiver_id:
                ready[sender_id] = data  # latest from each sender
            elif deliver_time > t:
                remaining.append(item)
            # else: already delivered, discard
        self._buffer = remaining
        return ready

    def reset(self):
        self._buffer.clear()
        self.bytes_sent = 0
        self.bytes_dropped = 0
        self.packets_sent = 0
        self.packets_dropped = 0

    @property
    def bytes_per_second(self) -> float:
        """Average bytes/s (call at end of sim with known duration)."""
        return self.bytes_sent  # caller divides by duration
