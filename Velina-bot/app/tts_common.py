"""Shared Discord PCM audio source used by streaming TTS providers."""

from __future__ import annotations

import queue
from typing import Optional

import discord


class PCMSource(discord.AudioSource):
    """Queue-backed 48 kHz stereo PCM source for Discord's voice player."""

    FRAME_BYTES = 3840

    def __init__(self):
        self._chunks: queue.Queue[Optional[bytes]] = queue.Queue()
        self._buffer = bytearray()
        self._close_signaled = False
        self._ended = False

    def push_mono_pcm(self, audio_buffer: memoryview) -> int:
        mono = bytes(audio_buffer)
        mono = mono[: len(mono) - (len(mono) % 2)]
        stereo = bytearray(len(mono) * 2)
        for offset in range(0, len(mono), 2):
            sample = mono[offset : offset + 2]
            stereo[offset * 2 : offset * 2 + 2] = sample
            stereo[offset * 2 + 2 : offset * 2 + 4] = sample
        self._chunks.put(bytes(stereo))
        return len(audio_buffer)

    def close(self):
        if not self._close_signaled:
            self._close_signaled = True
            self._chunks.put(None)

    def read(self) -> bytes:
        while not self._ended and len(self._buffer) < self.FRAME_BYTES:
            try:
                chunk = self._chunks.get(timeout=30)
            except queue.Empty:
                self._ended = True
                break
            if chunk is None:
                self._ended = True
                break
            self._buffer.extend(chunk)
        if not self._buffer:
            return b""
        frame = bytes(self._buffer[: self.FRAME_BYTES])
        del self._buffer[: self.FRAME_BYTES]
        return frame + b"\x00" * (self.FRAME_BYTES - len(frame))

    def is_opus(self) -> bool:
        return False

    def cleanup(self):
        self.close()
