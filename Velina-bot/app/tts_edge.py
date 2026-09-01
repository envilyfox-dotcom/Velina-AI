"""Microsoft Edge online TTS provider."""

from __future__ import annotations

import asyncio
import os
import tempfile

import edge_tts
from discord import FFmpegPCMAudio


async def speak(voice_client, text: str, voice: str, rate: str, volume: str, pitch: str, **_kwargs):
    while voice_client.is_playing():
        await asyncio.sleep(0.15)
    output_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    try:
        communicate = edge_tts.Communicate(
            text[:3000], voice.strip(), rate=rate, volume=volume, pitch=pitch
        )
        await communicate.save(output_path)
        voice_client.play(FFmpegPCMAudio(output_path))
        while voice_client.is_playing():
            await asyncio.sleep(0.15)
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass
