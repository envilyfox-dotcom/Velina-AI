"""Azure Speech TTS provider."""

from __future__ import annotations

import asyncio
import html
from typing import Any, cast

import azure.cognitiveservices.speech as speechsdk

from app.tts_common import PCMSource


class _PushCallback(speechsdk.audio.PushAudioOutputStreamCallback):
    def __init__(self, source: PCMSource):
        super().__init__()
        self.source = source

    def write(self, audio_buffer: memoryview) -> int:
        return self.source.push_mono_pcm(audio_buffer)

    def close(self):
        self.source.close()


async def speak(voice_client, text: str, voice: str, rate: str, volume: str, pitch: str, api_key: str, region: str):
    if not api_key or not region:
        raise RuntimeError("Azure TTS requires AZURE_TTS_API_KEY and AZURE_TTS_REGION")
    while voice_client.is_playing():
        await asyncio.sleep(0.15)

    voice_name = voice.strip()
    locale = "-".join(voice_name.split(":", 1)[0].split("-")[:2])
    ssml = (
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
        f"xml:lang='{html.escape(locale, quote=True)}'>"
        f"<voice name='{html.escape(voice_name, quote=True)}'>"
        f"<prosody rate='{html.escape(rate, quote=True)}' "
        f"volume='{html.escape(volume, quote=True)}' "
        f"pitch='{html.escape(pitch, quote=True)}'>"
        f"{html.escape(text[:3000])}</prosody></voice></speak>"
    )
    source = PCMSource()

    def synthesize():
        try:
            config = speechsdk.SpeechConfig(subscription=api_key, region=region)
            config.speech_synthesis_voice_name = voice_name
            config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm
            )
            stream = speechsdk.audio.PushAudioOutputStream(_PushCallback(source))
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=config,
                audio_config=speechsdk.audio.AudioOutputConfig(stream=stream),
            )
            result = cast(Any, synthesizer.speak_ssml_async(ssml).get())
            if result is None:
                raise RuntimeError("Azure TTS returned no synthesis result")
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                details = speechsdk.CancellationDetails(result)
                raise RuntimeError(f"Azure TTS failed: {details.reason}; {details.error_details}")
        finally:
            source.close()

    try:
        voice_client.play(source)
        await asyncio.get_running_loop().run_in_executor(None, synthesize)
    finally:
        source.close()
