"""Provider selector for voice synthesis."""

from __future__ import annotations

async def speak(provider: str, voice_client, text: str, **settings):
    provider_name = (provider or "azure").strip().lower()
    if provider_name == "azure":
        from app import tts_azure
        implementation = tts_azure.speak
    elif provider_name == "edge":
        from app import tts_edge
        implementation = tts_edge.speak
    else:
        raise RuntimeError(f"Unsupported TTS_PROVIDER: {provider}")
    await implementation(voice_client, text, **settings)
