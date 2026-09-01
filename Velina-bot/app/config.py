"""Application configuration loaded from the environment.

This module deliberately contains no bot logic. Secrets are read from the
environment and are never stored in source code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


__all__ = [
    "DISCORD_TOKEN", "TARGET_CHANNEL_IDS", "DISABLED_CHANNEL_IDS", "OWNER_IDS",
    "OLLAMA_MODEL", "OLLAMA_URL", "OLLAMA_KEEP_ALIVE",
    "WHISPER_MODEL_SIZE", "WHISPER_DEVICE", "WHISPER_COMPUTE_TYPE",
    "AZURE_TTS_API_KEY", "AZURE_TTS_REGION", "TTS_VOICE", "TTS_RATE",
    "TTS_VOLUME", "TTS_PITCH", "TTS_PROVIDER", "MAX_HISTORY", "SUMMARY_TRIGGER_THRESHOLD",
    "SUMMARY_HARD_CAP", "MAX_PERSONA_LENGTH", "MAX_PERSONA_NAME_LENGTH",
    "MAX_REPLY_WORDS", "PERSONAS_FILE", "STATE_FILE", "VOICE_SILENCE_SECONDS",
    "VOICE_SAMPLE_RATE", "VOICE_MIN_UTTERANCE_BYTES",
]


PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
load_dotenv(ROOT_DIR / ".env")


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"ERROR: {name} is missing. Check your .env file.")
    return value


def _id_set(value: str) -> set[int]:
    return {int(part.strip()) for part in value.split(",") if part.strip()}


DISCORD_TOKEN = _required("DISCORD_TOKEN")
TARGET_CHANNEL_IDS_STR = os.getenv("TARGET_CHANNEL_IDS") or os.getenv("TARGET_CHANNEL_ID")
if not TARGET_CHANNEL_IDS_STR:
    sys.exit("ERROR: TARGET_CHANNEL_IDS is missing. Check your .env file.")

TARGET_CHANNEL_IDS = _id_set(TARGET_CHANNEL_IDS_STR)
DISABLED_CHANNEL_IDS = _id_set(os.getenv("DISABLED_CHANNELS", ""))

OWNER_IDS = _id_set(os.getenv("OWNER_IDS") or os.getenv("OWNER_ID", "0")) - {0}

OLLAMA_MODEL = _required("OLLAMA_MODEL")
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

AZURE_TTS_API_KEY = os.getenv("AZURE_TTS_API_KEY")
AZURE_TTS_REGION = os.getenv("AZURE_TTS_REGION")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AshleyNeural")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "azure").strip().lower()
TTS_RATE = os.getenv("TTS_RATE", "+0%")
TTS_VOLUME = os.getenv("TTS_VOLUME", "+0%")
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz")

MAX_HISTORY = 10
SUMMARY_TRIGGER_THRESHOLD = MAX_HISTORY + 6
SUMMARY_HARD_CAP = SUMMARY_TRIGGER_THRESHOLD * 2
MAX_PERSONA_LENGTH = 1900
MAX_PERSONA_NAME_LENGTH = 32
MAX_REPLY_WORDS = 150

PERSONAS_FILE = str(PROJECT_DIR / "personas.json")
STATE_FILE = str(PROJECT_DIR / "bot_state.json")

VOICE_SILENCE_SECONDS = 0.7
VOICE_SAMPLE_RATE = 48000
VOICE_MIN_UTTERANCE_BYTES = int(VOICE_SAMPLE_RATE * 2 * 2 * 0.5)
