import os
import sys
import asyncio
import json
import logging
import random
import re
import tempfile
import time
import wave

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands, voice_recv
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import edge_tts
from typing import Optional

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
TARGET_CHANNEL_IDS_STR = os.getenv("TARGET_CHANNEL_IDS") or os.getenv("TARGET_CHANNEL_ID")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "Velina-V1")
OLLAMA_URL = "http://localhost:11434/api/chat"

if not DISCORD_TOKEN:
    sys.exit("ERROR: DISCORD_TOKEN is missing. Check your .env file.")
if not TARGET_CHANNEL_IDS_STR:
    sys.exit("ERROR: TARGET_CHANNEL_IDS is missing. Check your .env file.")

# Comma-separated list of channel IDs the bot listens/replies in, e.g.
#   TARGET_CHANNEL_IDS=123456789012345678,987654321098765432
# This lets you keep a private test-server channel and your main channel
# both active at once. Whitespace around commas is ignored.
TARGET_CHANNEL_IDS = {
    int(part.strip()) for part in TARGET_CHANNEL_IDS_STR.split(",") if part.strip()
}
# Comma-separated list of Discord user IDs treated as bot owners, e.g.
#   OWNER_IDS=123456789012345678,987654321098765432
# Useful if you control the bot from more than one Discord account.
# OWNER_ID (singular) is still read for backwards compatibility with older
# .env files; if both are set, OWNER_IDS wins.
_OWNER_IDS_STR = os.getenv("OWNER_IDS") or os.getenv("OWNER_ID", "0")
OWNER_IDS = {
    int(part.strip())
    for part in _OWNER_IDS_STR.split(",")
    if part.strip() and part.strip() != "0"
}

# Guild IDs derived from TARGET_CHANNEL_IDS at runtime. Public commands are
# refused outside these guilds, so even if the bot ends up installed
# somewhere unauthorized (e.g. via "Add App" / user install), it won't let
# outsiders change shared state like SYSTEM_PROMPT that affects every
# server the bot is in. This is a backstop -- the real fix is disabling
# "Public Bot" and "User Install" in the Discord Developer Portal so the
# bot can't be added anywhere else in the first place.
ALLOWED_GUILD_IDS: set = set()

DEFAULT_SYSTEM_PROMPT = ""
# Per-channel personas. Missing key == DEFAULT_SYSTEM_PROMPT for that
# channel. Keyed by channel_id so that testing a persona in one channel
# (e.g. a private test server) never affects any other channel the bot is
# active in.
CHANNEL_PERSONAS: dict[int, str] = {}
MAX_HISTORY = 10  # number of past messages to remember per channel

# Discord messages cap out at 2000 chars. The /persona command echoes the
# current persona back wrapped in a ```code block```` (6 chars of backticks
# + newlines), so we cap stored personas comfortably under that ceiling.
MAX_PERSONA_LENGTH = 1900

# Named persona presets are stored on disk (not in memory) so they survive
# bot restarts and cost effectively zero RAM when not actively loaded.
PERSONAS_FILE = "personas.json"
MAX_PERSONA_NAME_LENGTH = 32

# --- Voice (speech-to-text / text-to-speech) config ---
# STT runs locally via faster-whisper. Model is lazy-loaded on first /join,
# not at startup, so it costs zero RAM until someone actually uses voice.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")  # tiny/base/small/medium/large
# TTS runs via free edge-tts (no API key needed -- uses the same cloud
# voices as Microsoft Edge's "Read Aloud" feature). Full voice list: run
# `edge-tts --list-voices`, or see https://github.com/rany2/edge-tts
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")
# Speed/volume/pitch tweaks. Format matches edge-tts's own syntax and also
# works as Azure SSML <prosody> values:
#   rate/volume: signed percentage, e.g. "+15%" or "-10%"
#   pitch: signed Hz, e.g. "+20Hz" or "-15Hz"
# These are mutable at runtime via /voice_settings, not just .env.
TTS_RATE = os.getenv("TTS_RATE", "+0%")
TTS_VOLUME = os.getenv("TTS_VOLUME", "+0%")
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz")
# How long a user has to go silent before we treat their utterance as
# finished and send it off for transcription.
VOICE_SILENCE_SECONDS = 1.2
# Discord voice PCM is 48kHz, 16-bit, stereo. This sets a minimum buffered
# duration (in bytes) before we bother transcribing, to filter out noise
# blips / accidental key taps.
VOICE_SAMPLE_RATE = 48000
VOICE_MIN_UTTERANCE_BYTES = int(VOICE_SAMPLE_RATE * 2 * 2 * 0.5)  # ~0.5s

# Whether the bot is currently responding to messages in the target channel.
PAUSED = False

# Filled in on_ready.
bot_start_time: Optional[float] = None
message_count = 0

# Style directives for /start. A random one is picked each time so the
# opener is never the same canned greeting twice in a row. These are
# instructions to the model, never shown to users directly.
STARTER_STYLES = [
    "Bring up a strange hypothetical scenario and ask what the reader would do.",
    "Share a surprising or little-known fact and react to it briefly.",
    "Muse out loud about something ordinary as if noticing it for the first time.",
    "Ask an unusual would-you-rather question.",
    "Start mid-thought, as if continuing an internal monologue out loud.",
    "Make an oddly specific observation about the current moment.",
    "Pose a short, playful riddle or puzzle.",
    "Recount a tiny, invented daydream in one or two sentences.",
    "Bring up a half-finished thought about a random topic and invite the reader to weigh in.",
    "Confess a small, harmless, made-up opinion about something trivial.",
]

logging.basicConfig(level=logging.INFO)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# simple in-memory conversation history per channel
conversation_history = {}


def get_persona(channel_id: int) -> str:
    """Returns the persona in effect for a given channel: the channel's
    own override if one has been set, otherwise the global default."""
    return CHANNEL_PERSONAS.get(channel_id, DEFAULT_SYSTEM_PROMPT)


def load_personas() -> dict:
    if not os.path.exists(PERSONAS_FILE):
        return {}
    try:
        with open(PERSONAS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logging.exception("Failed to read %s, treating as empty", PERSONAS_FILE)
        return {}


def save_personas(personas: dict) -> None:
    with open(PERSONAS_FILE, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)


def clean_reply(reply: str) -> str:
    match = re.search(r'\n[\w\s]{1,32}:\s', reply)
    if match:
        reply = reply[:match.start()]
    return reply.strip()


async def call_ollama(messages: list) -> str:
    """Low-level call to Ollama given a fully-built messages list. Does not
    touch conversation_history itself -- callers are responsible for that,
    since different commands (normal chat, /regenerate, /start) build and
    update history differently."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Ollama error {resp.status}: {text}")
            data = await resp.json()
            return clean_reply(data["message"]["content"])


async def query_ollama(channel_id: int, user_message: str) -> str:
    history = conversation_history.setdefault(channel_id, [])
    history.append({"role": "user", "content": user_message})

    messages = history[-MAX_HISTORY:]
    persona = get_persona(channel_id)
    if persona:
        messages = [{"role": "system", "content": persona}] + messages

    reply = await call_ollama(messages)

    history.append({"role": "assistant", "content": reply})
    conversation_history[channel_id] = history[-MAX_HISTORY:]
    return reply


async def generate_starter(channel_id: int) -> str:
    """Generate a random, non-generic conversation opener and seed the
    channel's history with it as the first assistant turn."""
    style = random.choice(STARTER_STYLES)
    instruction = (
        "You are about to start a brand new conversation by speaking first, "
        "before the other person has said anything. "
        f"Style for this message: {style} "
        "Do not use a generic greeting like 'Hi' or 'Hello there' and do not "
        "ask 'how are you'. Respond with ONLY the message itself: no preamble, "
        "no explanation, no quotation marks, 1-3 sentences."
    )
    persona = get_persona(channel_id)
    combined_system = f"{persona}\n\n{instruction}" if persona else instruction
    messages = [
        {"role": "system", "content": combined_system},
        {"role": "user", "content": "(Begin the conversation now.)"},
    ]
    reply = await call_ollama(messages)
    conversation_history[channel_id] = [{"role": "assistant", "content": reply}]
    return reply


def owner_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if not OWNER_IDS:
            await interaction.response.send_message(
                "-- OWNER_IDS is not configured in .env, admin commands are disabled. --",
                ephemeral=True,
            )
            return False
        if interaction.user.id not in OWNER_IDS:
            await interaction.response.send_message(
                "-- This command is restricted to the bot owner. --", ephemeral=True
            )
            return False
        return True
    return app_commands.check(predicate)


async def refresh_allowed_guilds():
    """Recompute ALLOWED_GUILD_IDS from the guilds that TARGET_CHANNEL_IDS
    currently belong to. Call this after startup and whenever
    TARGET_CHANNEL_IDS changes (setchannel/removechannel)."""
    global ALLOWED_GUILD_IDS
    guild_ids = set()
    for cid in TARGET_CHANNEL_IDS:
        channel = bot.get_channel(cid)
        if channel is None:
            try:
                channel = await bot.fetch_channel(cid)
            except discord.HTTPException:
                logging.warning("Could not resolve channel %s to a guild", cid)
                continue
        guild = getattr(channel, "guild", None)
        if guild is not None:
            guild_ids.add(guild.id)
    ALLOWED_GUILD_IDS = guild_ids
    logging.info("Allowed guilds: %s", ALLOWED_GUILD_IDS)


def guild_allowed():
    """Refuses public commands run outside ALLOWED_GUILD_IDS. This is a
    backstop against the bot being added to unauthorized servers (e.g. via
    'Add App' / user install) -- without it, anyone who can invoke a public
    command from anywhere could change shared state that affects every
    server the bot is in. Bot owners always pass, since they're trusted
    regardless of which server they're testing from."""
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.user.id in OWNER_IDS:
            return True
        if interaction.guild_id is None or interaction.guild_id not in ALLOWED_GUILD_IDS:
            await interaction.response.send_message(
                "-- This bot isn't set up for use in this server. --", ephemeral=True
            )
            return False
        return True
    return app_commands.check(predicate)


def get_messageable_channel(channel_id: int) -> Optional[discord.abc.Messageable]:
    """bot.get_channel() returns a broad union that includes channel types
    without a .send() method (ForumChannel, CategoryChannel, etc.). This
    narrows to only channel-like objects that actually support sending,
    returning None otherwise so callers can't accidentally call .send()
    on something that doesn't have it."""
    channel = bot.get_channel(channel_id)
    if isinstance(channel, discord.abc.Messageable):
        return channel
    return None


async def send_chunked(sendable, text: str):
    """Send text in <=2000 char chunks using either an interaction followup
    or a text channel, whichever is passed in."""
    for i in range(0, len(text), 2000):
        await sendable.send(text[i:i + 2000])


async def require_channel_id(interaction: discord.Interaction) -> Optional[int]:
    """interaction.channel_id is typed as int | None by discord.py. In
    practice it's always set for our commands (they only ever run in real
    guild text channels), but we guard explicitly rather than assume, so
    both the type checker and runtime are satisfied. Sends an ephemeral
    error and returns None if it's ever missing; callers should bail out
    when this returns None."""
    if interaction.channel_id is None:
        await interaction.response.send_message(
            "-- This command needs to be used in a text channel. --", ephemeral=True
        )
        return None
    return interaction.channel_id


# ---------- VOICE (speech-to-text / text-to-speech) ----------

_whisper_model = None  # lazy-loaded, see get_whisper_model()


def get_whisper_model():
    """Loads the local faster-whisper model on first use only, so idle
    RAM stays untouched until someone actually joins voice."""
    global _whisper_model
    if _whisper_model is None:
        logging.info("Loading faster-whisper model '%s' (first use)...", WHISPER_MODEL_SIZE)
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        logging.info("faster-whisper model loaded.")
    return _whisper_model


class VoiceSession:
    """Per-guild state for an active voice connection: the discord voice
    client, per-user PCM buffers being accumulated, and a background task
    that watches for silence to know when an utterance is "done"."""

    def __init__(self, guild_id: int, text_channel_id: int):
        self.guild_id = guild_id
        self.text_channel_id = text_channel_id  # drives persona/history + where transcripts get posted
        self.voice_client: Optional[voice_recv.VoiceRecvClient] = None
        self.buffers: dict[int, bytearray] = {}
        self.last_write: dict[int, float] = {}
        self.watcher_task: Optional[asyncio.Task] = None


voice_sessions: dict[int, VoiceSession] = {}  # guild_id -> VoiceSession


def _voice_write_callback(guild_id: int):
    """Runs on discord-ext-voice-recv's receiver thread, not the event
    loop. Kept intentionally trivial (just buffer bytes + timestamp) per
    the library's own warning that heavy work here will cause problems."""
    def callback(user, data: voice_recv.VoiceData):
        if user is None or user.bot:
            return
        session = voice_sessions.get(guild_id)
        if session is None:
            return
        buf = session.buffers.setdefault(user.id, bytearray())
        buf.extend(data.pcm)
        session.last_write[user.id] = time.monotonic()
    return callback


async def _transcribe_and_respond(guild_id: int, user_id: int, pcm_bytes: bytes):
    session = voice_sessions.get(guild_id)
    if session is None:
        return

    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(VOICE_SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

    loop = asyncio.get_running_loop()
    try:
        def _run_transcription():
            model = get_whisper_model()
            segments, _info = model.transcribe(wav_path, beam_size=1)
            return " ".join(seg.text for seg in segments).strip()

        text = await loop.run_in_executor(None, _run_transcription)
    except Exception:
        logging.exception("STT transcription failed")
        return
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    if not text:
        return  # silence/noise, nothing was actually said

    text_channel = get_messageable_channel(session.text_channel_id)

    try:
        reply = await query_ollama(session.text_channel_id, text)
    except Exception as e:
        logging.exception("Error querying Ollama from voice input")
        if text_channel is not None:
            await text_channel.send(f"⚠️ Error generating response: {e}")
        return

    if text_channel is not None:
        guild = bot.get_guild(guild_id)
        member = guild.get_member(user_id) if guild is not None else None
        speaker = member.display_name if member else f"User {user_id}"
        await send_chunked(text_channel, f"🎙️ **{speaker}:** {text}\n{reply}")

    await _speak(session, reply)


async def _speak(session: VoiceSession, text: str):
    if session.voice_client is None or not session.voice_client.is_connected():
        return

    mp3_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    try:
        # edge-tts is natively async (it streams over a websocket), so no
        # executor thread is needed here.
        communicate = edge_tts.Communicate(
            text[:3000], voice=TTS_VOICE, rate=TTS_RATE, volume=TTS_VOLUME, pitch=TTS_PITCH
        )
        await communicate.save(mp3_path)
    except Exception:
        logging.exception("TTS generation failed")
        try:
            os.remove(mp3_path)
        except OSError:
            pass
        return

    # Wait for any current playback to finish rather than overlapping.
    while session.voice_client.is_playing():
        await asyncio.sleep(0.3)

    def _cleanup(error):
        if error:
            logging.error("Voice playback error: %s", error)
        try:
            os.remove(mp3_path)
        except OSError:
            pass

    session.voice_client.play(discord.FFmpegPCMAudio(mp3_path), after=_cleanup)


async def _silence_watcher(guild_id: int):
    """Background loop: for each user with buffered audio, once they've
    been silent for VOICE_SILENCE_SECONDS, treat the utterance as complete
    and hand it off for transcription."""
    while True:
        session = voice_sessions.get(guild_id)
        if session is None or session.voice_client is None or not session.voice_client.is_connected():
            return
        now = time.monotonic()
        for user_id in list(session.buffers.keys()):
            buf = session.buffers.get(user_id)
            last = session.last_write.get(user_id, 0)
            if buf and len(buf) >= VOICE_MIN_UTTERANCE_BYTES and (now - last) > VOICE_SILENCE_SECONDS:
                pcm_bytes = bytes(buf)
                session.buffers[user_id] = bytearray()
                asyncio.create_task(_transcribe_and_respond(guild_id, user_id, pcm_bytes))
            elif buf and (now - last) > VOICE_SILENCE_SECONDS:
                # Too short to be real speech -- discard instead of accumulating forever.
                session.buffers[user_id] = bytearray()
        await asyncio.sleep(0.5)


@bot.event
async def on_ready():
    global bot_start_time
    bot_start_time = time.monotonic()
    if bot.user is not None:
        print(f"Logged in as {bot.user} (id: {bot.user.id})")
    else:
        print("Logged in, but bot.user is unexpectedly None")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} slash command(s) globally")
        # NOTE: global command syncs can take up to an hour to reach all
        # clients. While testing changes, sync to a specific guild instead
        # for near-instant updates:
        #
        #   guild = discord.Object(id=YOUR_TEST_GUILD_ID)
        #   bot.tree.copy_global_to(guild=guild)
        #   synced = await bot.tree.sync(guild=guild)
        #
        # This makes sure you're actually testing the new command schema
        # and not a stale cached version.
    except Exception:
        logging.exception("Failed to sync slash commands")

    await refresh_allowed_guilds()


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    # Safety net so a bug in any slash command never surfaces to users as a
    # silent "The application did not respond" timeout.
    if isinstance(error, app_commands.CheckFailure):
        # owner_only() predicate already sent a response; nothing more to do.
        return
    logging.exception("Slash command error", exc_info=error)
    message = "-- Something went wrong running that command. --"
    try:
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
    except discord.HTTPException:
        pass


@bot.event
async def on_message(message: discord.Message):
    global message_count

    if message.author.bot:
        return
    if message.channel.id not in TARGET_CHANNEL_IDS:
        return
    if not message.content.strip():
        return
    if PAUSED:
        return

    async with message.channel.typing():
        try:
            reply = await query_ollama(message.channel.id, message.content)
        except Exception as e:
            logging.exception("Error querying Ollama")
            await message.channel.send(f"⚠️ Error generating response: {e}")
            return

    message_count += 1
    await send_chunked(message.channel, reply)
    await bot.process_commands(message)


# ---------- SLASH COMMANDS ----------

@bot.tree.command(name="reset", description="Clear conversation history and reset this channel's persona")
@guild_allowed()
async def reset_command(interaction: discord.Interaction):
    channel_id = await require_channel_id(interaction)
    if channel_id is None:
        return
    conversation_history[channel_id] = []
    CHANNEL_PERSONAS.pop(channel_id, None)
    await interaction.response.send_message("-- Conversation history cleared and persona reset for this channel. --")


@bot.tree.command(name="forget", description="Clear conversation history only, keep the current persona")
@guild_allowed()
async def forget_command(interaction: discord.Interaction):
    channel_id = await require_channel_id(interaction)
    if channel_id is None:
        return
    conversation_history[channel_id] = []
    await interaction.response.send_message("-- Conversation history cleared. Persona unchanged. --")


@bot.tree.command(name="persona", description="View or temporarily change Velina's persona for this channel")
@guild_allowed()
@app_commands.describe(
    new_prompt=f"New system prompt for THIS channel, max {MAX_PERSONA_LENGTH} chars "
               "(Warning: This is effectively a reset, the bot will forget the old persona and conversation history for this channel)"
)
async def persona_command(
    interaction: discord.Interaction,
    new_prompt: Optional[app_commands.Range[str, 1, MAX_PERSONA_LENGTH]] = None,
):
    # Server-side enforcement. Never trust the client-side cap alone: stale
    # command sync, mobile clients, or paste events can all let an
    # oversized value reach the bot. This check guarantees we never store
    # something that could later blow past Discord's 2000-char message cap.
    if new_prompt is not None and len(new_prompt) > MAX_PERSONA_LENGTH:
        await interaction.response.send_message(
            f"-- That's {len(new_prompt)} characters, which is over the "
            f"{MAX_PERSONA_LENGTH}-character limit. Persona was NOT updated. "
            "Please shorten it and try again. --",
            ephemeral=True,
        )
        return

    channel_id = await require_channel_id(interaction)
    if channel_id is None:
        return

    if new_prompt is None:
        current = get_persona(channel_id) or "(empty)"
        display = current
        if len(display) > 1900:
            display = display[:1900] + "\n... (truncated for display)"
        await interaction.response.send_message(
            f"Current persona for this channel:\n```{display}```", ephemeral=True
        )
    else:
        CHANNEL_PERSONAS[channel_id] = new_prompt
        conversation_history[channel_id] = []  # clear so old tone doesn't linger
        await interaction.response.send_message("-- Persona updated and history cleared for this channel. --")


@bot.tree.command(name="save_persona", description="Save this channel's current persona to disk under a name")
@guild_allowed()
@app_commands.describe(name=f"Name for this preset, max {MAX_PERSONA_NAME_LENGTH} chars")
async def save_persona_command(
    interaction: discord.Interaction,
    name: app_commands.Range[str, 1, MAX_PERSONA_NAME_LENGTH],
):
    channel_id = await require_channel_id(interaction)
    if channel_id is None:
        return

    current_persona = get_persona(channel_id)
    if not current_persona:
        await interaction.response.send_message(
            "-- Current persona for this channel is empty, nothing to save. Set one with /persona first. --",
            ephemeral=True,
        )
        return

    key = name.strip().lower()
    personas = load_personas()
    is_overwrite = key in personas
    personas[key] = current_persona
    try:
        save_personas(personas)
    except OSError:
        logging.exception("Failed to write %s", PERSONAS_FILE)
        await interaction.response.send_message(
            "-- Failed to save persona to disk, check bot logs. --", ephemeral=True
        )
        return

    verb = "Overwrote" if is_overwrite else "Saved"
    await interaction.response.send_message(f"-- {verb} persona preset '{key}'. --", ephemeral=True)


@bot.tree.command(name="load_persona", description="Load a saved persona preset by name into this channel")
@guild_allowed()
@app_commands.describe(name="Name of the saved preset to load")
async def load_persona_command(interaction: discord.Interaction, name: str):
    key = name.strip().lower()
    personas = load_personas()
    if key not in personas:
        await interaction.response.send_message(
            f"-- No preset named '{key}' found. Use /list_personas to see saved presets. --",
            ephemeral=True,
        )
        return

    channel_id = await require_channel_id(interaction)
    if channel_id is None:
        return
    CHANNEL_PERSONAS[channel_id] = personas[key]
    conversation_history[channel_id] = []  # clear so old tone doesn't linger
    await interaction.response.send_message(f"-- Loaded persona preset '{key}' into this channel and cleared history. --")


@bot.tree.command(name="list_personas", description="List all saved persona presets")
@guild_allowed()
async def list_personas_command(interaction: discord.Interaction):
    personas = load_personas()
    if not personas:
        await interaction.response.send_message("-- No saved persona presets yet. --", ephemeral=True)
        return

    names = ", ".join(sorted(personas.keys()))
    await interaction.response.send_message(f"Saved persona presets: {names}", ephemeral=True)


@bot.tree.command(name="undo", description="Remove the last message exchange from history")
@guild_allowed()
async def undo_command(interaction: discord.Interaction):
    channel_id = await require_channel_id(interaction)
    if channel_id is None:
        return
    history = conversation_history.get(channel_id, [])
    removed = []
    if history and history[-1]["role"] == "assistant":
        removed.append(history.pop())
    if history and history[-1]["role"] == "user":
        removed.append(history.pop())
    conversation_history[channel_id] = history

    if not removed:
        await interaction.response.send_message("-- Nothing to undo. --", ephemeral=True)
    else:
        await interaction.response.send_message("-- Removed the last exchange from history. --", ephemeral=True)


@bot.tree.command(name="regenerate", description="Re-run the last message to get a different response")
@guild_allowed()
async def regenerate_command(interaction: discord.Interaction):
    global message_count

    channel_id = await require_channel_id(interaction)
    if channel_id is None:
        return

    history = conversation_history.get(channel_id, [])
    if history and history[-1]["role"] == "assistant":
        history.pop()
    if not history or history[-1]["role"] != "user":
        await interaction.response.send_message(
            "-- Nothing to regenerate yet, send a message first. --", ephemeral=True
        )
        return

    await interaction.response.defer()

    messages = history[-MAX_HISTORY:]
    persona = get_persona(channel_id)
    if persona:
        messages = [{"role": "system", "content": persona}] + messages

    try:
        reply = await call_ollama(messages)
    except Exception as e:
        logging.exception("Error querying Ollama during regenerate")
        await interaction.followup.send(f"⚠️ Error generating response: {e}")
        return

    history.append({"role": "assistant", "content": reply})
    conversation_history[channel_id] = history[-MAX_HISTORY:]
    message_count += 1
    await send_chunked(interaction.followup, reply)


@bot.tree.command(name="start", description="Have the bot start a brand new, random conversation")
@guild_allowed()
async def start_command(interaction: discord.Interaction):
    global message_count

    channel_id = await require_channel_id(interaction)
    if channel_id is None:
        return

    await interaction.response.defer()
    try:
        reply = await generate_starter(channel_id)
    except Exception as e:
        logging.exception("Error querying Ollama during start")
        await interaction.followup.send(f"⚠️ Error generating response: {e}")
        return

    message_count += 1
    await send_chunked(interaction.followup, reply)


@bot.tree.command(name="history", description="Show the recent conversation history the bot is holding")
@guild_allowed()
@app_commands.describe(count="How many recent messages to show (default 10, max 10)")
async def history_command(interaction: discord.Interaction, count: Optional[app_commands.Range[int, 1, MAX_HISTORY]] = None):
    channel_id = await require_channel_id(interaction)
    if channel_id is None:
        return
    history = conversation_history.get(channel_id, [])
    if not history:
        await interaction.response.send_message("-- No conversation history for this channel yet. --", ephemeral=True)
        return

    n = count or MAX_HISTORY
    recent = history[-n:]
    lines = [f"**{m['role']}:** {m['content']}" for m in recent]
    display = "\n".join(lines)
    if len(display) > 1900:
        display = display[:1900] + "\n... (truncated for display)"
    await interaction.response.send_message(display, ephemeral=True)


@bot.tree.command(name="stats", description="Show bot status and stats")
@guild_allowed()
async def stats_command(interaction: discord.Interaction):
    uptime_seconds = int(time.monotonic() - bot_start_time) if bot_start_time else 0
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"

    status = "PAUSED" if PAUSED else "active"
    lines = [
        f"**Status:** {status}",
        f"**Uptime:** {uptime_str}",
        f"**Latency:** {round(bot.latency * 1000)}ms",
        f"**Messages processed:** {message_count}",
        f"**Model:** {OLLAMA_MODEL}",
    ]
    await interaction.response.send_message("\n".join(lines), ephemeral=True)


@bot.tree.command(name="ping", description="Check if the bot is responsive")
@guild_allowed()
async def ping_command(interaction: discord.Interaction):
    await interaction.response.send_message(f"Pong! Latency: {round(bot.latency * 1000)}ms")


PUBLIC_COMMANDS = [
    ("/reset", "Clear conversation history AND reset this channel's persona back to default."),
    ("/forget", "Clear conversation history only. Persona stays as-is."),
    ("/persona [new_prompt]", "View this channel's current persona (no argument), or set a new one for this channel (also clears its history)."),
    ("/save_persona <name>", "Save this channel's current persona to disk under a name for later reuse."),
    ("/load_persona <name>", "Load a previously saved persona preset into this channel (also clears its history)."),
    ("/list_personas", "List all saved persona preset names."),
    ("/undo", "Remove the last message exchange from history."),
    ("/regenerate", "Re-run the last message to get a different response."),
    ("/start", "Have the bot start a brand new, randomly-styled conversation instead of waiting for you to speak first."),
    ("/history [count]", "Show the recent conversation history the bot is holding (max 10)."),
    ("/join", "Join your current voice channel and start listening/talking (speech-to-text + text-to-speech)."),
    ("/leave", "Leave the voice channel and stop listening."),
    ("/stats", "Show bot status: uptime, latency, messages processed, active model."),
    ("/ping", "Check if the bot is responsive."),
    ("/help", "Show this list."),
]

OWNER_COMMANDS = [
    ("/setchannel <channel>", "Add a channel to the bot's active channel list."),
    ("/removechannel <channel>", "Remove a channel from the bot's active channel list."),
    ("/listchannels", "List channels the bot is currently active in."),
    ("/setmodel <model_name>", "Change the Ollama model in use."),
    ("/announce [channel]", "Opens a form to type a multi-line, hand-formatted message and post it as the bot (bypasses the AI)."),
    ("/voice_settings [voice] [rate] [volume] [pitch]", "View or adjust TTS voice, speed, volume, and pitch."),
    ("/pause", "Stop the bot from responding in the target channel(s)."),
    ("/resume", "Resume the bot responding in the target channel(s)."),
]


@bot.tree.command(name="help", description="List available commands and what they do")
@guild_allowed()
async def help_command(interaction: discord.Interaction):
    lines = ["**Available commands:**"]
    for name, desc in PUBLIC_COMMANDS:
        lines.append(f"`{name}` — {desc}")

    if interaction.user.id in OWNER_IDS:
        lines.append("")
        lines.append("**Owner-only commands:**")
        for name, desc in OWNER_COMMANDS:
            lines.append(f"`{name}` — {desc}")

    display = "\n".join(lines)
    if len(display) > 1900:
        display = display[:1900] + "\n... (truncated for display)"
    await interaction.response.send_message(display, ephemeral=True)


# ---------- VOICE COMMANDS ----------

@bot.tree.command(name="join", description="Join your voice channel and start listening/talking")
@guild_allowed()
async def join_command(interaction: discord.Interaction):
    if interaction.guild is None:
        await interaction.response.send_message("-- This only works in a server. --", ephemeral=True)
        return

    member = interaction.guild.get_member(interaction.user.id)
    if member is None or member.voice is None or member.voice.channel is None:
        await interaction.response.send_message(
            "-- You need to be in a voice channel first. --", ephemeral=True
        )
        return

    channel_id = await require_channel_id(interaction)
    if channel_id is None:
        return

    await interaction.response.defer(ephemeral=True)

    guild_id = interaction.guild.id
    existing = voice_sessions.get(guild_id)
    if existing is not None and existing.voice_client is not None and existing.voice_client.is_connected():
        await interaction.followup.send("-- Already connected to a voice channel here. --", ephemeral=True)
        return

    try:
        voice_client = await member.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
    except Exception as e:
        logging.exception("Failed to connect to voice channel")
        await interaction.followup.send(f"⚠️ Couldn't join voice channel: {e}", ephemeral=True)
        return

    session = VoiceSession(guild_id=guild_id, text_channel_id=channel_id)
    session.voice_client = voice_client
    voice_sessions[guild_id] = session
    voice_client.listen(voice_recv.BasicSink(_voice_write_callback(guild_id)))
    session.watcher_task = asyncio.create_task(_silence_watcher(guild_id))

    # Loading the Whisper model can take a few seconds on first use; do it
    # now so the first real utterance doesn't stall.
    asyncio.get_running_loop().run_in_executor(None, get_whisper_model)

    await interaction.followup.send(
        f"-- Joined {member.voice.channel.mention}. Listening now. --", ephemeral=True
    )


@bot.tree.command(name="leave", description="Leave the voice channel and stop listening")
@guild_allowed()
async def leave_command(interaction: discord.Interaction):
    if interaction.guild is None:
        await interaction.response.send_message("-- This only works in a server. --", ephemeral=True)
        return

    guild_id = interaction.guild.id
    session = voice_sessions.get(guild_id)
    if session is None or session.voice_client is None:
        await interaction.response.send_message("-- Not currently in a voice channel. --", ephemeral=True)
        return

    if session.watcher_task is not None:
        session.watcher_task.cancel()
    await session.voice_client.disconnect(force=True)
    del voice_sessions[guild_id]

    await interaction.response.send_message("-- Left the voice channel. --", ephemeral=True)


# ---------- OWNER-ONLY ADMIN COMMANDS ----------

@bot.tree.command(name="setchannel", description="[Owner only] Add a channel the bot listens/replies in")
@owner_only()
@app_commands.describe(channel="The channel to add to the bot's active channel list")
async def setchannel_command(interaction: discord.Interaction, channel: discord.TextChannel):
    TARGET_CHANNEL_IDS.add(channel.id)
    await refresh_allowed_guilds()
    await interaction.response.send_message(f"-- Added {channel.mention} to active channels. --", ephemeral=True)


@bot.tree.command(name="removechannel", description="[Owner only] Remove a channel from the bot's active channel list")
@owner_only()
@app_commands.describe(channel="The channel to remove")
async def removechannel_command(interaction: discord.Interaction, channel: discord.TextChannel):
    if channel.id not in TARGET_CHANNEL_IDS:
        await interaction.response.send_message(
            f"-- {channel.mention} isn't in the active channel list. --", ephemeral=True
        )
        return
    if len(TARGET_CHANNEL_IDS) == 1:
        await interaction.response.send_message(
            "-- Can't remove the last active channel, the bot needs at least one. --", ephemeral=True
        )
        return
    TARGET_CHANNEL_IDS.discard(channel.id)
    await refresh_allowed_guilds()
    await interaction.response.send_message(f"-- Removed {channel.mention} from active channels. --", ephemeral=True)


@bot.tree.command(name="listchannels", description="[Owner only] List channels the bot is currently active in")
@owner_only()
async def listchannels_command(interaction: discord.Interaction):
    mentions = ", ".join(f"<#{cid}>" for cid in TARGET_CHANNEL_IDS)
    await interaction.response.send_message(f"Active channels: {mentions}", ephemeral=True)


@bot.tree.command(name="setmodel", description="[Owner only] Change the Ollama model in use")
@owner_only()
@app_commands.describe(model_name="Name of the Ollama model to switch to")
async def setmodel_command(interaction: discord.Interaction, model_name: str):
    global OLLAMA_MODEL
    OLLAMA_MODEL = model_name
    await interaction.response.send_message(f"-- Model set to '{model_name}'. --", ephemeral=True)


@bot.tree.command(name="voice_settings", description="[Owner only] View or adjust TTS voice, speed, volume, pitch")
@owner_only()
@app_commands.describe(
    voice="Voice name, e.g. en-US-AriaNeural (see Azure Voice Gallery to preview)",
    rate="Speed, e.g. +15% or -10%",
    volume="Volume, e.g. +10% or -20%",
    pitch="Pitch, e.g. +20Hz or -15Hz",
)
async def voice_settings_command(
    interaction: discord.Interaction,
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    volume: Optional[str] = None,
    pitch: Optional[str] = None,
):
    global TTS_VOICE, TTS_RATE, TTS_VOLUME, TTS_PITCH

    if voice is None and rate is None and volume is None and pitch is None:
        await interaction.response.send_message(
            f"**Current voice settings:**\n"
            f"Voice: `{TTS_VOICE}`\nRate: `{TTS_RATE}`\nVolume: `{TTS_VOLUME}`\nPitch: `{TTS_PITCH}`",
            ephemeral=True,
        )
        return

    if voice is not None:
        TTS_VOICE = voice
    if rate is not None:
        TTS_RATE = rate
    if volume is not None:
        TTS_VOLUME = volume
    if pitch is not None:
        TTS_PITCH = pitch

    await interaction.response.send_message(
        f"-- Voice settings updated: `{TTS_VOICE}`, rate `{TTS_RATE}`, "
        f"volume `{TTS_VOLUME}`, pitch `{TTS_PITCH}`. --",
        ephemeral=True,
    )


class AnnounceModal(discord.ui.Modal, title="Post Announcement"):
    """Discord slash-command text options are rendered as single-line boxes
    -- Enter/Shift+Enter can't add line breaks there. A modal's TextInput
    (paragraph style) is a real multi-line text area, so this is what
    actually lets you type multi-paragraph, formatted announcements."""

    message = discord.ui.TextInput(
        label="Message",
        style=discord.TextStyle.paragraph,
        placeholder="Markdown supported: **bold**, *italics*, # headers, etc. Enter = line break.",
        max_length=4000,
        required=True,
    )

    def __init__(self, target: discord.abc.Messageable, target_mention: Optional[str]):
        super().__init__()
        self.target = target
        self.target_mention = target_mention

    async def on_submit(self, interaction: discord.Interaction):
        await send_chunked(self.target, str(self.message))
        where = f" in {self.target_mention}" if self.target_mention else ""
        await interaction.response.send_message(f"-- Announcement posted{where}. --", ephemeral=True)


@bot.tree.command(name="announce", description="[Owner only] Post a raw, hand-written message as the bot (bypasses the AI)")
@owner_only()
@app_commands.describe(
    channel="Where to post it. Defaults to the channel you run this command in. "
            "Doesn't have to be one of the bot's active AI channels.",
)
async def announce_command(
    interaction: discord.Interaction,
    channel: Optional[discord.TextChannel] = None,
):
    target = channel or interaction.channel
    if not isinstance(target, discord.abc.Messageable):
        await interaction.response.send_message(
            "-- Can't post to that channel type. --", ephemeral=True
        )
        return

    # send_modal() must be the very first response to the interaction, so
    # no deferring/checks can happen after this point.
    await interaction.response.send_modal(AnnounceModal(target, channel.mention if channel is not None else None))


@bot.tree.command(name="pause", description="[Owner only] Stop the bot from responding in the target channel")
@owner_only()
async def pause_command(interaction: discord.Interaction):
    global PAUSED
    PAUSED = True
    await interaction.response.send_message("-- Bot paused. It will not respond until /resume is run. --")


@bot.tree.command(name="resume", description="[Owner only] Resume the bot responding in the target channel")
@owner_only()
async def resume_command(interaction: discord.Interaction):
    global PAUSED
    PAUSED = False
    await interaction.response.send_message("-- Bot resumed. --")


bot.run(DISCORD_TOKEN)