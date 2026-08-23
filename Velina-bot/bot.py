import os
import sys
import json
import logging
import random
import re
import time

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
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
OWNER_ID = int(os.getenv("OWNER_ID", "0"))  # set your Discord user ID in .env

DEFAULT_SYSTEM_PROMPT = ""
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
MAX_HISTORY = 10  # number of past messages to remember per channel

# Discord messages cap out at 2000 chars. The /persona command echoes the
# current persona back wrapped in a ```code block```` (6 chars of backticks
# + newlines), so we cap stored personas comfortably under that ceiling.
MAX_PERSONA_LENGTH = 1900

# Named persona presets are stored on disk (not in memory) so they survive
# bot restarts and cost effectively zero RAM when not actively loaded.
PERSONAS_FILE = "personas.json"
MAX_PERSONA_NAME_LENGTH = 32

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
    if SYSTEM_PROMPT:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

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
    combined_system = f"{SYSTEM_PROMPT}\n\n{instruction}" if SYSTEM_PROMPT else instruction
    messages = [
        {"role": "system", "content": combined_system},
        {"role": "user", "content": "(Begin the conversation now.)"},
    ]
    reply = await call_ollama(messages)
    conversation_history[channel_id] = [{"role": "assistant", "content": reply}]
    return reply


def owner_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if OWNER_ID == 0:
            await interaction.response.send_message(
                "-- OWNER_ID is not configured in .env, admin commands are disabled. --",
                ephemeral=True,
            )
            return False
        if interaction.user.id != OWNER_ID:
            await interaction.response.send_message(
                "-- This command is restricted to the bot owner. --", ephemeral=True
            )
            return False
        return True
    return app_commands.check(predicate)


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

@bot.tree.command(name="reset", description="Clear conversation history and reset persona")
async def reset_command(interaction: discord.Interaction):
    global SYSTEM_PROMPT
    channel_id = await require_channel_id(interaction)
    if channel_id is None:
        return
    conversation_history[channel_id] = []
    SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
    await interaction.response.send_message("-- Conversation history cleared and persona reset. --")


@bot.tree.command(name="forget", description="Clear conversation history only, keep the current persona")
async def forget_command(interaction: discord.Interaction):
    channel_id = await require_channel_id(interaction)
    if channel_id is None:
        return
    conversation_history[channel_id] = []
    await interaction.response.send_message("-- Conversation history cleared. Persona unchanged. --")


@bot.tree.command(name="persona", description="View or temporarily change Velina's persona")
@app_commands.describe(
    new_prompt=f"New system prompt, max {MAX_PERSONA_LENGTH} chars "
               "(Warning: This is effectively a reset, the bot will forget the old persona and conversation history)"
)
async def persona_command(
    interaction: discord.Interaction,
    new_prompt: Optional[app_commands.Range[str, 1, MAX_PERSONA_LENGTH]] = None,
):
    global SYSTEM_PROMPT

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

    if new_prompt is None:
        current = SYSTEM_PROMPT if SYSTEM_PROMPT else "(empty)"
        display = current
        if len(display) > 1900:
            display = display[:1900] + "\n... (truncated for display)"
        await interaction.response.send_message(f"Current persona:\n```{display}```", ephemeral=True)
    else:
        channel_id = await require_channel_id(interaction)
        if channel_id is None:
            return
        SYSTEM_PROMPT = new_prompt
        conversation_history[channel_id] = []  # clear so old tone doesn't linger
        await interaction.response.send_message("-- Persona updated and history cleared. --")


@bot.tree.command(name="save_persona", description="Save the current persona to disk under a name")
@app_commands.describe(name=f"Name for this preset, max {MAX_PERSONA_NAME_LENGTH} chars")
async def save_persona_command(
    interaction: discord.Interaction,
    name: app_commands.Range[str, 1, MAX_PERSONA_NAME_LENGTH],
):
    if not SYSTEM_PROMPT:
        await interaction.response.send_message(
            "-- Current persona is empty, nothing to save. Set one with /persona first. --",
            ephemeral=True,
        )
        return

    key = name.strip().lower()
    personas = load_personas()
    is_overwrite = key in personas
    personas[key] = SYSTEM_PROMPT
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


@bot.tree.command(name="load_persona", description="Load a saved persona preset by name")
@app_commands.describe(name="Name of the saved preset to load")
async def load_persona_command(interaction: discord.Interaction, name: str):
    global SYSTEM_PROMPT

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
    SYSTEM_PROMPT = personas[key]
    conversation_history[channel_id] = []  # clear so old tone doesn't linger
    await interaction.response.send_message(f"-- Loaded persona preset '{key}' and cleared history. --")


@bot.tree.command(name="list_personas", description="List all saved persona presets")
async def list_personas_command(interaction: discord.Interaction):
    personas = load_personas()
    if not personas:
        await interaction.response.send_message("-- No saved persona presets yet. --", ephemeral=True)
        return

    names = ", ".join(sorted(personas.keys()))
    await interaction.response.send_message(f"Saved persona presets: {names}", ephemeral=True)


@bot.tree.command(name="undo", description="Remove the last message exchange from history")
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
    if SYSTEM_PROMPT:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

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
        f"**Target channels:** {', '.join(f'<#{cid}>' for cid in TARGET_CHANNEL_IDS)}",
    ]
    await interaction.response.send_message("\n".join(lines), ephemeral=True)


@bot.tree.command(name="ping", description="Check if the bot is responsive")
async def ping_command(interaction: discord.Interaction):
    await interaction.response.send_message(f"Pong! Latency: {round(bot.latency * 1000)}ms")


PUBLIC_COMMANDS = [
    ("/reset", "Clear conversation history AND reset the persona back to default."),
    ("/forget", "Clear conversation history only. Persona stays as-is."),
    ("/persona [new_prompt]", "View the current persona (no argument), or set a new one (also clears history)."),
    ("/save_persona <name>", "Save the current persona to disk under a name for later reuse."),
    ("/load_persona <name>", "Load a previously saved persona preset (also clears history)."),
    ("/list_personas", "List all saved persona preset names."),
    ("/undo", "Remove the last message exchange from history."),
    ("/regenerate", "Re-run the last message to get a different response."),
    ("/start", "Have the bot start a brand new, randomly-styled conversation instead of waiting for you to speak first."),
    ("/history [count]", "Show the recent conversation history the bot is holding (max 10)."),
    ("/stats", "Show bot status: uptime, latency, messages processed, active model, target channel."),
    ("/ping", "Check if the bot is responsive."),
    ("/help", "Show this list."),
]

OWNER_COMMANDS = [
    ("/setchannel <channel>", "Add a channel to the bot's active channel list."),
    ("/removechannel <channel>", "Remove a channel from the bot's active channel list."),
    ("/listchannels", "List channels the bot is currently active in."),
    ("/setmodel <model_name>", "Change the Ollama model in use."),
    ("/pause", "Stop the bot from responding in the target channel(s)."),
    ("/resume", "Resume the bot responding in the target channel(s)."),
]


@bot.tree.command(name="help", description="List available commands and what they do")
async def help_command(interaction: discord.Interaction):
    lines = ["**Available commands:**"]
    for name, desc in PUBLIC_COMMANDS:
        lines.append(f"`{name}` — {desc}")

    if OWNER_ID != 0 and interaction.user.id == OWNER_ID:
        lines.append("")
        lines.append("**Owner-only commands:**")
        for name, desc in OWNER_COMMANDS:
            lines.append(f"`{name}` — {desc}")

    display = "\n".join(lines)
    if len(display) > 1900:
        display = display[:1900] + "\n... (truncated for display)"
    await interaction.response.send_message(display, ephemeral=True)


# ---------- OWNER-ONLY ADMIN COMMANDS ----------

@bot.tree.command(name="setchannel", description="[Owner only] Add a channel the bot listens/replies in")
@owner_only()
@app_commands.describe(channel="The channel to add to the bot's active channel list")
async def setchannel_command(interaction: discord.Interaction, channel: discord.TextChannel):
    TARGET_CHANNEL_IDS.add(channel.id)
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