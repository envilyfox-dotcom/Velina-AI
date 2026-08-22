import os
import sys
import logging
import re

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
TARGET_CHANNEL_ID_STR = os.getenv("TARGET_CHANNEL_ID")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "Velina-V1")
OLLAMA_URL = "http://localhost:11434/api/chat"

if not DISCORD_TOKEN:
    sys.exit("ERROR: DISCORD_TOKEN is missing. Check your .env file.")
if not TARGET_CHANNEL_ID_STR:
    sys.exit("ERROR: TARGET_CHANNEL_ID is missing. Check your .env file.")

TARGET_CHANNEL_ID = int(TARGET_CHANNEL_ID_STR)
OWNER_ID = int(os.getenv("OWNER_ID", "0"))  # set your Discord user ID in .env

DEFAULT_SYSTEM_PROMPT = ""
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
MAX_HISTORY = 10  # number of past messages to remember per channel

# Discord messages cap out at 2000 chars. The /persona command echoes the
# current persona back wrapped in a ```code block```` (6 chars of backticks
# + newlines), so we cap stored personas comfortably under that ceiling.
MAX_PERSONA_LENGTH = 1900

logging.basicConfig(level=logging.INFO)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# simple in-memory conversation history per channel
conversation_history = {}


def clean_reply(reply: str) -> str:
    match = re.search(r'\n[\w\s]{1,32}:\s', reply)
    if match:
        reply = reply[:match.start()]
    return reply.strip()


async def query_ollama(channel_id: int, user_message: str) -> str:
    history = conversation_history.setdefault(channel_id, [])
    history.append({"role": "user", "content": user_message})

    messages = history[-MAX_HISTORY:]
    if SYSTEM_PROMPT:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

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
            reply = clean_reply(data["message"]["content"])

    history.append({"role": "assistant", "content": reply})
    conversation_history[channel_id] = history[-MAX_HISTORY:]
    return reply


@bot.event
async def on_ready():
    if bot.user is not None:
        print(f"Logged in as {bot.user} (id: {bot.user.id})")
    else:
        print("Logged in, but bot.user is unexpectedly None")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} slash command(s) globally")
        # NOTE: global command syncs can take up to an hour to reach all
        # clients. While testing changes like the persona length cap, sync
        # to a specific guild instead for near-instant updates:
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
    if message.author.bot:
        return
    if message.channel.id != TARGET_CHANNEL_ID:
        return
    if not message.content.strip():
        return

    async with message.channel.typing():
        try:
            reply = await query_ollama(message.channel.id, message.content)
        except Exception as e:
            logging.exception("Error querying Ollama")
            await message.channel.send(f"⚠️ Error generating response: {e}")
            return

    for i in range(0, len(reply), 2000):
        await message.channel.send(reply[i:i + 2000])

    await bot.process_commands(message)


# ---------- SLASH COMMANDS ----------

@bot.tree.command(name="reset", description="Clear conversation history and reset persona")
async def reset_command(interaction: discord.Interaction):
    global SYSTEM_PROMPT
    conversation_history[interaction.channel_id] = []
    SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
    await interaction.response.send_message("-- Conversation history cleared and persona reset. --")


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
        # Defensive truncation for display too, in case SYSTEM_PROMPT was
        # ever set some other way (e.g. edited directly in code) and is
        # longer than what fits in a single Discord message.
        display = current
        if len(display) > 1900:
            display = display[:1900] + "\n... (truncated for display)"
        await interaction.response.send_message(f"Current persona:\n```{display}```", ephemeral=True)
    else:
        SYSTEM_PROMPT = new_prompt
        conversation_history[interaction.channel_id] = []  # clear so old tone doesn't linger
        await interaction.response.send_message("-- Persona updated and history cleared. --")


@bot.tree.command(name="ping", description="Check if the bot is responsive")
async def ping_command(interaction: discord.Interaction):
    await interaction.response.send_message(f"Pong! Latency: {round(bot.latency * 1000)}ms")


bot.run(DISCORD_TOKEN)