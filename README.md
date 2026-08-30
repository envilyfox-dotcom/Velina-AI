# Velina AI

Velina is a personality-driven Discord bot powered by a fine-tuned **Meta
Llama 3.1 8B Instruct** model running through [Ollama](https://ollama.com).
The bot is designed for channel-based conversation, persistent local memory,
persona switching, and optional voice interaction.

## Features

- Fine-tuned personality and conversational style
- English and Indonesian tone switching
- Local Ollama text generation
- Per-channel conversation history and rolling summaries
- Persistent runtime state in `bot_state.json`
- Per-channel persona presets stored in `personas.json`
- Multiple-speaker channel support
- Prompt-injection filtering and user cooldowns
- 150-word limit for model-generated replies
- Optional speech-to-text with faster-whisper
- Optional text-to-speech with Edge TTS
- Owner-only administration and announcements

Normal AI messages and public AI commands work only in the configured target
channels. Channels listed in `DISABLED_CHANNELS` remain silent for normal AI
behavior. Owner-only announcements can intentionally bypass that disabled
list.

## Discord commands

### Public commands

| Command | Description |
|---|---|
| `/reset` | Clear this channel's history, summary, and active persona. |
| `/forget` | Clear this channel's history and summary while keeping its persona. |
| `/persona` | View or set the current persona for this channel. Setting one clears history. |
| `/save_persona <name>` | Save the current channel persona as a preset. |
| `/list_personas` | Privately show saved presets and choose one from a dropdown to load it. |
| `/delete_persona` | Owner-only dropdown to delete one saved preset. |
| `/undo` | Remove the last user/bot exchange. |
| `/regenerate` | Generate a different response to the last user message. |
| `/start` | Start a new randomly styled conversation. |
| `/history [count]` | Privately show recent stored history. |
| `/join` | Join the user's voice channel and enable voice interaction. |
| `/leave` | Leave the current voice channel. |
| `/stats` | Show status, uptime, latency, message count, and model. |
| `/ping` | Check bot latency. |
| `/help` | Show available commands. |

### Owner-only commands

Owner-only commands require a user ID listed in `OWNER_IDS`.

| Command | Description |
|---|---|
| `/setchannel <channel>` | Add a channel to the active target list at runtime. |
| `/removechannel <channel>` | Remove a channel from the active target list at runtime. |
| `/listchannels` | Show active target channels. |
| `/setmodel <model_name>` | Change the Ollama model and persist the override. |
| `/voice_settings` | View or change voice, rate, volume, and pitch. |
| `/announce channel [channel]` | Open a multiline modal for a manual announcement to one channel. |
| `/announce all` | Open a multiline modal for a manual announcement to all configured channels. |
| `/unmute <user>` | Clear an injection-attempt cooldown. |
| `/pause [channel]` | Pause normal AI responses in a channel. |
| `/resume [channel]` | Resume normal AI responses in a channel. |
| `/list_paused` | Show paused channels. |

`/announce` uses two subcommands:

- `/announce channel` posts to the selected channel, or to the channel where
  the command was used if no channel is selected.
- `/announce all` posts to every configured target channel, including channels
  listed in `DISABLED_CHANNELS`.

Announcements bypass the AI and are not added to conversation memory.

## Persistent state

The bot stores runtime state in `bot_state.json`, including:

- Recent conversation history
- Rolling channel summaries
- Active channel personas
- Paused channels
- The current Ollama model override

State is loaded at startup and written atomically after relevant changes.
`bot_state.json` contains private conversation data and is excluded by
`.gitignore`. Do not commit or share it.

Saved persona presets are stored separately in `personas.json`, which is also
private and excluded from Git.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com) installed and running
- A Discord application and bot token
- FFmpeg, if using voice playback
- A Discord bot installation with the required message and voice permissions

The bot uses faster-whisper locally for speech recognition. Edge TTS is a
network service, so voice replies sent through Edge TTS are not fully local.

## Setup

### Install dependencies

From the repository root:

```bash
pip install -r Velina-bot/requirements.txt
```

### Configure environment variables

Copy `Velina-bot/.env.example` to `.env` in the repository root and fill in
your private values. Never commit the real `.env` file.

Required variables:

```env
DISCORD_TOKEN=your_discord_bot_token_here
TARGET_CHANNEL_IDS=123456789012345678,987654321098765432
```

Optional variables:

```env
OWNER_IDS=123456789012345678
DISABLED_CHANNELS=
OLLAMA_MODEL=Velina-V1
WHISPER_MODEL_SIZE=base
TTS_VOICE=en-US-AriaNeural
TTS_RATE=+0%
TTS_VOLUME=+0%
TTS_PITCH=+0Hz
```

`TARGET_CHANNEL_IDS` accepts comma-separated Discord channel IDs. The older
singular `TARGET_CHANNEL_ID` variable is also supported for compatibility.
If `OWNER_IDS` is missing, owner-only commands are disabled.

`DISABLED_CHANNELS` is optional. It accepts comma-separated IDs and overrides
the target-channel list for normal AI behavior. Leave it empty to disable
nothing.

### Build the Ollama model

The `Modelfile` and GGUF source model are kept inside `Velina-bot`. From that
directory, create the model using the name expected by the default
configuration:

```bash
cd Velina-bot
ollama create Velina-V1 -f Modelfile
```

The GGUF file is large and should remain untracked. If you use another Ollama
model, set its name with `OLLAMA_MODEL` or the owner-only `/setmodel` command.

### Run the bot

From the `Velina-bot` directory:

```bash
python bot.py
```

Running from this directory keeps `personas.json` and `bot_state.json` beside
the bot source. The root `.env` is discovered by the environment loader.

## Model and training data

The intended base model is Meta Llama 3.1 8B Instruct. Velina's personality
was fine-tuned using QLoRA with Unsloth on a custom conversational dataset.
The dataset and model weights are not included in the repository because of
size, privacy, and model-license considerations.

The project dataset contains roughly 220 examples covering casual chat,
emotional support, humor, banter, greetings, bilingual behavior, and
multi-turn exchanges.

## Security notes

- Keep the Discord token private and rotate it immediately if exposed.
- Disable public bot and user-install options where appropriate in the
  Discord Developer Portal.
- Keep the bot's OAuth installation restricted to trusted servers.
- Use `TARGET_CHANNEL_IDS` and `DISABLED_CHANNELS` as application-level
  safeguards even if Discord installation settings are misconfigured.
- Review `bot_state.json` and `personas.json` before sharing the repository;
  both may contain private content.
