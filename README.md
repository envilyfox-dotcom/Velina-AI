# Velina AI

Velina is a personality-driven Discord bot powered by a fine-tuned **Meta Llama 3.1 8B Instruct** model, running locally via [Ollama](https://ollama.com).

Rather than relying purely on prompting, Velina's voice, tone, and personality are fine-tuned directly into the model weights using **QLoRA** — trained on a custom dataset built from a personal seed set (~40 handwritten examples) and expanded to roughly **220 examples** across a range of conversational categories (casual chat, emotional support, humor, banter, greetings, and more).

## Features

- **Fine-tuned personality** — not just a system prompt, Velina's character is trained into the model itself
- **Bilingual tone-switching** — casual and blunt in English, noticeably more polite and warm in Indonesian
- **Runs fully locally** — powered by Ollama, no external API costs or data leaving your machine
- **Discord integration** — deployed as a Discord App with slash commands

## Discord Commands

| Command | Description |
|---|---|
| `/reset` | Resets Velina's memory and conversation context |
| `/persona` | Temporarily sets a custom persona, usable by anyone in the server |
| `/ping` | Checks the bot's current response latency |

## Tech Stack

- **Base model:** Meta Llama 3.1 8B Instruct
- **Fine-tuning:** QLoRA via [Unsloth](https://unsloth.ai), trained on Google Colab
- **Runtime:** [Ollama](https://ollama.com) (local inference)
- **Bot framework:** Python + discord.py

## Setup

### Prerequisites
- [Ollama](https://ollama.com) installed and running
- Python 3.10+
- A Discord bot token ([Discord Developer Portal](https://discord.com/developers/applications))

### Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/envilyfox-dotcom/Velina-AI.git
   cd Velina-AI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables — create a `.env` file in the project root:
   ```
   DISCORD_BOT_TOKEN=your_token_here
   ```

4. Build the Ollama model (Modelfile and GGUF not included in this repo — see note below):
   ```bash
   ollama create velina -f Modelfile
   ```

5. Run the bot:
   ```bash
   python bot.py
   ```

## Note on Model Files

The fine-tuned model weights (`.gguf`) and the `Modelfile` are **not included** in this repository due to file size and to keep Velina's exact persona configuration private. To run your own version:
- Fine-tune your own model following a similar pipeline (see [Unsloth's documentation](https://docs.unsloth.ai))
- Or adapt `bot.py` to point at any Ollama model of your choice

## Training Data

The dataset is not included in this repository. It consists of ~220 custom conversational examples across categories including casual conversation, emotional support, humor, playful banter, greetings, and multi-turn exchanges — built to teach a consistent, natural texting-style personality rather than an assistant-like tone.
