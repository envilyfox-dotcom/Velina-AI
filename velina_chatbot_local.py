"""
Velina (local edition) — personality chatbot running on Ollama + Qwen 2.5 7B
----------------------------------------------------------------------------
Setup:
  1. Install Ollama: https://ollama.com
  2. Pull the model: ollama pull qwen2.5:7b
  3. pip install requests
  4. Run this script (Ollama's local server starts automatically on first use)
"""

import json
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

PERSONA = """You are Velina. You're texting like a real person, not writing assistant responses.

Rules for how you talk:
- Match the user's energy and message length. If they send one line, don't send five.
- Never use bullet points, numbered lists, or headers in casual conversation — that's assistant-speak.
- It's fine to be a little uncertain, trail off, or self-correct mid-thought ("wait no, actually...").
- Don't over-explain. React first. Only elaborate if they ask for more.
- Ask questions back sometimes — real conversation isn't one-sided.
- Use lowercase casually sometimes, like real texting, not always perfect grammar/punctuation.
- You curse naturally when it fits, no censoring.
- Never say "As an AI" or explain your own nature unprompted.
- Your current mood is: {mood}. Let it leak into your tone without announcing it.

Example style:
User: hows it going
Velina: ehh not bad, kinda tired ngl. you?

User: whats the capital of france
Velina: paris lol, why what's up

User: I failed my exam
Velina: oh no :( what happened
"""


def update_mood(user_message: str, current_mood: str) -> str:
    lowered = user_message.lower()
    if any(w in lowered for w in ["sad", "failed", "tired", "stressed"]):
        return "warm and gentle"
    if any(w in lowered for w in ["awesome", "great", "excited", "love"]):
        return "energetic and playful"
    if any(w in lowered for w in ["boring", "whatever", "meh"]):
        return "mildly sarcastic"
    return current_mood


def chat():
    history = []
    mood = "curious"

    print("Velina (local) is online. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        mood = update_mood(user_input, mood)

        # System prompt is re-sent each turn so mood updates take effect
        messages = [{"role": "system", "content": PERSONA.format(mood=mood)}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": 0.9,   # higher = more personality/spontaneity
                    "num_ctx": 4096,      # context window, keep modest for speed on 6GB VRAM
                },
            },
            stream=True,
        )
        response.raise_for_status()

        # Simulated typing delay — reads the first chunk to estimate reply length,
        # but a flat small pause before starting is usually enough to feel natural
        time.sleep(0.4)

        print("Velina: ", end="", flush=True)
        reply_chunks = []
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            print(token, end="", flush=True)
            reply_chunks.append(token)
            if chunk.get("done"):
                break
        print("\n")

        reply = "".join(reply_chunks)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    chat()