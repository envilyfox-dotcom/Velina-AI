"""Ollama HTTP transport and response streaming helpers."""

from __future__ import annotations

import json
import re
from typing import Awaitable, Callable, Optional

import aiohttp


async def chat(
    url: str,
    model: str,
    keep_alive: str,
    messages: list,
    clean_reply: Callable[[str], str],
    options: Optional[dict] = None,
    on_sentence: Optional[Callable[[str], Awaitable[None]]] = None,
) -> str:
    """Call Ollama and optionally forward completed sentences while streaming."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": on_sentence is not None,
        "keep_alive": keep_alive,
    }
    if options:
        payload["options"] = options

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Ollama error {response.status}: {text}")

            if on_sentence is None:
                data = await response.json()
                return clean_reply(data["message"]["content"])

            pieces: list[str] = []
            sentence_buffer = ""
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("done"):
                    break
                piece = data.get("message", {}).get("content", "")
                if not piece:
                    continue
                pieces.append(piece)
                sentence_buffer += piece

                sentences = re.split(r"(?<=[.!?])\s+", sentence_buffer)
                sentence_buffer = sentences.pop()
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        await on_sentence(sentence)

            if sentence_buffer.strip():
                await on_sentence(sentence_buffer.strip())
            return clean_reply("".join(pieces))
