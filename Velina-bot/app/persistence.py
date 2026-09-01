"""Disk persistence for persona presets and runtime bot state."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import MutableMapping, MutableSet
from typing import Any


def load_personas(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        logging.exception("Failed to read %s, treating as empty", path)
        return {}


def save_personas(path: str, personas: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(personas, file, ensure_ascii=False, indent=2)


def load_state(
    path: str,
    conversation_history: MutableMapping[int, list],
    channel_summaries: MutableMapping[int, str],
    channel_personas: MutableMapping[int, str],
    paused_channel_ids: MutableSet[int],
    disabled_join_channel_ids: MutableSet[int],
) -> None:
    """Restore state into the bot's in-memory collections."""
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as file:
            state = json.load(file)
    except (json.JSONDecodeError, OSError):
        logging.exception("Failed to read %s; starting with empty runtime state", path)
        return

    if not isinstance(state, dict):
        logging.warning("Ignoring %s because its root value is not an object", path)
        return

    def channel_entries(name: str):
        saved = state.get(name, {})
        if not isinstance(saved, dict):
            return
        for raw_channel_id, value in saved.items():
            try:
                channel_id = int(raw_channel_id)
            except (TypeError, ValueError):
                continue
            yield channel_id, value

    for channel_id, entries in channel_entries("conversation_history"):
        if isinstance(entries, list):
            conversation_history[channel_id] = entries

    for channel_id, summary in channel_entries("channel_summaries"):
        if isinstance(summary, str) and summary:
            channel_summaries[channel_id] = summary

    for channel_id, persona in channel_entries("channel_personas"):
        if isinstance(persona, str):
            channel_personas[channel_id] = persona

    for name, destination in (
        ("paused_channel_ids", paused_channel_ids),
        ("disabled_join_channel_ids", disabled_join_channel_ids),
    ):
        values = state.get(name, [])
        if isinstance(values, list):
            for raw_channel_id in values:
                try:
                    destination.add(int(raw_channel_id))
                except (TypeError, ValueError):
                    continue

    logging.info("Loaded persistent bot state from %s", path)


def save_state(
    path: str,
    conversation_history: MutableMapping[int, list],
    channel_summaries: MutableMapping[int, str],
    channel_personas: MutableMapping[int, str],
    paused_channel_ids: MutableSet[int],
    disabled_join_channel_ids: MutableSet[int],
) -> None:
    """Persist runtime state atomically."""
    state: dict[str, Any] = {
        "conversation_history": {
            str(channel_id): entries
            for channel_id, entries in conversation_history.items()
        },
        "channel_summaries": {
            str(channel_id): summary
            for channel_id, summary in channel_summaries.items()
        },
        "channel_personas": {
            str(channel_id): persona
            for channel_id, persona in channel_personas.items()
        },
        "paused_channel_ids": sorted(paused_channel_ids),
        "disabled_join_channel_ids": sorted(disabled_join_channel_ids),
    }
    temporary_path = f"{path}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)
    except OSError:
        logging.exception("Failed to write %s", path)
        try:
            os.remove(temporary_path)
        except OSError:
            pass
