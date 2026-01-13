# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project builds a **fully local real-time voice assistant** on NVIDIA DGX Spark (Grace Blackwell GB10) with a **Gradio web UI** for remote access.

```
[Browser]                    [DGX Spark Server]
    │                              │
Mic recording ──────────────► Whisper STT
                                   │
                              LLM inference
                                   │
Audio playback ◄─────────────── Piper TTS
```

Hardware: DGX Spark with 119GB unified memory, CUDA 13.0, NVIDIA GB10 GPU.

## Commands

```bash
# Run the application
uv run python -m src.main

# Add dependencies
uv add <package>

# Sync dependencies from lockfile
uv sync
```

## Project Structure

```
src/
├── main.py          # Entry point (launches Gradio server)
├── pipeline.py      # Orchestrates STT → LLM → TTS
├── ui/              # Gradio web interface
├── stt/             # Speech-to-text (Whisper)
├── llm/             # LLM inference (llama.cpp)
└── tts/             # Text-to-speech (Piper)

models/              # Model files (gitignored)
tmp/                 # Working files (gitignored)
```

## Technology Stack

- **Python 3.12** (aarch64)
- **uv** for package management
- **Gradio** for web UI
- **CUDA 13.0** for GPU acceleration

## Development Workflow

**Git commit rules:**
- Commit after each completed task (granular commits)
- Always push to remote immediately after commit
- This preserves progress and prevents knowledge loss

**Task tracking:**
- Use `bd` (beads) for structured task tracking
- Session logs in `docs/sessions/` for human-readable context

**Session tracking:**
- Each session is documented in `docs/sessions/YYYY-MM-DD.md`
- At session start: read the latest session file for context
- At session end: update or create session file with summary
- Include: initial prompt, work completed, next steps, notes

**Working files:**
- Use `tmp/` for runtime/working files (test audio, intermediate outputs, etc.)
- `tmp/` is gitignored - do not store anything permanent there
- Keeps project root clean

## Documentation

- `docs/RESEARCH-REPORT.md` - Architecture research and component options
- `docs/ENVIRONMENT.md` - DGX Spark environment details
- `docs/sessions/` - Session-based progress tracking
