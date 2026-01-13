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

## Documentation

- `docs/RESEARCH-REPORT.md` - Architecture research and component options
- `docs/ENVIRONMENT.md` - DGX Spark environment details
