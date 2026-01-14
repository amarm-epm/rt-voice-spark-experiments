"""
Real-time voice assistant entry point.

Run with: uv run python -m src.main
"""

from src.ui.app import main as launch_app


def main():
    launch_app()


if __name__ == "__main__":
    main()
