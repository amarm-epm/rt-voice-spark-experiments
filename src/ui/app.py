"""
Gradio web interface for the voice assistant.
"""

import tempfile
from pathlib import Path

import gradio as gr

from src.stt.whisper_stt import WhisperSTT
from src.llm.llama_llm import LlamaLLM
from src.tts.piper_tts import PiperTTS


class VoiceAssistant:
    """Voice assistant pipeline: STT → LLM → TTS."""

    def __init__(self):
        print("Loading models...")
        print("  - Loading STT (Whisper)...")
        self.stt = WhisperSTT()
        print("  - Loading LLM (Llama)...")
        self.llm = LlamaLLM()
        print("  - Loading TTS (Piper)...")
        self.tts = PiperTTS(use_cuda=False)  # CPU for now
        print("Models loaded!")

    def process(self, audio_path: str) -> tuple[str, str, str]:
        """
        Process audio through the full pipeline.

        Args:
            audio_path: Path to input audio file

        Returns:
            Tuple of (transcript, response_text, response_audio_path)
        """
        if audio_path is None:
            return "", "", None

        # STT: Audio → Text
        transcript = self.stt.transcribe(audio_path)

        # LLM: Text → Response
        response_text = self.llm.chat(transcript)

        # TTS: Response → Audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        self.tts.synthesize(response_text, output_path)

        return transcript, response_text, output_path


def create_app(assistant: VoiceAssistant = None) -> gr.Blocks:
    """Create the Gradio app."""

    if assistant is None:
        assistant = VoiceAssistant()

    with gr.Blocks(title="Voice Assistant") as app:
        gr.Markdown("# Voice Assistant")
        gr.Markdown("Record your question, get a spoken response.")

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or upload audio",
                )
                submit_btn = gr.Button("Submit", variant="primary")
                gr.Markdown("*Note: Microphone requires HTTPS. Use file upload over HTTP.*")

            with gr.Column():
                transcript_output = gr.Textbox(
                    label="Your question (transcript)",
                    lines=2,
                )
                response_output = gr.Textbox(
                    label="Assistant response",
                    lines=4,
                )
                audio_output = gr.Audio(
                    label="Response audio",
                    type="filepath",
                    autoplay=True,
                )

        submit_btn.click(
            fn=assistant.process,
            inputs=[audio_input],
            outputs=[transcript_output, response_output, audio_output],
        )

    return app


def main():
    """Launch the Gradio app."""
    print("Initializing Voice Assistant...")
    assistant = VoiceAssistant()

    print("Creating Gradio app...")
    app = create_app(assistant)

    print("Launching server...")
    # SSL certs for HTTPS (microphone requires secure context)
    cert_dir = Path(__file__).parent.parent.parent / "certs"
    ssl_certfile = cert_dir / "cert.pem"
    ssl_keyfile = cert_dir / "key.pem"

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        ssl_certfile=str(ssl_certfile) if ssl_certfile.exists() else None,
        ssl_keyfile=str(ssl_keyfile) if ssl_keyfile.exists() else None,
    )


if __name__ == "__main__":
    main()
