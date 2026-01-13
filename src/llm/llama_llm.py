"""
LLM wrapper using llama-cpp-python.
"""

from pathlib import Path
from llama_cpp import Llama

# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"


class LlamaLLM:
    """Wrapper for Llama model inference."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,  # -1 = offload all layers to GPU
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def chat(
        self,
        user_message: str,
        system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational.",
        max_tokens: int = 256,
    ) -> str:
        """Generate a response to a user message."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )

        return response["choices"][0]["message"]["content"]


def test_llm():
    """Quick test of LLM functionality."""
    print("Loading model...")
    llm = LlamaLLM()

    print("Model loaded. Testing inference...")
    prompt = "What is 2 + 2?"
    print(f"User: {prompt}")

    response = llm.chat(prompt)
    print(f"Assistant: {response}")

    return True


if __name__ == "__main__":
    test_llm()
