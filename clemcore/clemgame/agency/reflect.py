"""
Reflection component for error correction in clembench games.
Enables agents to reflect on mistakes and improve their responses.
"""
import abc
from typing import Dict, List

from clemcore import backends


class ReflectComponent(abc.ABC):
    """Base class for reflection components that help agents learn from their errors."""

    @abc.abstractmethod
    def generate_reflection(self, error_info: Dict, last_response: str, memory: List[Dict]) -> str:
        """Generate a reflection about an error.

        Args:
            error_info: Information about the error that occurred.
            last_response: The player's response that caused the error.
            memory: The player's conversation history.

        Returns:
            A reflection message to insert into the agent's context.
        """
        pass


class ErrorReflectComponent(ReflectComponent):
    """
    Reflects on errors to help agents correct mistakes.
    Triggered when ParseError or GameError occurs during gameplay.
    """

    def __init__(self, model: backends.Model):
        self.model = model

    def generate_reflection(self, error_info: Dict, last_response: str, memory: List[Dict]) -> str:
        """Generate a reflection about the error.

        Args:
            error_info: Dictionary with 'type' (error type) and 'reason' (error description).
            last_response: The response that caused the error.
            memory: Conversation history.

        Returns:
            A reflection message.
        """
        if self.model is not None:
            return self._model_reflection(error_info, last_response, memory)
        else:
            print("No reflexion model is specified. Therefore, no reflexion is happening.")

    def _model_reflection(self, error_info: Dict, last_response: str, memory: List[Dict]) -> str:
        """Generate a model-based reflection."""
        error_type = error_info.get("type", "Error")
        error_reason = error_info.get("reason", "Unknown error")

        reflection_prompt = f"""You just made a mistake. Analyze what went wrong:

                In 2-3 sentences, explain:
                1. What was the mistake?
                2. How should you correct it?

                IMPORTANT: Only provide your analysis in the "explanation:" format. Do NOT include a "guess:" in your reflection."""

        reflection_messages = memory + [{"role": "user", "content": reflection_prompt}]

        try:
            _, _, reflection_text = self.model.generate_response(reflection_messages)
            return f"[REFLECTION]\n{reflection_text}"
        except Exception:
            print("Something went wrong while generating reflection.")
            return None
