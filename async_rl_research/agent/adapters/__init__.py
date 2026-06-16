"""Repo-owned slime adapter variants handling model-specific rendering quirks
without patching slime core. See ``qwen.py``."""

from .qwen import QwenOpenAIAdapter

__all__ = ["QwenOpenAIAdapter"]
