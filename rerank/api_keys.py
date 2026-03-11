import os
from typing import Dict, Tuple

from dotenv import load_dotenv


def get_openai_api_key(resource='baidu', model_name='gpt-4o-2024-08-06') -> str:
    load_dotenv(dotenv_path=f".env.local")
    key = os.getenv("OPEN_AI_API_KEY")
    return key


def get_api_key_and_base(model_name: str) -> Tuple[str, str]:
    """Return (api_key, base_url) for the given model."""
    load_dotenv(dotenv_path=".env.local")

    if 'deepseek' in model_name:
        key = os.getenv("DEEPSEEK_API_KEY", os.getenv("OPEN_AI_API_KEY", ""))
        base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        return key, base_url
    else:
        key = os.getenv("OPEN_AI_API_KEY", "")
        base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        return key, base_url

