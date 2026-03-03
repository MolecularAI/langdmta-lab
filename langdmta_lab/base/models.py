"""Module contains a class or instance for the LLM model."""

import os
from typing import Any, Optional

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from langdmta_lab.config import (
    AZURE_OPENAI_GPT4o_VERSION,
)


class AzureOpenAIBaseModel(AzureChatOpenAI):
    """Base class for wrapping Azure OpenAI endpoints."""

    def __init__(
        self, api_key: Optional[str], model_name: str, api_version: str, **kwargs: Any
    ) -> None:
        if not api_key:
            api_key = AzureOpenAIBaseModel._get_api_key_from_env()
        super().__init__(
            openai_api_key=api_key,
            openai_api_type="azure",
            openai_api_version=api_version,
            azure_deployment=model_name,
            model_name=model_name,
            **kwargs,
        )

    @staticmethod
    def _get_api_key_from_env() -> str:
        return os.environ["AZURE_OPENAI_API_KEY"]


class AzureOpenAIGPT4oModel(AzureOpenAIBaseModel):

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(
            api_key=api_key,
            model_name="gpt-4o",
            api_version=AZURE_OPENAI_GPT4o_VERSION,
            **kwargs,
        )


class OpenAIGPT4oModel(ChatOpenAI):
    """Class for wrapping OpenAI GPT-4o model."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            model_name="gpt-4o",
            **kwargs,
        )


MODEL_REGISTRY = {
    "gpt-4o-azure": AzureOpenAIGPT4oModel,
    "gpt-4o-openai": OpenAIGPT4oModel,
}
