#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 22:59
@Author  : alexanderwu
@File    : __init__.py
"""
# Import providers in a fail-soft way so that optional cloud SDKs are not
# required unless their corresponding api_type is actually used.

from maas.provider.openai_api import OpenAILLM

try:  # optional
    from maas.provider.google_gemini_api import GeminiLLM
except Exception:  # pragma: no cover
    GeminiLLM = None  # type: ignore

try:  # optional
    from maas.provider.ollama_api import OllamaLLM
except Exception:  # pragma: no cover
    OllamaLLM = None  # type: ignore

try:  # optional
    from maas.provider.zhipuai_api import ZhiPuAILLM
except Exception:  # pragma: no cover
    ZhiPuAILLM = None  # type: ignore

try:  # optional
    from maas.provider.azure_openai_api import AzureOpenAILLM
except Exception:  # pragma: no cover
    AzureOpenAILLM = None  # type: ignore

try:  # optional
    from maas.provider.metagpt_api import MetaGPTLLM
except Exception:  # pragma: no cover
    MetaGPTLLM = None  # type: ignore

try:  # optional
    from maas.provider.human_provider import HumanProvider
except Exception:  # pragma: no cover
    HumanProvider = None  # type: ignore

try:  # optional
    from maas.provider.spark_api import SparkLLM
except Exception:  # pragma: no cover
    SparkLLM = None  # type: ignore

try:  # optional
    from maas.provider.qianfan_api import QianFanLLM
except Exception:  # pragma: no cover
    QianFanLLM = None  # type: ignore

try:  # optional
    from maas.provider.dashscope_api import DashScopeLLM
except Exception:  # pragma: no cover
    DashScopeLLM = None  # type: ignore

try:  # optional
    from maas.provider.anthropic_api import AnthropicLLM
except Exception:  # pragma: no cover
    AnthropicLLM = None  # type: ignore

try:  # optional
    from maas.provider.bedrock_api import BedrockLLM
except Exception:  # pragma: no cover
    BedrockLLM = None  # type: ignore

try:  # optional
    from maas.provider.ark_api import ArkLLM
except Exception:  # pragma: no cover
    ArkLLM = None  # type: ignore

__all__ = [
    "GeminiLLM",
    "OpenAILLM",
    "ZhiPuAILLM",
    "AzureOpenAILLM",
    "MetaGPTLLM",
    "OllamaLLM",
    "HumanProvider",
    "SparkLLM",
    "QianFanLLM",
    "DashScopeLLM",
    "AnthropicLLM",
    "BedrockLLM",
    "ArkLLM",
]
