import json
import base64
from io import BytesIO
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Sequence, Union
from dataclasses import dataclass

import requests
import numpy as np
from PIL import Image
 
 
MessageList= Sequence[Dict[str, Any]]
 
def response_adjustment(response):
    raw = response.text if hasattr(response, "text") else response
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
 
    raw = raw.lstrip("\ufeff").strip()
    if raw.startswith("```"):
        # drop first fence line
        raw = raw.split("\n", 1)[1] if "\n" in raw else ""
        # drop trailing fence
        if raw.rstrip().endswith("```"):
            raw = raw[: raw.rfind("```")].rstrip()
    return raw
 
class LLMModel(Enum):
    GEMMA_27B = "gemma_27b"
    GEMMA_4B = "gemma_4b"
    GEMMA_270M = "gemma_270m"
 
 
@dataclass
class LLMConfig:
    gemma_270m_url = "https://gemma-3-270m-it-cpu-rhoai-test.apps.ocpdataprod.domain.bankanet.com.tr/v1/chat/completions"
    gemma_27b_url = "https://gemma-3-27b-it-quantized-gpu-rhoai-test.apps.ocpdataprod.domain.bankanet.com.tr/v1/chat/completions"
    gemma_4b_url = "https://gemma-3-gpu-rhoai-test.apps.ocpdataprod.domain.bankanet.com.tr:443/v1/chat/completions"
 
 
class LLMClient:
    """Unified client that provides access to all configured LLM endpoints."""
 
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()
        self._handlers: Dict[LLMModel, Callable[..., str]] = {
            LLMModel.GEMMA_27B: self._call_gemma27b,
            LLMModel.GEMMA_4B: self._call_gemma4b,
            LLMModel.GEMMA_270M: self._call_gemma270m
        }
    def available_models(self) -> Sequence[str]:
        return [model.value for model in self._handlers]
 
 
    def generate(
        self,
        model: Union[LLMModel, str],
        messages: Optional[MessageList] = None,
        *,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, dict]:
        # Input Validation
        if messages is not None and (
            system_prompt is not None or user_prompt is not None
        ):
            raise ValueError(
                "Cannot provide both 'messages' or 'system_prompt/user_prompt'. Select the format"
            )
 
        if messages is None:
            if system_prompt is None:
                raise ValueError(
                    "Must provide either 'messages' or 'system_prompt' with optional 'user_prompt'"
                )
            messages = [{"role": "system", "content": system_prompt}]
            if user_prompt:
                messages.append({"role":"user", "content": user_prompt})
        model_enum = self._coerce_model(model)
        handler = self._handlers.get(model_enum)
        if handler is None:
            raise ValueError(f"Model '{model_enum.value}' is not registered")
        return handler(messages = messages, **kwargs)
 
    def generate_with_image(
        self,
        model: Union[LLMModel, str],
        images: Sequence[np.ndarray],
        messages: Optional[MessageList]= None,
        *,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        image_format: str = "PNG",
        **kwargs: Any,
    ) -> Union[str, dict]:
        """Send one or more images to a multimodal model.

        ``images`` is a sequence of NumPy arrays (one per image). Each array is a
        single-channel (224, 224) signature, normalized to the [0, 1] range. The
        arrays are encoded in-memory and attached to the last user message as
        base64 data URIs — no files are read from disk.
        """
        # Input Validation
        if messages is not None and (
            system_prompt is not None or user_prompt is not None
        ):
            raise ValueError(
                "Cannot provide both 'messages' or 'system_prompt/user_prompt'. Select the format"
            )

        if messages is None:
            if system_prompt is None:
                raise ValueError(
                    "Must provide either 'messages' or 'system_prompt' with optional 'user_prompt'"
                )
            messages = [{"role": "system", "content": system_prompt}]
            if user_prompt:
                messages.append({"role":"user", "content": user_prompt})
        messages = list(messages)

        last_user_idx= None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            raise ValueError("No user message found to attach images to")
        user_msg = messages[last_user_idx]
        current_content = user_msg['content']

        # Build multimodal content
        if isinstance(current_content, list):
            content = list(current_content)
        elif isinstance(current_content, str):
            content = [{"type": "text", "text": current_content}]
        else:
            content = [{"type": "text", "text": str(current_content)}]

        # Attach each NumPy array as an inline base64 image
        mime_subtype = "jpeg" if image_format.upper() in ("JPG", "JPEG") else image_format.lower()
        mime_type = f"image/{mime_subtype}"
        for array in images:
            encoded = self.encode_array_to_base64(array, image_format=image_format)
            data_uri = f"data:{mime_type};base64,{encoded}"
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
        messages[last_user_idx]["content"] = content
        return self.generate(model, messages=messages, **kwargs)
 
 
    def _call_gemma27b(
        self,
        *,
        messages: Optional[MessageList],
        temperature: float = 0.2,
        max_output_tokens: int = 5000,
        raw_response: bool = False
    ) -> Union[str, dict]:
        url = self.config.gemma_27b_url
        headers = {
            "Content-Type": "application/json" 
        }
 
        payload = {
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_output_tokens
        }
 
        response = requests.post(url, headers= headers, data=json.dumps(payload), verify=False)
        response.raise_for_status()
        response_data = response.json()
        if raw_response:
            return response_data
        return response_data["choices"][0]["message"]["content"]
 
 
    def _call_gemma4b(
        self,
        *,
        messages: Optional[MessageList],
        temperature: float = 0.2,
        max_output_tokens: int = 5000,
        raw_response: bool = False
    ) -> Union[str, dict]:
        url = self.config.gemma_4b_url
        headers = {
            "Content-Type": "application/json" 
        }
 
        payload = {
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_output_tokens
        }
 
        response = requests.post(url, headers= headers, data=json.dumps(payload), verify=False)
        response.raise_for_status()
        response_data = response.json()
        if raw_response:
            return response_data
        return response_data["choices"][0]["message"]["content"]
 
    def _call_gemma270m(
        self,
        *,
        messages: Optional[MessageList],
        temperature: float = 0.2,
        max_output_tokens: int = 3000,
        raw_response: bool = False
    ) -> Union[str, dict]:
        url = self.config.gemma_270m_url
        headers = {
            "Content-Type": "application/json" 
        }
 
        payload = {
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_output_tokens
        }
 
        response = requests.post(url, headers= headers, data=json.dumps(payload), verify=False)
        response.raise_for_status()
        response_data = response.json()
        if raw_response:
            return response_data
        return response_data["choices"][0]["message"]["content"]
 
    @staticmethod
    def _coerce_model(model: Union[LLMModel, str]) -> LLMModel:
        if isinstance(model, LLMModel):
            return model
        try:
            return  LLMModel(model)
        except ValueError as exc:
            raise ValueError(f"Unknown model identifier: {model}") from exc
 
    @staticmethod
    def encode_array_to_base64(array: np.ndarray, image_format: str = "PNG") -> str:
        """Encode a NumPy image array to a base64 string.

        Expects a single-channel (224, 224) array normalized to [0, 1]. The array
        is scaled to the 0-255 pixel range, converted to a grayscale image, and
        encoded (PNG by default — lossless, so no recompression artifacts are
        introduced on the already-preprocessed signature). uint8 arrays are passed
        through unchanged.
        """
        arr = np.asarray(array)
        # Drop any singleton channel dim, e.g. (224, 224, 1) -> (224, 224)
        if arr.ndim > 2:
            arr = np.squeeze(arr)

        if arr.dtype != np.uint8:
            # Arrays arrive normalized to [0, 1]; scale back to pixel range.
            arr = np.clip(arr.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)

        image = Image.fromarray(arr)  # 2-D uint8 -> PIL mode "L" (grayscale)
        buffer = BytesIO()
        image.save(buffer, format=image_format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
 
# Singleton pattern: tek bir client yaratmak istiyorsak.
_DEFAULT_CLIENT: Optional[LLMClient] = None
 
def get_default_llm_client() -> LLMClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = LLMClient()
    return _DEFAULT_CLIENT
