from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class DirectQwenLLM:
    model_path: str = "/workspace/StructRAG/model/Qwen2.5-32B-Instruct"
    model_name: str = "Qwen2.5-32B-Instruct"
    temperature: float = 0.0
    max_new_tokens: int = 2048
    top_p: float = 0.95
    dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None
    tokenizer_use_fast: bool = True
    _tokenizer: Any = field(init=False, default=None, repr=False)
    _model: Any = field(init=False, default=None, repr=False)
    _torch: Any = field(init=False, default=None, repr=False)
    _loaded: bool = field(init=False, default=False, repr=False)
    _load_lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.model_path = os.environ.get("DAWON_MODEL_PATH", self.model_path)
        self.model_name = os.environ.get("DAWON_MODEL_NAME", self.model_name)
        self.dtype = os.environ.get("DAWON_DTYPE", self.dtype)
        self.device_map = os.environ.get("DAWON_DEVICE_MAP", self.device_map)
        self.attn_implementation = os.environ.get("DAWON_ATTN_IMPL", self.attn_implementation or "") or None
        self.model_path = str(Path(self.model_path).expanduser())

    @property
    def model(self) -> str:
        return self.model_name

    def healthcheck(self) -> None:
        self._ensure_loaded()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_loaded()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._tokenizer(prompt, return_tensors="pt")
        input_device = self._input_device()
        encoded = {key: value.to(input_device) for key, value in encoded.items()}

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p

        with self._torch.inference_mode():
            output_ids = self._model.generate(**encoded, **generate_kwargs)

        prompt_length = encoded["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        content = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not content:
            raise RuntimeError("DirectQwenLLM returned empty content.")
        return content

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"Model path does not exist: {self.model_path}. "
                    "Set --model_path or DAWON_MODEL_PATH to your local Qwen checkpoint."
                )
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "DirectQwenLLM requires 'torch' and 'transformers'. "
                    "Run with /workspace/venvs/structrag/bin/python or install the dependencies in your env."
                ) from exc

            torch_dtype = self._resolve_torch_dtype(torch)
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": torch_dtype,
                "device_map": self.device_map,
                "trust_remote_code": self.trust_remote_code,
                "low_cpu_mem_usage": True,
            }
            if self.attn_implementation:
                model_kwargs["attn_implementation"] = self.attn_implementation

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                use_fast=self.tokenizer_use_fast,
            )
            if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
            self._model.eval()
            self._torch = torch
            self._loaded = True

    def _input_device(self) -> Any:
        hf_device_map = getattr(self._model, "hf_device_map", None) or {}
        for device in hf_device_map.values():
            if isinstance(device, str) and device not in {"cpu", "disk", "meta"}:
                return device
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            return model_device
        if self._torch.cuda.is_available():
            return self._torch.device("cuda:0")
        return self._torch.device("cpu")

    def _resolve_torch_dtype(self, torch_module: Any) -> Any:
        normalized = str(self.dtype).strip().lower()
        if normalized in {"auto", ""}:
            return "auto"
        mapping = {
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "half": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
        }
        if normalized not in mapping:
            supported = ", ".join(sorted(mapping))
            raise ValueError(f"Unsupported dtype '{self.dtype}'. Choose one of: auto, {supported}.")
        return mapping[normalized]
