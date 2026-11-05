"""A local Hugging Face model with an optional PEFT adapter, behind the same `LLMClient` interface.

Ollama cannot load a PEFT adapter, so E5 runs both the base model and the adapter through this client, so the runtime
is the same. Nothing heavy is imported or loaded until the first call (or `load`). The base model is loaded in 4-bit,
as in training, so it fits in 8 GB; `unload` frees it before another model is loaded."""
from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from contextlens.llm.client import LLMError, LLMResult, _Base


class HFClient(_Base):
    def __init__(self, model_path: str, adapter_path: str | None = None) -> None:
        self.model_path, self.adapter_path = model_path, adapter_path
        self.model = model_path + (f"+{adapter_path}" if adapter_path else "")
        self._model: Any = None
        self._tok: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, quantization_config=quant, device_map={"": 0})
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._model, self._tok = model, AutoTokenizer.from_pretrained(self.model_path)

    def unload(self) -> None:
        """Drop the model and give its GPU memory back."""
        self._model = self._tok = None
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(self, messages: list[dict], max_tokens: int, temperature: float, timeout_s: float) -> tuple[str, int, int]:
        """Text, prompt tokens and generated tokens. Thinking is off, as in training. `max_time` stops a long reply."""
        import torch

        self.load()
        prompt = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        ids = self._tok(prompt, return_tensors="pt", add_special_tokens=False).to(self._model.device)
        sample = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        self._model.eval()
        if getattr(self._model, "is_gradient_checkpointing", False):  # a training setting must not leak into generation
            self._model.gradient_checkpointing_disable()
        with torch.no_grad():
            out = self._model.generate(**ids, max_new_tokens=max_tokens, max_time=timeout_s, use_cache=True,
                                       pad_token_id=self._tok.pad_token_id, **sample)
        new = out[0][ids["input_ids"].shape[1]:]
        return self._tok.decode(new, skip_special_tokens=True), int(ids["input_ids"].shape[1]), int(new.shape[0])

    def _raw(self, messages: list[dict], schema: type[BaseModel] | None, max_tokens: int, temperature: float,
             timeout_s: float) -> LLMResult:
        t0 = time.perf_counter()
        try:
            text, n_prompt, n_new = self._generate(messages, max_tokens, temperature, timeout_s)
        except (RuntimeError, ValueError, OSError) as err:
            raise LLMError(f"local model call failed: {type(err).__name__}") from err
        return LLMResult(text=text, model=self.model, prompt_tokens=n_prompt, completion_tokens=n_new,
                         latency_ms=(time.perf_counter() - t0) * 1000)
