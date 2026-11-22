from __future__ import annotations

import logging
import re
from pathlib import Path

from src.text.rules import detect_flags, gloss_to_sentence

logger = logging.getLogger(__name__)

SYSTEM = (
    "You rewrite ISL gloss sequences into ONE grammatical English sentence. "
    "Rules: never introduce a content word absent from the draft; "
    "keep total words <= glosses + 4; use ONLY draft words plus "
    "is/are/do/can/please/a/the/to/for/of/with/and. "
    "If a token looks like FS(NAME), copy NAME exactly. "
    "Output exactly ONE sentence, no preamble, no quotes. "
    "End with '.' or '?' ('?' only if QUESTION=yes)."
)

SYSTEM_OVERRIDE: str | None = None


def _system() -> str:
    return SYSTEM_OVERRIDE or SYSTEM


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.?])\s+", text.strip().strip("\"'"))
    return parts[0] if parts else ""


def _content_ok(candidate: str, glosses: list[str], lexicon: set[str]) -> bool:
    allow = {"is", "are", "do", "does", "can", "please", "a", "the", "to",
             "for", "of", "with", "and"}
    toks = re.findall(r"[A-Za-z-]+", candidate.lower())
    if not toks:
        return False
    gloss_set = {g.lower() for g in glosses}
    for t in toks:
        if t in allow or t in gloss_set:
            continue
        stem = t[:4]
        if any(w.startswith(stem) or stem.startswith(w[:4]) for w in gloss_set | lexicon):
            continue
        return False
    content = [g.lower() for g in glosses if g.lower() not in allow]
    kept = sum(1 for g in content if g[:4] in candidate.lower())
    return kept >= 0.8 * len(content) if content else True


class RulesOnlyEngine:
    def polish(self, glosses: list[str], flags: dict | None = None) -> tuple[str, str]:
        draft = gloss_to_sentence(glosses, flags or detect_flags(glosses))
        return draft, "rules"


class LlamaCppPolishEngine:
    def __init__(self, model_path: str | Path, lexicon: set[str] | None = None,
                 n_ctx: int = 1024, n_gpu_layers: int = -1):
        self.model_path = str(model_path)
        self.lexicon = lexicon or set()
        self._llm = None
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers

    def _ensure(self):
        if self._llm is None:
            from llama_cpp import Llama

            self._llm = Llama(model_path=self.model_path, n_ctx=self._n_ctx,
                              n_gpu_layers=self._n_gpu_layers, verbose=False)
        return self._llm

    def _generate(self, system: str, user: str) -> str:
        llm = self._ensure()
        out = llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0, max_tokens=64,
        )
        return out["choices"][0]["message"]["content"]

    def polish(self, glosses: list[str], flags: dict | None = None) -> tuple[str, str]:
        flags = flags or detect_flags(glosses)
        draft = gloss_to_sentence(glosses, flags)
        try:
            user = (f"GLOSSES: {' '.join(glosses)}\nDRAFT: {draft}\n"
                    f"QUESTION={'yes' if flags.get('Q_WH') or flags.get('Q_POLAR') else 'no'}\nRewrite:")
            cand = _first_sentence(self._generate(_system(), user))
            if not cand or not _content_ok(cand, glosses, self.lexicon):
                return draft, "rules-fallback"
            if len(cand.split()) > len(glosses) + 6 or len(cand.split()) < 2:
                return draft, "rules-fallback"
            return cand, "llm"
        except Exception as e:
            logger.warning(f"LLM polish failed, using rules: {e}")
            return draft, "rules-fallback"


class TransformersPolishEngine(LlamaCppPolishEngine):
    def _ensure(self):
        if self._llm is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.float16,
                device_map="cuda", trust_remote_code=True).eval()
            self._llm = (tok, mdl)
        return self._llm

    def _generate(self, system: str, user: str) -> str:
        import torch

        tok, mdl = self._ensure()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        attn = enc.get("attention_mask")
        attn = attn.to("cuda") if attn is not None else None
        with torch.inference_mode():
            out = mdl.generate(ids, attention_mask=attn, max_new_tokens=64, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
