import json
import re
from typing import List, Optional, TypedDict, Any, Dict
import inspect
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TORCH"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from peft import PeftModel
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from nltk.tokenize import sent_tokenize

# ----------------------
# State schema
# ----------------------
class SimplifyState(TypedDict, total=False):
    level: str
    sents: List[str]
    labels: List[str]
    rewritten: List[str]
    draft: str

    # Loop counters
    passes: int  # number of CEFR repair attempts taken
    salience_passes: int  # number of salience repair attempts taken
    loops: int  # total control-loop iterations (safety cap)

    # Control routing (set by control_node)
    next_step: str  # one of: "repair", "repair_cefr", "stitch"

    # CEFR check
    cefr_ok: bool
    cefr_pred: Optional[str]

    # Salience / preservation checks
    salient_questions: List[Dict[str, Any]]
    salience_fail_ids: List[int]

    cefr_fail_ids: List[int]     # sentence ids whose CEFR is above target
    cefr_sent_preds: List[str]   # per-sentence CEFR predictions aligned to sents   

# ----------------------
# Debug helpers
# ----------------------
DEBUG = False

def _dbg(state, where: str, max_preview: int = 160) -> None:
    """Print a compact snapshot of pipeline state for debugging."""
    if not DEBUG:
        return
    try:
        print(f"\n=== {where} ===")
        keys = [
            "level",
            "passes",
            "salience_passes",
            "cefr_ok",
            "cefr_pred",
            "salience_fail_ids",
            "cefr_fail_ids",
            "cefr_sent_preds",
            "loops",
            "next_step",
        ]
        for k in keys:
            print(f"{k}: {state.get(k, None)}")

        sents = state.get("sents", None)
        labels = state.get("labels", None)
        rewritten = state.get("rewritten", None)
        draft = state.get("draft", None)

        def _len(x):
            return len(x) if isinstance(x, (list, str)) else None

        print(f"sents: type={type(sents).__name__} len={_len(sents)}")
        print(f"labels: type={type(labels).__name__} len={_len(labels)}")
        print(f"rewritten: type={type(rewritten).__name__} len={_len(rewritten)}")
        print(f"draft: type={type(draft).__name__} chars={_len(draft)}")
        if isinstance(draft, str):
            prev = draft.strip().replace("\n", " ")
            print(f"draft_preview: {prev[:max_preview]}")

        # Helpful per-sentence preview (print all sentences)
        if isinstance(sents, list) and isinstance(labels, list) and isinstance(rewritten, list):
            n = min(len(sents), len(labels), len(rewritten))
            for i in range(n):
                s0 = str(sents[i]).strip().replace("\n", " ")
                r0 = str(rewritten[i]).strip().replace("\n", " ")
                print(f"[{i}] label={labels[i]} | sent={s0} | rew={r0}")
    except Exception as e:
        print(f"[dbg] failed at {where}: {e}")

def _assert_invariants(state, where: str) -> None:
    """Fail fast on common state-shape bugs."""
    assert isinstance(state, dict), f"{where}: state must be a dict-like object"

    if "sents" in state:
        assert isinstance(state["sents"], list), f"{where}: state['sents'] must be a list"

    if "labels" in state and "sents" in state:
        assert isinstance(state["labels"], list), f"{where}: state['labels'] must be a list"
        assert len(state["labels"]) == len(state["sents"]), (
            f"{where}: labels len {len(state['labels'])} != sents len {len(state['sents'])}"
        )

    if "rewritten" in state and "sents" in state:
        assert isinstance(state["rewritten"], list), f"{where}: state['rewritten'] must be a list"
        assert len(state["rewritten"]) == len(state["sents"]), (
            f"{where}: rewritten len {len(state['rewritten'])} != sents len {len(state['sents'])}"
        )

    if "draft" in state:
        assert isinstance(state["draft"], str), f"{where}: state['draft'] must be a str"


# ----------------------
# Model loaders + invokers
# ----------------------

# ----------------------
# LLM for planning + question generation/checking (OpenAI-compatible API)
# Default: DeepSeek OpenAI-compatible endpoint.
# Set these env vars:
#   export DEEPSEEK_API_KEY="..."
#   export DEEPSEEK_MODEL="deepseek-chat"   # or "deepseek-reasoner"
#   export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
# ----------------------

def make_openai_compatible_chat(temperature: float = 0.0) -> ChatOpenAI:
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.environ.get("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing API key. Set DEEPSEEK_API_KEY (recommended) or OPENAI_API_KEY in your environment."
        )
    # Ensure the underlying `openai` client can always see a key, even if LangChain
    # version-specific field names change.
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_BASE_URL", base_url)

    # langchain_openai uses pydantic; accepted field names vary across versions.
    # Prefer model_fields / __fields__ (accurate for pydantic) and fall back to signature.
    fields = set(getattr(ChatOpenAI, "model_fields", {}).keys())
    if not fields:
        fields = set(getattr(ChatOpenAI, "__fields__", {}).keys())

    sig = inspect.signature(ChatOpenAI.__init__)
    params = set(sig.parameters.keys())

    def accept(name: str) -> bool:
        return (name in fields) or (name in params)

    kwargs: Dict[str, Any] = {}

    # Temperature
    if accept("temperature"):
        kwargs["temperature"] = temperature

    # Model
    if accept("model"):
        kwargs["model"] = model
    elif accept("model_name"):
        kwargs["model_name"] = model

    # API key
    if accept("api_key"):
        kwargs["api_key"] = api_key
    if accept("openai_api_key"):
        kwargs["openai_api_key"] = api_key

    # Base URL
    if accept("base_url"):
        kwargs["base_url"] = base_url
    if accept("openai_api_base"):
        kwargs["openai_api_base"] = base_url

    return ChatOpenAI(**kwargs)


chat = make_openai_compatible_chat(temperature=0)

# ----------------------
# DeepSeek text invoker (used ONLY for qgen/answer_check)
# ----------------------
_openai_text_prompt = ChatPromptTemplate.from_messages([
    ("user", "{prompt}")
])
_openai_text_chain = _openai_text_prompt | chat | StrOutputParser()


class OpenAIInvoker:
    """Minimal wrapper to match `.invoke(prompt)` used throughout the pipeline."""

    def invoke(self, prompt: str) -> str:
        return _openai_text_chain.invoke({"prompt": prompt}).strip()

# ----------------------
# Local fine-tuned Flan-T5 (cheap rewrites)
# ----------------------
# Point this to the local fine-tuned checkpoint directory.

def _current_flan_env():
    """Read Flan paths/debug flags from the current process environment.
    """
    local_path = os.environ.get("FLAN_LOCAL_PATH", "google/flan-t5-base")
    base_model = os.environ.get("FLAN_BASE_MODEL", "google/flan-t5-base")
    debug = os.environ.get("FLAN_DEBUG", "0") == "1"
    return local_path, base_model, debug

# Cache for lazy-loaded tokenizer/model + the env they were loaded under.
_flan_tokenizer = None
_flan_model = None
_flan_loaded_env = None  # (local_path, base_model)

def get_flan_model():
    """Lazy-load Flan-T5 once.

    Supports two formats:
      1) Full checkpoint directory / HF repo at FLAN_LOCAL_PATH (has config.json)
      2) Adapter-only (PEFT/LoRA) directory at FLAN_LOCAL_PATH (has adapter_config.json)
         In this case, we load FLAN_BASE_MODEL and apply the adapter.
    """
    global _flan_tokenizer, _flan_model
    local_path, base_model, debug = _current_flan_env()

    # If the environment changed since last load, force a reload.
    global _flan_loaded_env
    if _flan_loaded_env is not None and _flan_loaded_env != (local_path, base_model):
        _flan_tokenizer = None
        _flan_model = None

    if _flan_tokenizer is not None and _flan_model is not None:
        return _flan_tokenizer, _flan_model

    p = Path(local_path)
    has_config = (p / "config.json").exists() if p.exists() else False
    has_adapter = (p / "adapter_config.json").exists() if p.exists() else False

    if debug:
        print(f"[flan] (load) FLAN_LOCAL_PATH={local_path}")
        print(f"[flan] (load) exists={p.exists()} has_config={has_config} has_adapter={has_adapter} FLAN_BASE_MODEL={base_model}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if has_config:
        # Full fine-tuned checkpoint
        _flan_tokenizer = AutoTokenizer.from_pretrained(local_path)
        _flan_model = AutoModelForSeq2SeqLM.from_pretrained(local_path)
    elif has_adapter:
        # Adapter-only folder: load base and apply adapter
        _flan_tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        base.to(device)
        base.eval()
        _flan_model = PeftModel.from_pretrained(base, local_path)
        # Merge for faster inference and to avoid PEFT-specific forward hooks
        try:
            _flan_model = _flan_model.merge_and_unload()
        except Exception:
            print("[flan] (load) warning: failed to merge LoRA adapter; using non-merged model")
            pass
    else:
        # If the path exists locally but is not a valid model/adapter folder, fail fast.
        if p.exists():
            try:
                preview = [x.name for x in list(p.iterdir())[:25]]
            except Exception:
                preview = ["<unable to list dir>"]
            raise ValueError(
                "FLAN_LOCAL_PATH points to an existing local directory, but it does not look like a HF checkpoint "
                "(missing config.json) or an adapter-only folder (missing adapter_config.json). "
                f"FLAN_LOCAL_PATH={local_path} contents={preview}"
            )
        # Otherwise, treat FLAN_LOCAL_PATH as an HF repo id
        _flan_tokenizer = AutoTokenizer.from_pretrained(local_path)
        _flan_model = AutoModelForSeq2SeqLM.from_pretrained(local_path)

    _flan_model.to(device)
    _flan_model.eval()
    _flan_loaded_env = (local_path, base_model)
    return _flan_tokenizer, _flan_model

class LocalFlan:
    """Minimal wrapper to match `.invoke(prompt)` used throughout the pipeline."""

    def __init__(
        self,
        max_input_tokens: int = 512,
        max_new_tokens: int = 600,
    ):
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    def invoke(self, prompt: str) -> str:
        tok, model = get_flan_model()
        device = next(model.parameters()).device
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return tok.decode(out_ids[0], skip_special_tokens=True).strip()


# Global instance used by rewrite/repair/stitch nodes (local fine-tuned Flan-T5).
flan = LocalFlan()

# Separate OpenAI invoker used ONLY for qgen/answer_check.
openai_text = OpenAIInvoker()

# ----------------------
# QSalience (question salience scoring)
# ----------------------
# From the QSalience repo README, their fine-tuned Flan-T5 model is available on HF.
QSAL_MODEL_NAME = "lingchensanwen/t5_model_1st"  # flan-t5-base fine-tuned for salience
QSAL_MAX_INPUT_CHARS = 4500  # cheap guardrail; T5 has limited context

_qsal_tokenizer = None
_qsal_model = None


def get_qsalience_model():
    """Lazy-load QSalience model once."""
    global _qsal_tokenizer, _qsal_model
    if _qsal_model is not None and _qsal_tokenizer is not None:
        return _qsal_tokenizer, _qsal_model

    _qsal_tokenizer = AutoTokenizer.from_pretrained(QSAL_MODEL_NAME)
    _qsal_model = AutoModelForSeq2SeqLM.from_pretrained(QSAL_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    _qsal_model.to(device)
    _qsal_model.eval()
    return _qsal_tokenizer, _qsal_model


def _format_qsalience_input(article: str, question: str) -> str:
    # Matches the paper-style format (instruction-free form also works well for the released T5 checkpoint).
    # Keep it terse to fit context limits.
    article = article.strip().replace("\n", " ")
    question = question.strip().replace("\n", " ")
    if len(article) > QSAL_MAX_INPUT_CHARS:
        article = article[:QSAL_MAX_INPUT_CHARS]
    return f"article: {article}\nquestion: {question}\n"


def predict_qsalience(article: str, question: str) -> Optional[int]:
    """Return an integer salience score in [1,5] if parseable, else None."""
    tok, model = get_qsalience_model()
    device = next(model.parameters()).device
    inp = _format_qsalience_input(article, question)
    inputs = tok(inp, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )
    text = tok.decode(out_ids[0], skip_special_tokens=True).strip()
    # Robust parse: grab first digit 1-5
    m = re.search(r"\b([1-5])\b", text)
    if not m:
        return None
    return int(m.group(1))


# ----------------------
# CEFR classifier (AbdullahBarayan ModernBERT)
# ----------------------
# Model: https://huggingface.co/AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr
CEFR_MODEL_NAME = "AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr"

_cefr_tokenizer = None
_cefr_model = None

# Order used to compare difficulty; higher means more advanced.
_CEFR_ORDER = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}

def get_cefr_model():
    """Lazy-load CEFR classifier once."""
    global _cefr_tokenizer, _cefr_model
    if _cefr_tokenizer is not None and _cefr_model is not None:
        return _cefr_tokenizer, _cefr_model

    _cefr_tokenizer = AutoTokenizer.from_pretrained(CEFR_MODEL_NAME)
    _cefr_model = AutoModelForSequenceClassification.from_pretrained(CEFR_MODEL_NAME)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    _cefr_model.to(device)
    _cefr_model.eval()
    return _cefr_tokenizer, _cefr_model


# ----------------------
# Pure text utils
# ----------------------

# Sentence splitting helpers (chunk -> sentences)
# ----------------------
# Split on:
#  - one-or-more newlines, OR
#  - sentence-ending punctuation followed by whitespace, OR
#  - sentence-ending punctuation followed by a quote/bracket then whitespace, OR
#  - sentence-ending punctuation followed by an uppercase letter/number/quote (no whitespace)
_SENT_SPLIT_REGEX = re.compile(
    r"(?:\n+|(?<=[.!?])\s+|(?<=[.!?][\"\”\’\'\)\]])\s+|(?<=[.!?])(?=[\"\”\’\'\)\]]?[A-Z0-9]))"
)


def split_chunk_into_sentences(text: str) -> List[str]:
    """Split a pre-chunked text block into sentences.

    This pipeline operates on sentence lists (state['sents']). If your data is already
    chunked by tokens (e.g., ~512 tokens per chunk), pass each chunk string here.

    Prefer NLTK if available (better boundaries); fall back to regex.
    We do NOT download NLTK data at runtime.
    """
    t = (text or "").strip()
    if not t:
        return []

    # Prefer NLTK if available.
    try:
        sents = sent_tokenize(t)
        sents = [s.strip() for s in sents if s and s.strip()]
        if sents:
            return sents
    except Exception:
        pass

    # Fallback: handle punctuation boundaries and newlines.
    sents = _SENT_SPLIT_REGEX.split(t)
    sents = [s.strip() for s in sents if s and s.strip()]

    # Second-pass split if we still only got one sentence but punctuation suggests more.
    if len(sents) <= 1 and (t.count(".") + t.count("?") + t.count("!") >= 2):
        sents = re.split(r"(?<=[.!?])(?=[A-Z\"\”\’\'(\[])", t)
        sents = [s.strip() for s in sents if s and s.strip()]

    return sents

# Merge tiny leading quote fragments (common in news writing) into the following sentence.
def _merge_quote_fragments(sents: List[str]) -> List[str]:
    """Merge tiny leading quote fragments (common in news writing) into the following sentence.

    Example: ['“Like this?', 'Never before,” ...'] -> ['“Like this? Never before,” ...']
    """
    out: List[str] = []
    i = 0
    while i < len(sents):
        s = (sents[i] or "").strip()
        if not s:
            i += 1
            continue

        # Heuristic: very short fragment that starts with a quote and has no closing quote.
        if i + 1 < len(sents):
            starts_quote = s[0] in ('"', '“', '‘', "'")
            has_close_quote = any(q in s[1:] for q in ('"', '”', '’', "'"))
            few_words = len(re.findall(r"\w+", s)) <= 3
            if starts_quote and few_words and not has_close_quote:
                nxt = (sents[i + 1] or "").strip()
                if nxt:
                    out.append((s + " " + nxt).strip())
                    i += 2
                    continue

        out.append(s)
        i += 1

    return out  

# Planner JSON cleanup helpers
def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_first_json_array(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r"\[[\s\S]*\]", s)
    return (m.group(0) if m else s).strip()

# JSON parsing helpers
def _safe_json_list(raw: str) -> List[str]:
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            qs = []
            for it in obj:
                if isinstance(it, dict) and isinstance(it.get("q"), str) and it["q"].strip():
                    qs.append(it["q"].strip())
            return qs
    except Exception:
        pass
    return []

def _safe_json_q(raw: str) -> Optional[str]:
    """Parse a single question from either {"q":...} or [{"q":...}, ...]."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("q"), str) and obj["q"].strip():
            return obj["q"].strip()
        if isinstance(obj, list) and obj:
            it = obj[0]
            if isinstance(it, dict) and isinstance(it.get("q"), str) and it["q"].strip():
                return it["q"].strip()
    except Exception:
        pass
    return None

def _safe_json_bool(raw: str) -> Optional[bool]:
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("answered"), bool):
            return obj["answered"]
    except Exception:
        pass
    return None

def _local_context_window(sents: List[str], i: int, window: int = 2) -> str:
    lo = max(0, i - window)
    hi = min(len(sents), i + window + 1)
    ctx = " ".join(s.strip() for s in sents[lo:hi] if s.strip())
    return ctx

# ----------------------
# Planner (labels sentences: copy/rephrase/split/delete)
# ----------------------
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", "You label sentences for CEFR simplification."),
    ("user",
     "Target level: {level}\n"
     "Label each sentence as one of: copy, rephrase, split, delete.\n"
     "Guidelines:\n"
     "- delete: low-importance detail (examples, quotes, extra commentary)\n"
     "- split: very long sentence or multiple clauses\n"
     "- rephrase: keep meaning but simplify\n"
     "- copy: already simple\n\n"
     "Sentences:\n{sentences}\n\n"
     "Return ONLY valid JSON: [{{\"sid\":0,\"label\":\"rephrase\"}}, ...]")
])



planner_chain = planner_prompt |chat | StrOutputParser()


def plan_node(state: SimplifyState) -> SimplifyState:
    _dbg(state, "plan_node:in")
    _assert_invariants(state, "plan_node:in")

    sents = state.get("sents") or []
    sents_block = "\n".join([f"{i}. {s}" for i, s in enumerate(sents)])
    raw = (planner_chain.invoke({"level": state["level"], "sentences": sents_block}) or "").strip()

    raw_clean = _strip_code_fences(raw)
    raw_json = _extract_first_json_array(raw_clean)

    labels = ["rephrase"] * len(sents)
    try:
        items = json.loads(raw_json)
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                sid = it.get("sid")
                lab = (it.get("label") or "").strip().lower()
                if isinstance(sid, int) and 0 <= sid < len(labels) and lab in {"copy", "rephrase", "split", "delete"}:
                    labels[sid] = lab
    except Exception as e:
        print("\n[plan_node] JSON parse failed.")
        print(f"[plan_node] error: {e}")
        print("[plan_node] ----- RAW (verbatim) BEGIN -----")
        print(raw)
        print("[plan_node] ----- RAW (verbatim) END -----")
        print("[plan_node] ----- CLEANED/EXTRACTED JSON (repr) BEGIN -----")
        print(repr(raw_json))
        print("[plan_node] ----- CLEANED/EXTRACTED JSON (repr) END -----\n")

    state["labels"] = labels
    # For debugging readability, initialize rewritten placeholders.
    state["rewritten"] = [""] * len(sents)

    _assert_invariants(state, "plan_node:out")
    _dbg(state, "plan_node:out")
    return state

# ----------------------
# Rewrite helpers
# ----------------------

# Helper to keep local Flan prompts in-distribution (no extra instruction text).
def flan_rewrite_plain(level: str, sent: str) -> str:
    """Local Flan rewrite using ONLY the training-style prefix.

    Important: our fine-tune distribution was `simplify to {level}: <text>`.
    If we prepend extra instructions, the model often copies them into the output.
    """
    sent = (sent or "").strip()
    if not sent:
        return ""
    return flan.invoke(f"simplify to {level}: " + sent).strip()

def split_sentence_openai(level: str, sent: str, extra: str = "") -> str:
    """Split a sentence into shorter sentences using the OpenAI-compatible LLM."""
    sent = (sent or "").strip()
    if not sent:
        return ""
    prompt = (
        f"simplify to {level}: "
        "Split the sentence into 2-4 shorter sentences. Simplify the sentences. Keep facts the same. "
        "Do not add new information. "
        f"{extra}\n\n"
        f"Sentence: {sent}\n\n"
        "Return ONLY the rewritten sentence(s)."
    )
    return openai_text.invoke(prompt).strip()


# --- Helper: split with DeepSeek, then run each split sentence through local Flan ---
def split_then_flan(level: str, sent: str, extra: str = "") -> str:
    """Split with DeepSeek-compatible LLM, then run each split sentence through local fine-tuned Flan."""
    sent = (sent or "").strip()
    if not sent:
        return ""

    split_text = (split_sentence_openai(level, sent, extra=extra) or "").strip()
    if not split_text:
        return ""

    # Break into sentences, then rewrite each piece with local Flan for consistency.
    pieces = split_chunk_into_sentences(split_text)
    pieces = _merge_quote_fragments(pieces)
    if not pieces:
        pieces = [split_text]

    rew_pieces: List[str] = []
    for p in pieces:
        p = (p or "").strip()
        if not p:
            continue
        rew_pieces.append(flan_rewrite_plain(level, p))

    return " ".join([x for x in rew_pieces if x.strip()]).strip()


def rewrite_sentence(level: str, label: str, sent: str) -> str:
    """Rewrite a single sentence based on a planner label."""
    sent = (sent or "").strip()
    if not sent:
        return ""

    lab = (label or "rephrase").strip().lower()

    if lab == "delete":
        return ""
    if lab == "copy":
        return sent
    if lab == "split":
        return split_then_flan(level, sent)

    # Default: rephrase locally (in-distribution prefix)
    return flan_rewrite_plain(level, sent)

# ----------------------
# Rewrite node
# ----------------------
def rewrite_node(state: SimplifyState) -> SimplifyState:
    _dbg(state, "rewrite_node:in")
    _assert_invariants(state, "rewrite_node:in")

    sents = state.get("sents") or []
    labels = state.get("labels")
    if not isinstance(labels, list) or len(labels) != len(sents):
        labels = ["rephrase"] * len(sents)
        state["labels"] = labels

    out: List[str] = []
    for lab, s in zip(labels, sents):
        out.append(rewrite_sentence(state["level"], lab, s))

    state["rewritten"] = out
    _assert_invariants(state, "rewrite_node:out")
    _dbg(state, "rewrite_node:out")
    return state
# ----------------------
# Draft builder node
# ----------------------

def build_draft_node(state: SimplifyState) -> SimplifyState:
    _dbg(state, "build_draft_node:in")
    _assert_invariants(state, "build_draft_node:in")
    # drop empty (deleted) sents
    parts = [s for s in state["rewritten"] if s.strip()]
    state["draft"] = " ".join(parts)
    _assert_invariants(state, "build_draft_node:out")
    _dbg(state, "build_draft_node:out")
    return state

# ----------------------
# Salience helpers
# ----------------------


# Number of candidate questions generated per sentence for salience selection
N_Q_CANDIDATES = 2  # generate N candidate questions per sentence, then select the most salient

def qgen_local_invoke(context: str, anchor: str) -> str:
    """Generate a small list of anchor-grounded questions as JSON.

    Returns a raw JSON string like: [{"q":"..."}, ...]
    We later score each candidate with QSalience and keep only the best one.
    """
    prompt = (
        f"You write {N_Q_CANDIDATES} short questions to test whether a rewrite preserved key info in ONE sentence.\n\n"
        "Optional background context (DO NOT ask about this; it is only to disambiguate pronouns/names):\n"
        f"{context}\n\n"
        "ANCHOR SENTENCE (each question must be about this sentence ONLY):\n"
        f"{anchor}\n\n"
        f"Task: Write exactly {N_Q_CANDIDATES} short questions.\n"
        "Rules:\n"
        "- Each question MUST be answerable using ONLY facts stated in the anchor sentence.\n"
        "- Do NOT ask for causes, reasons, or missing details (no 'why', no 'what factors', etc.).\n"
        "- Do NOT ask about the background context.\n"
        "- Keep each question concise (<= 15 words).\n"
        "Return ONLY valid JSON (no markdown, no extra text): [{\"q\":\"...\"}, ...]"
    )
    # Note: N_Q_CANDIDATES is baked into the instruction; keep it deterministic.
    return openai_text.invoke(prompt).strip()

def answer_check_local_invoke(question: str, draft: str) -> str:
    """Check answeredness using API call; returns raw text (ideally JSON)."""
    prompt = (
        "You check whether a draft answers a given question.\n\n"
        f"Question:\n{question}\n\n"
        f"Simplified draft:\n{draft}\n\n"
        "Does the draft clearly answer the question based only on the draft content?\n"
        "Reply with ONLY JSON (no markdown, no extra text): {\"answered\": true/false}"
    )
    return openai_text.invoke(prompt).strip()

def salience_check_node(state: SimplifyState) -> SimplifyState:
    _dbg(state, "salience_check_node:in")
    _assert_invariants(state, "salience_check_node:in")
    """Integrate QSalience by:
      1) generating a few anchor-grounded candidate questions per sentence,
      2) scoring that question's importance with QSalience (1-5) using the full original text,
      3) checking whether the simplified draft answers the question.

    A sentence is flagged as a salience failure if:
      - the top-salience question has score >= 4, AND
      - the draft does NOT answer it.

    Notes:
      - Questions are generated only once. On subsequent passes, we reuse stored questions
        and only re-check answeredness against the current draft.
    """

    draft = (state.get("draft") or "").strip()
    sents = state.get("sents") or []

    labels = state.get("labels")
    if not isinstance(labels, list):
        labels = ["rephrase"] * len(sents)

    def _check_answered(q: str) -> bool:
        raw = answer_check_local_invoke(question=q, draft=draft)
        ans = _safe_json_bool(raw)
        return bool(ans) if ans is not None else False

    def _record(out_list: List[Dict[str, Any]], sid: int, lab: str, q: str, salience: int, answered: bool) -> None:
        out_list.append({
            "sid": int(sid),
            "label": lab,
            "q": q,
            "salience": int(salience),
            "answered": bool(answered),
        })

    # ----------------------
    # Reuse path: if questions already exist, do NOT regenerate; only re-check answeredness.
    # ----------------------
    stored = state.get("salient_questions") or []
    if stored:
        qs_out: List[Dict[str, Any]] = []
        fail_ids: List[int] = []

        for obj in stored:
            try:
                sid = int(obj.get("sid", -1))
            except Exception:
                sid = -1
            if sid < 0 or sid >= len(sents):
                continue

            q = str(obj.get("q", "")).strip()
            if not q:
                continue

            try:
                score = int(obj.get("salience", 3))
            except Exception:
                score = 3

            # Prefer current planner label if available; fall back to stored.
            lab = labels[sid] if sid < len(labels) else str(obj.get("label", "rephrase"))

            answered = _check_answered(q)
            _record(qs_out, sid, lab, q, score, answered)

            if score >= 4 and not answered:
                fail_ids.append(sid)

        state["salient_questions"] = qs_out
        state["salience_fail_ids"] = sorted(set(fail_ids))
        _assert_invariants(state, "salience_check_node:out")
        print("SALIENT QUESTIONS (one per sentence):", json.dumps(state["salient_questions"], indent=2, ensure_ascii=False))
        _dbg(state, "salience_check_node:out")
        return state

    # ----------------------
    # First-time generation path
    # ----------------------
    qs_out: List[Dict[str, Any]] = []
    fail_ids: List[int] = []

    for i, (lab, sent) in enumerate(zip(labels, sents)):

        # Small local window only to disambiguate names/pronouns; question must stay anchor-grounded
        context = _local_context_window(sents, i, window=1)

        # Generate a few anchor-grounded candidate questions
        raw_qs = qgen_local_invoke(context=context, anchor=sent)
        cand_qs = _safe_json_list(raw_qs)
        if not cand_qs:
            # Fallback: keeps the pipeline running.
            cand_qs = ["What does this sentence say?"]

        # Use broader context ONLY to score importance (which decides whether failure matters)
        article_for_scoring = " ".join(s.strip() for s in sents if s.strip())

        best_q = cand_qs[0]
        best_score = None
        for q in cand_qs:
            s = predict_qsalience(article_for_scoring, q)
            if s is None:
                continue
            if (best_score is None) or (int(s) > int(best_score)):
                best_score = int(s)
                best_q = q

        # If scoring failed for all candidates, default to 3.
        top_score = int(best_score) if best_score is not None else 3

        answered = _check_answered(best_q)
        _record(qs_out, i, lab, best_q, int(top_score), answered)

        if int(top_score) >= 4 and not answered:
            fail_ids.append(i)

    state["salient_questions"] = qs_out
    state["salience_fail_ids"] = sorted(set(fail_ids))
    _assert_invariants(state, "salience_check_node:out")
    print("SALIENT QUESTIONS (one per sentence):", json.dumps(state["salient_questions"], indent=2, ensure_ascii=False))
    _dbg(state, "salience_check_node:out")
    return state

# ----------------------
# CEFR check node
# ----------------------

def cefr_check_node(state: SimplifyState) -> SimplifyState:
    """Document-level CEFR gate: classify the whole draft once and, if it fails, simplify all sentences.

    Rationale: many CEFR classifiers (especially "doc" checkpoints) are unreliable on single sentences.

    Behavior:
      - Predict CEFR on the full draft (or concatenated rewritten text).
      - If predicted level is above the target, mark *all* sentences as failing (cefr_fail_ids = all ids).
      - Otherwise, cefr_fail_ids = [].
    """
    _dbg(state, "cefr_check_node:in")
    _assert_invariants(state, "cefr_check_node:in")

    target = str(state.get("level", "")).strip().upper()
    if target not in _CEFR_ORDER:
        state["cefr_pred"] = None
        state["cefr_ok"] = False
        state["cefr_fail_ids"] = list(range(len(state.get("sents") or [])))
        state["cefr_sent_preds"] = []
        _dbg(state, "cefr_check_node:out")
        return state

    sents = state.get("sents") or []
    # Prefer the built draft; fall back to joining rewritten sentences.
    draft = (state.get("draft") or "").strip()
    if not draft:
        rewritten = state.get("rewritten") or []
        if isinstance(rewritten, list) and rewritten:
            draft = " ".join([x.strip() for x in rewritten if isinstance(x, str) and x.strip()]).strip()

    if not draft:
        # If we have no text to classify, fail conservatively.
        state["cefr_pred"] = None
        state["cefr_ok"] = False
        state["cefr_fail_ids"] = list(range(len(sents)))
        state["cefr_sent_preds"] = []
        _dbg(state, "cefr_check_node:out")
        return state

    tok, model = get_cefr_model()
    device = next(model.parameters()).device
    id2label = getattr(model.config, "id2label", {}) or {}

    inputs = tok(draft, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        pred_id = int(logits.argmax(-1).item())

    pred_level = id2label.get(pred_id) or id2label.get(str(pred_id))
    if not isinstance(pred_level, str):
        pred_level = str(pred_id)
    pred_level = pred_level.strip().upper()

    # Conservative: unknown label counts as failure.
    is_ok = (pred_level in _CEFR_ORDER) and (_CEFR_ORDER[pred_level] <= _CEFR_ORDER[target])

    state["cefr_sent_preds"] = [pred_level] * len(sents)
    state["cefr_ok"] = bool(is_ok)
    state["cefr_fail_ids"] = [] if is_ok else list(range(len(sents)))
    state["cefr_pred"] = f"doc={pred_level}"

    _dbg(state, "cefr_check_node:out")
    return state

# ----------------------
# Repair node + counters + Stitch node
# ----------------------

def repair_cefr_node(state: SimplifyState) -> SimplifyState:
    """Repair ONLY the sentences that failed the CEFR sentence-level check.

    This node focuses on simplifying language (keep prompts in-distribution) and does NOT
    enforce salience constraints.
    """
    _dbg(state, "repair_cefr_node:in")
    _assert_invariants(state, "repair_cefr_node:in")

    ids = [] if state.get("cefr_ok", True) else (state.get("cefr_fail_ids") or [])
    ids = sorted({int(i) for i in ids if isinstance(i, (int, str)) and str(i).isdigit() and 0 <= int(i) < len(state.get("sents") or [])})

    if not ids:
        _assert_invariants(state, "repair_cefr_node:out")
        _dbg(state, "repair_cefr_node:out")
        return state

    for i in ids:
        orig = state["sents"][i]
        labels = state.get("labels")
        lab = labels[i] if isinstance(labels, list) and i < len(labels) else "rephrase"

        # For CEFR repair we prefer to simplify the *current* rewritten text (if any),
        # so we don't undo earlier repairs.
        cur = ""
        try:
            cur = (state.get("rewritten") or [""])[i] or ""
        except Exception:
            cur = ""

        base_text = (cur.strip() if isinstance(cur, str) and cur.strip() else (orig or "").strip())

        # Keep things clean and in-distribution.   
        state["rewritten"][i] = flan_rewrite_plain(state["level"], base_text)


    _assert_invariants(state, "repair_cefr_node:out")
    _dbg(state, "repair_cefr_node:out")
    return state

def repair_node(state: SimplifyState) -> SimplifyState:
    _dbg(state, "repair_node:in")
    _assert_invariants(state, "repair_node:in")
    # Salience repair ONLY: fix sentences whose important information was lost.
    salience_ids = state.get("salience_fail_ids") or []

    ids_set = set()
    for x in list(salience_ids):
        try:
            i = int(x)
        except Exception:
            continue
        if 0 <= i < len(state.get("sents") or []):
            ids_set.add(i)

    ids = sorted(ids_set)

    if not ids:
        _assert_invariants(state, "repair_node:out")
        _dbg(state, "repair_node:out")
        return state

    for i in ids:
        sent = state["sents"][i]
        labels = state.get("labels")
        lab = labels[i] if isinstance(labels, list) and i < len(labels) else "rephrase"

        # Try to pull the stored salient question for this sentence (if we have it)
        salient_q = None
        for obj in state.get("salient_questions", []):
            if obj.get("sid") == i:
                salient_q = obj.get("q")
                break

        extra = f"Make sure the rewrite answers this question: {salient_q}. " if salient_q else ""

        # Preserve control labels: do not downgrade 'split' to 'rephrase'.
        if lab == "split":
            # Split with OpenAI-compatible LLM, then run split pieces through local Flan.
            state["rewritten"][i] = split_then_flan(state["level"], sent, extra=extra)
        else:
            # Keep local Flan prompts in-distribution. If we need to enforce a constraint
            # (e.g., salience question), route that sentence to OpenAI instead.
            if salient_q:
                # Use OpenAI-compatible chat for constrained repairs.
                prompt = (
                    f"simplify to {state['level']}: "
                    "Rewrite the sentence using simple words and short sentences. "
                    "Keep facts the same. Do not add new information. "
                    f"The rewrite MUST clearly answer this question: {salient_q}\n\n"
                    f"Sentence: {sent}\n\n"
                    "Return ONLY the rewritten sentence(s)."
                )
                state["rewritten"][i] = openai_text.invoke(prompt).strip()
            else:
                # fallback: plain Flan rewrite
                state["rewritten"][i] = flan_rewrite_plain(state["level"], sent)

    _assert_invariants(state, "repair_node:out")
    _dbg(state, "repair_node:out")
    return state

def stitch_node(state: SimplifyState) -> SimplifyState:
    """_dbg(state, "stitch_node:in")
    _assert_invariants(state, "stitch_node:in")
    prompt = (
        f"simplify to {state['level']}: "
        "Rewrite the text into one coherent article. Remove repeats. Fix unclear pronouns by using names. "
        "It is OK to drop low-importance details. Keep facts the same. Do not add new information.\n\n"
        f"{state['draft'].strip()}"
    )
    state["draft"] = flan.invoke(prompt).strip()
    _assert_invariants(state, "stitch_node:out")
    _dbg(state, "stitch_node:out")"""
    return state

# Max attempts for each repair type.
MAX_CEFR_REPAIRS = 3
MAX_SALIENCE_REPAIRS = 3
# Hard cap on total control-loop iterations to prevent infinite looping.
MAX_TOTAL_LOOPS = 6

def init_passes(state: SimplifyState) -> SimplifyState:
    # `passes` counts CEFR repair attempts taken.
    state["passes"] = 0
    # `salience_passes` counts salience repair attempts taken.
    state["salience_passes"] = 0
    # `loops` caps total iterations across all repairs.
    state["loops"] = 0
    # Routing decision written by `control_node`.
    state["next_step"] = "stitch"
    return state

def control_node(state: SimplifyState) -> SimplifyState:
    """Decide the next action after running BOTH checks (salience + CEFR).

    Goal: keep iterating until BOTH constraints are satisfied:
      - CEFR passes AND
      - no salience failures.

    We prioritize salience repair when salience is failing (even if CEFR is also failing),
    because losing key information is usually harder to recover after aggressive simplification.
    """
    state["loops"] = state.get("loops", 0) + 1

    cefr_ok = bool(state.get("cefr_ok", False))
    salience_fails = bool(state.get("salience_fail_ids"))

    # Success: both constraints satisfied.
    if cefr_ok and (not salience_fails):
        state["next_step"] = "stitch"
        return state

    # Safety cap: stop looping even if not perfect.
    if state.get("loops", 0) >= MAX_TOTAL_LOOPS:
        state["next_step"] = "stitch"
        return state

    # If key info is missing, try to restore it first (bounded).
    if salience_fails and state.get("salience_passes", 0) < MAX_SALIENCE_REPAIRS:
        state["salience_passes"] = state.get("salience_passes", 0) + 1
        state["next_step"] = "repair"
        return state

    # Otherwise, if CEFR is too hard, simplify further (bounded).
    if (not cefr_ok) and state.get("passes", 0) < MAX_CEFR_REPAIRS:
        state["passes"] = state.get("passes", 0) + 1
        state["next_step"] = "repair_cefr"
        return state

    # If we exhausted the relevant repair budget(s), stop.
    state["next_step"] = "stitch"
    return state


def route_after_control(state: SimplifyState) -> str:
    choice = (state.get("next_step") or "stitch").strip()
    if choice not in {"repair", "repair_cefr", "stitch"}:
        choice = "stitch"
    if DEBUG:
        print(
            f"[route_after_control] loops={state.get('loops')} passes={state.get('passes')} salience_passes={state.get('salience_passes')} "
            f"cefr_ok={state.get('cefr_ok')} salience_fail_ids={state.get('salience_fail_ids')} -> {choice}"
        )
    return choice


# ----------------------
# State graph definition
# ----------------------

g = StateGraph(SimplifyState)

g.add_node("init", init_passes)
g.add_node("plan", plan_node)
g.add_node("rewrite", rewrite_node)
g.add_node("build", build_draft_node)
g.add_node("cefr", cefr_check_node)
g.add_node("salience", salience_check_node)
g.add_node("repair", repair_node)
g.add_node("repair_cefr", repair_cefr_node)
g.add_node("stitch", stitch_node)
g.add_node("control", control_node)

g.set_entry_point("init")
g.add_edge("init", "plan")
g.add_edge("plan", "rewrite")
g.add_edge("rewrite", "build")
# Option B: run salience before CEFR so questions are generated once and can guide later repairs.
g.add_edge("build", "salience")
g.add_edge("salience", "cefr")
g.add_edge("cefr", "control")

g.add_conditional_edges("control", route_after_control, {
    "repair": "repair",
    "repair_cefr": "repair_cefr",
    "stitch": "stitch",
})
g.add_edge("repair", "build")   # rebuild draft after repair
g.add_edge("repair_cefr", "build")   # rebuild draft after CEFR-only repair
g.add_edge("stitch", END)

app = g.compile()


def simplify_chunk(level: str, chunk_text: str) -> str:
    """Run the LangGraph simplify pipeline on ONE pre-chunked text string."""
    sents = split_chunk_into_sentences(chunk_text)
    sents = _merge_quote_fragments(sents)
    state: SimplifyState = {
        "level": level,
        "sents": sents,
    }
    out = app.invoke(state)
    return (out.get("draft") or "").strip()

# ----------------------
# Quick local debugging helpers
# ----------------------

if __name__ == "__main__":
    level = os.environ.get("CEFR_LEVEL", "A2")

    # Provide one already-token-chunked text block via env var CHUNK_TEXT.
    chunk_text = os.environ.get("CHUNK_TEXT", "").strip()
    if not chunk_text:
        raise ValueError(
            "CHUNK_TEXT is not set (or is empty). Example usage:\n"
            "  CEFR_LEVEL=A2 CHUNK_TEXT=\"...\" python langgraph_pipeline.py"
        )

    print(simplify_chunk(level=level, chunk_text=chunk_text))
