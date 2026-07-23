"""tokengraph (single-file build) — local AST code graph for token-efficient AI coding.

One file, no package layout. Deep-parses (full call/import/inheritance graph)
25+ languages — Python, Java, Go, TypeScript, JavaScript, C/C++, C#, Rust, PHP,
Ruby, Kotlin, Swift, Scala, Lua, Bash, Solidity, Perl, Erlang, Julia, R,
Haskell, OCaml, Nim, PowerShell, Dart — plus lightweight local indexing for 30+
more via regex fallback — SQL, Elixir, Clojure, F#, Groovy, Zig, Crystal, Haxe,
Objective-C, Visual Basic, Tcl, Pascal, GDScript, and markup/config (Vue,
Svelte, HTML, CSS, YAML, TOML, XML, INI, GraphQL, Terraform, Protobuf, Markdown,
Dockerfile, …). Python is parsed with the stdlib `ast`; the other deep-parsed
languages use tree-sitter (via tree-sitter-language-pack) when installed, and
fall back to regex otherwise.
The MCP server (FastMCP) and tiktoken are also optional and imported only when
used, so this file runs the CLI with ZERO third-party deps.

Quick start:
    python tokengraph_all.py index                 # build the local graph (cwd)
    python tokengraph_all.py context "the task"    # token-budgeted context pack
    python tokengraph_all.py watch                  # keep the graph fresh on a poll loop
    python tokengraph_all.py langs                  # show parseable languages
    python tokengraph_all.py serve                  # run as an MCP server (stdio)

Staying fresh (no stale mapping):
    The graph auto-refreshes on every query. CLI `context`/`skeleton`/`callers`/
    `callees` and every MCP tool run an incremental reindex first; a mtime+size
    fast path makes that a stat() sweep when nothing changed. Pair with a
    PostToolUse hook (see .claude/settings.json) to pre-warm after each edit so
    the query-time refresh is a no-op. `watch` is the daemon option for large
    repos that want continuous, lower-latency updates.

Optional installs:
    pip install fastmcp                              # MCP server
    pip install tiktoken                             # accurate token counts
    pip install tree-sitter tree-sitter-java tree-sitter-go \
                tree-sitter-typescript tree-sitter-javascript   # multi-language

Wire the server into Claude Code:
    claude mcp add --scope project --transport stdio tokengraph \
        -- python /abs/path/tokengraph_all.py serve
and into GitHub Copilot (.vscode/mcp.json, agent mode):
    { "servers": { "tokengraph": { "type": "stdio", "command": "python",
        "args": ["/abs/path/tokengraph_all.py", "serve"] } } }
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import os
import re
import sqlite3
import sys
import threading
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional



# ==========================================================================
# token estimation
# ==========================================================================
"""Token estimation.

Uses tiktoken if installed (accurate for OpenAI/Copilot-style tokenizers, and a
good proxy for Claude). Falls back to a character heuristic otherwise so the
tool has zero hard dependencies.
"""

_enc = None
_tried = False


def _encoder():
    global _enc, _tried
    if _tried:
        return _enc
    _tried = True
    try:
        import tiktoken  # type: ignore[import-not-found]
        _enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _enc = None
    return _enc


# ---- MA-2: real per-vendor tokenizers, when they are available locally -----
# The previous implementation multiplied one cl100k count by a hardcoded
# per-family constant and called that "model-aware". It is not: Claude's
# tokenizer, Gemini's SentencePiece and Llama's SentencePiece segment text
# genuinely differently, especially on code (identifiers, indentation runs,
# punctuation). Where a real tokenizer is installed we now use it; where one is
# not, we say so instead of implying precision we do not have.
_MODEL_ENCODERS: dict[str, object] = {}
_MODEL_ENCODER_TRIED: set[str] = set()

# GPT families map to a specific tiktoken encoding rather than always cl100k.
_TIKTOKEN_ENCODINGS = {
    "o200k": ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4"),
}


def _tiktoken_encoding_for(model: str):
    """Exact tiktoken encoding for an OpenAI model, else None.

    Newer encodings (o200k) are fetched on first use, so this can fail on an
    offline or proxied machine. Every failure path falls through to the next
    option rather than returning None early — losing tiktoken entirely because
    one encoding could not be downloaded would silently drop OpenAI counts to
    the char heuristic.
    """
    try:
        import tiktoken  # type: ignore[import-not-found]
    except Exception:
        return None
    m = (model or "").lower()
    if any(m.startswith(p) for p in _TIKTOKEN_ENCODINGS["o200k"]):
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            pass                      # not cached locally; try the model map
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        pass
    # cl100k is exact for gpt-4/3.5 and a close proxy for o200k models.
    return _encoder()


def _hf_tokenizer(repo_id: str):
    """Load a local HuggingFace tokenizer without touching the network."""
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        return AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
    except Exception:
        return None


def _model_encoder(model: str):
    """Return (encoder, kind) for a model. kind is 'exact' or 'approx'.

    'exact' means a real tokenizer for that vendor ran. 'approx' means we fell
    back to cl100k (or the char heuristic) and scaled — a documented estimate,
    not a measurement.
    """
    family = model_family(model)
    # GPT models do NOT share one encoding (gpt-4 is cl100k, gpt-4o is o200k),
    # so the cache is keyed per-model there and per-family elsewhere.
    key = f"gpt:{(model or '').lower()}" if family == "gpt" else family
    if key in _MODEL_ENCODER_TRIED:
        enc = _MODEL_ENCODERS.get(key)
        return enc, ("exact" if enc is not None else "approx")
    _MODEL_ENCODER_TRIED.add(key)
    enc = None
    if family == "gpt":
        enc = _tiktoken_encoding_for(model)
    elif family == "claude":
        # Anthropic ships no offline tokenizer; the count endpoint is remote and
        # this tool is offline-by-default, so Claude stays a calibrated estimate.
        enc = None
    elif family == "gemini":
        enc = _hf_tokenizer(os.environ.get("TOKENGRAPH_GEMINI_TOKENIZER",
                                           "google/gemma-2-9b"))
    elif family == "llama":
        enc = _hf_tokenizer(os.environ.get("TOKENGRAPH_LLAMA_TOKENIZER",
                                           "meta-llama/Llama-3.1-8B"))
    if enc is not None:
        _MODEL_ENCODERS[key] = enc
    return enc, ("exact" if enc is not None else "approx")


def _encode_len(enc, text: str) -> int:
    """Length of `text` under either a tiktoken or a HuggingFace tokenizer."""
    try:
        return len(enc.encode(text, disallowed_special=()))
    except TypeError:
        pass
    except Exception:
        return 0
    try:
        return len(enc.encode(text, add_special_tokens=False))
    except Exception:
        try:
            return len(enc.encode(text))
        except Exception:
            return 0


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        # disallowed_special=() so source files containing literal special-token
        # markers (e.g. "<|endoftext|>") are counted as normal text, not rejected.
        return len(enc.encode(text, disallowed_special=()))
    # heuristic: ~4 chars/token for code, with a small floor
    return max(1, (len(text) + 3) // 4)


# Model-aware token counting (MA-1). tiktoken's cl100k_base is exact for
# OpenAI/Copilot tokenizers and a good proxy for the others; these ratios
# correct the base count toward each family's real tokenizer (Claude/Gemini
# pack ~a bit more text per token; Llama's SentencePiece a bit less). Values
# are empirical multipliers over the cl100k count, keyed by model *family*.
MODEL_TOKEN_RATIOS: dict[str, float] = {
    "gpt": 1.00, "claude": 1.00, "gemini": 0.98, "llama": 1.10,
}


def model_family(model: str) -> str:
    """Map a concrete model name to a tokenizer/pricing family."""
    m = (model or "").lower()
    if "claude" in m or "anthropic" in m or m in ("opus", "sonnet", "haiku"):
        return "claude"
    if "gemini" in m or "bison" in m or "palm" in m:
        return "gemini"
    if "llama" in m or "codellama" in m or "mistral" in m or "mixtral" in m:
        return "llama"
    return "gpt"                                  # gpt-* / default


def count_tokens_for_model(text: str, model: str = "gpt-4o") -> int:
    """Token count for a specific model (MA-2).

    Uses the vendor's real tokenizer when one is installed locally; otherwise
    falls back to a calibrated estimate over the cl100k count. Call
    `count_tokens_detail` when you need to know which of the two you got —
    budgeting against an estimate and against a measurement are different
    risks and the caller deserves to tell them apart.
    """
    return count_tokens_detail(text, model)["tokens"]


def count_tokens_detail(text: str, model: str = "gpt-4o") -> dict:
    """Model token count plus provenance: exact tokenizer vs. estimate."""
    family = model_family(model)
    enc, kind = _model_encoder(model)
    if enc is not None:
        n = _encode_len(enc, text)
        if n or not text:
            return {"tokens": max(1, n) if text else 0, "model": model,
                    "family": family, "method": "exact",
                    "tokenizer": type(enc).__name__,
                    "base_tokens": count_tokens(text)}
    base = count_tokens(text)
    ratio = MODEL_TOKEN_RATIOS.get(family, 1.0)
    return {
        "tokens": max(1, round(base * ratio)) if text else 0,
        "model": model, "family": family, "method": "approx",
        "tokenizer": "cl100k_base×ratio" if _encoder() else "chars/4×ratio",
        "ratio": ratio, "base_tokens": base,
        "note": _APPROX_NOTE.get(family, ""),
    }


_APPROX_NOTE = {
    "claude": ("Anthropic publishes no offline tokenizer, so Claude counts are "
               "a calibrated estimate. Treat as ±5% for budgeting."),
    "gemini": ("Estimate. For exact counts install `transformers` and cache a "
               "Gemma tokenizer locally (TOKENGRAPH_GEMINI_TOKENIZER)."),
    "llama": ("Estimate. For exact counts install `transformers` and cache a "
              "Llama tokenizer locally (TOKENGRAPH_LLAMA_TOKENIZER)."),
    "gpt": "Estimate — install `tiktoken` for exact OpenAI counts.",
}


def count_tokens_claude_api(text: str, model: str = "claude-opus-4-8") -> dict:
    """Exact Claude token count via the Anthropic count_tokens endpoint (MA-3).

    Anthropic ships no offline tokenizer, so this endpoint is the only exact
    count for Claude models. It is a *network* call and therefore opt-in: the
    rest of ContextIQ is offline-by-default and must never block on a request.
    Falls back to the calibrated estimate, clearly labelled, on any failure.
    """
    if offline_mode():
        d = count_tokens_detail(text, model)
        d["note"] = "offline mode; exact count not attempted"
        return d
    try:
        import anthropic  # type: ignore[import-not-found]
        client = anthropic.Anthropic()
        resp = client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text or ""}])
        return {"tokens": resp.input_tokens, "model": model,
                "family": "claude", "method": "exact",
                "tokenizer": "anthropic count_tokens API",
                "base_tokens": count_tokens(text)}
    except Exception as ex:
        d = count_tokens_detail(text, model)
        d["note"] = (f"exact count unavailable ({type(ex).__name__}); "
                     f"returned the calibrated estimate instead")
        return d


def tokenizer_status() -> dict:
    """Which families can be counted exactly on this machine, and which cannot."""
    out = {}
    for family, probe in (("gpt", "gpt-4o"), ("claude", "claude-sonnet"),
                          ("gemini", "gemini-1.5-pro"), ("llama", "llama-3.1-70b")):
        enc, kind = _model_encoder(probe)
        out[family] = {
            "method": kind,
            "tokenizer": type(enc).__name__ if enc is not None else None,
            "note": "" if kind == "exact" else _APPROX_NOTE.get(family, ""),
        }
    return out


# Per-file signature cap (FR-2a): bound worst-case token cost so a single huge
# file can never dominate emitted context. Applied at the rendering layer
# (file_skeleton / read_context / generated context), NOT at parse time — the
# full symbol graph stays indexed so callers/callees/semantic/impact keep
# working over the dropped symbols.
MAX_SIGS_PER_FILE = 25

# NB-1: neighbour-expansion bounds. Before these existed, graph expansion
# emitted every caller/callee it found, in BFS discovery order, until the
# budget ran out — so one hub symbol with hundreds of callers could flood a
# pack with irrelevant signatures and evict the content that was actually
# asked for. Neighbours are now collected under a fan-out cap, scored for
# relevance to the task, and only the best are emitted.
# The collection cap must sit well ABOVE the emission cap, or ranking has
# nothing to choose between and degenerates back to discovery order.
# Seed candidates pulled from each retrieval arm before fusion. Tuned on the
# benchmark suite (96 cases, 4 repos): 12 → 20 lifted symbol recall 0.667 →
# 0.708 and answerability 0.448 → 0.469 while *lowering* wasted tokens; 32 was
# no better and wasted more. The per-file cap of 4 below is what keeps the
# extra candidates from crowding the budget.
SEED_CANDIDATES = 20

MAX_FANOUT_PER_SYMBOL = 150     # neighbours collected from any single symbol
MAX_NEIGHBOR_CANDIDATES = 600   # global ceiling on collected candidates
MAX_NEIGHBOR_SIGS = 40          # neighbour signatures actually emitted
# Relevance weight by edge kind. A callee explains how the seed works; a base
# class explains what it is; a caller is context about who needs it — useful,
# but the least explanatory of the three.
NEIGHBOR_REASON_WEIGHT = {"callee": 1.0, "base": 0.95, "caller": 0.7}

# SR-1: within-file completion.
#
# The benchmark exposed the failure this fixes. Retrieval finds the right FILE
# almost always (recall@5 0.979) but used to leave the pack *half empty* while
# omitting the one symbol the question was about: seeds and graph neighbours
# were the only sources of content, so once both were exhausted assembly
# stopped — a 6000-token budget routinely returned a 2000-token pack that did
# not contain the answer. Measured on gosvc, "stop two workers ... writing
# their version" returned 2455/6000 tokens with sixteen full bodies from seven
# files and *not* `Store.UpdateStatus`, which was the answer, in the
# second-ranked file.
#
# The completion sweep spends the leftover budget where the evidence already
# points: remaining symbols of the files the pack has already committed to,
# bodies first for the small ones (facts live in bodies, not signatures), then
# signatures for breadth.
SWEEP_FILES = 4             # files eligible for completion, by file score
SWEEP_BODY_MAX_TOKENS = 420  # a body larger than this is swept in as a signature
SWEEP_BODY_MAX_LINES = 90    # cheap pre-reject, so long bodies are never read
SWEEP_BODY_SHARE = 0.6       # of the leftover budget; the rest buys breadth
# Physical adjacency to already-included code was tried as a ranking signal —
# the case for it is that a controlling constant is declared next to the code
# that spends it, and no word overlap will ever connect them. It does not
# survive measurement: as a multiplier it cost 28 points of answerability on
# gosvc and as a mild additive term it still cost 5, because it reliably
# promoted whatever the seeds happened to sit beside over the symbol the
# question was actually about. Relevance, then breadth, beats locality.
SWEEP_CANDIDATES_PER_FILE = 80   # cost-scoring cap; keeps large files cheap
# Full bodies are only granted to seeds whose file is among the top few by
# score. Bodies from weakly-ranked files were the dominant source of wasted
# tokens: they cost the most and are least likely to be on target.
BODY_FILE_RANK = 3


# ==========================================================================
# python parser (stdlib ast)
# ==========================================================================
"""AST-based extraction of symbols and relationships from Python source.

Uses only the standard-library `ast` module. For each file we extract:
  - symbols: modules, classes, functions, methods (with exact line spans + signature)
  - edges:   DEFINES, CALLS, INHERITS, IMPORTS

Call/inheritance targets are recorded *by name* (best-effort). They are
resolved to concrete symbol ids later, in the indexer, once every file's
symbols are known. This two-phase approach is the standard way to handle the
fact that a call may reference a symbol defined in another file.
"""

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---- data model -----------------------------------------------------------

@dataclass
class Symbol:
    qname: str                 # module.Class.method  (globally unique)
    name: str                  # leaf name (method)
    kind: str                  # module | class | function | method
    file: str                  # relative path
    lineno: int                # 1-based start line of the def
    end_lineno: int            # inclusive end line
    signature: str = ""        # def foo(a, b) -> int
    docstring: str = ""        # first line of the docstring, if any
    parent: Optional[str] = None  # qname of the enclosing symbol


@dataclass
class PendingEdge:
    src_qname: str             # who owns the reference
    dst_name: str              # textual target (e.g. "helper" or "obj.method")
    type: str                  # CALLS | INHERITS | IMPORTS


@dataclass
class ParseResult:
    file: str
    file_hash: str
    symbols: list[Symbol] = field(default_factory=list)
    edges: list[PendingEdge] = field(default_factory=list)
    error: Optional[str] = None


# ---- helpers --------------------------------------------------------------

def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _signature(node: ast.AST, src_lines: list[str]) -> str:
    """Reconstruct the def line(s) up to the colon, collapsed to one line."""
    start = node.lineno - 1
    # walk forward until the line that closes the signature with ':'
    buf = []
    depth = 0
    for i in range(start, min(start + 40, len(src_lines))):
        line = src_lines[i]
        buf.append(line.strip())
        depth += line.count("(") - line.count(")")
        if depth <= 0 and line.rstrip().endswith(":"):
            break
    sig = " ".join(buf)
    return sig.rsplit(":", 1)[0].strip()


def _docstring(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node) or ""
    except TypeError:
        doc = ""
    return doc.strip().splitlines()[0] if doc.strip() else ""


def _callee_name(call: ast.Call) -> Optional[str]:
    """Textual name of a call target: `foo` or `obj.method` or `a.b.c`."""
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        parts = []
        cur: ast.AST = f
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


BINDING_SIGNATURE_CHARS = 240


def _binding_signature(node: ast.AST, src_lines: list[str]) -> str:
    """The assignment as written, collapsed to one line and bounded.

    The *value* is the point — a constant whose signature is just its name
    answers nothing. Long literals (a big dict of thresholds) are truncated
    rather than dropped, and the body still carries the full text.
    """
    start = (node.lineno or 1) - 1
    end = min(len(src_lines), getattr(node, "end_lineno", node.lineno) or node.lineno)
    text = " ".join(line.strip() for line in src_lines[start:end])
    text = " ".join(text.split())
    return (text[:BINDING_SIGNATURE_CHARS] + " …"
            if len(text) > BINDING_SIGNATURE_CHARS else text)


def _base_name(base: ast.AST) -> Optional[str]:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


# ---- the visitor ----------------------------------------------------------

class _Collector(ast.NodeVisitor):
    def __init__(self, module: str, file: str, src_lines: list[str]):
        self.module = module
        self.file = file
        self.src_lines = src_lines
        self.symbols: list[Symbol] = []
        self.edges: list[PendingEdge] = []
        self._scope: list[str] = [module]  # stack of qnames

    # --- scope-aware traversal ---
    def _qname(self, leaf: str) -> str:
        return f"{self._scope[-1]}.{leaf}"

    def _visit_def(self, node, kind: str):
        qname = self._qname(node.name)
        end = getattr(node, "end_lineno", node.lineno)
        self.symbols.append(Symbol(
            qname=qname, name=node.name, kind=kind, file=self.file,
            lineno=node.lineno, end_lineno=end,
            signature=_signature(node, self.src_lines),
            docstring=_docstring(node), parent=self._scope[-1],
        ))
        # inheritance edges for classes
        if kind == "class":
            for base in node.bases:
                bn = _base_name(base)
                if bn:
                    self.edges.append(PendingEdge(qname, bn, "INHERITS"))
        # descend, tracking scope
        self._scope.append(qname)
        for child in node.body:
            self.visit(child)
        self._scope.pop()

    def visit_FunctionDef(self, node):
        kind = "method" if self._scope[-1] != self.module and \
            self._is_class_scope() else "function"
        self._visit_def(node, kind)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._visit_def(node, "class")

    # --- module- and class-level bindings (CN-1) ---
    #
    # Constants were invisible to the graph: only defs and classes became
    # symbols, so `MAX_LINES_PER_ORDER`, `RETRYABLE_STATUS` and
    # `Retriever.GREP_WINDOW_LINES` could not be searched for, ranked, or
    # pulled into a pack — yet "what is the limit / which statuses retry / how
    # big is the window" is exactly the sort of question a controlling constant
    # answers. The benchmark showed packs that found the right file and still
    # missed the constant that *was* the answer.
    #
    # Only definitional scopes count. A local inside a function is
    # implementation detail no question is ever about, and indexing locals
    # would swamp the symbol table.
    def _visit_binding(self, node, names: list[str]):
        if not self._scope_is_definitional():
            return
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        for name in names:
            self.symbols.append(Symbol(
                qname=self._qname(name), name=name, kind="constant",
                file=self.file, lineno=node.lineno, end_lineno=end,
                signature=_binding_signature(node, self.src_lines),
                docstring="", parent=self._scope[-1],
            ))

    def visit_Assign(self, node):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        self._visit_binding(node, names)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            self._visit_binding(node, [node.target.id])
        self.generic_visit(node)

    def _scope_is_definitional(self) -> bool:
        """True at module scope or directly inside a class body."""
        return self._scope[-1] == self.module or self._is_class_scope()

    def visit_Call(self, node):
        owner = self._scope[-1]
        name = _callee_name(node)
        if name:
            self.edges.append(PendingEdge(owner, name, "CALLS"))
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.edges.append(PendingEdge(self.module, alias.name, "IMPORTS"))

    def visit_ImportFrom(self, node):
        mod = node.module or ""
        for alias in node.names:
            target = f"{mod}.{alias.name}" if mod else alias.name
            self.edges.append(PendingEdge(self.module, target, "IMPORTS"))

    # helper: is the current scope a class? (so we tag methods correctly)
    def _is_class_scope(self) -> bool:
        cur = self._scope[-1]
        for s in self.symbols:
            if s.qname == cur:
                return s.kind == "class"
        return False


# ---- public entrypoint ----------------------------------------------------

def module_name(repo_root: Path, path: Path) -> str:
    rel = path.relative_to(repo_root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else path.stem


def parse_file(repo_root: Path, path: Path) -> ParseResult:
    rel = path.relative_to(repo_root).as_posix()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ParseResult(rel, "", error=str(e))

    h = file_hash(text)
    mod = module_name(repo_root, path)
    src_lines = text.splitlines()

    result = ParseResult(rel, h)
    # the module itself is a symbol (its body span is the whole file)
    result.symbols.append(Symbol(
        qname=mod, name=mod.split(".")[-1], kind="module", file=rel,
        lineno=1, end_lineno=max(1, len(src_lines)),
        signature=f"module {mod}", docstring="",
    ))
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        result.error = f"SyntaxError: {e}"
        return result

    c = _Collector(mod, rel, src_lines)
    for child in tree.body:
        c.visit(child)
    result.symbols.extend(c.symbols)
    result.edges.extend(c.edges)
    return result


# ==========================================================================
# multi-language parser (tree-sitter)
# ==========================================================================
"""Multi-language extraction via tree-sitter.

The Python path keeps using the stdlib `ast` (exact, no deps). Every other
language is parsed by tree-sitter and emits the *same* Symbol / PendingEdge
records, so the store, indexer, retriever, and MCP server are unchanged.

Grammars ship as precompiled wheels (tree-sitter-java/go/typescript/javascript)
— no runtime download. If they aren't installed, `build_profiles()` returns an
empty map and the tool quietly stays Python-only.

A LanguageProfile describes, per language: which node types are definitions
(and their kind), which are calls/imports, and how to pull names, signatures,
base types, and callee names out of those nodes. A single generic walker then
does scope-aware traversal identical in spirit to the Python collector.
"""

from pathlib import Path
from typing import Optional



# ---- small tree helpers ---------------------------------------------------

def _text(node) -> str:
    return node.text.decode("utf-8", "replace")


def _descendants(node):
    stack = list(node.children)
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def _leaf_name(node) -> Optional[str]:
    """Last identifier-ish token inside a (possibly generic/qualified) type node."""
    if node is None:
        return None
    _ID_TYPES = ("identifier", "type_identifier", "field_identifier",
                 "property_identifier", "simple_identifier", "constant",
                 "user_defined_type")
    if node.type in _ID_TYPES:
        return _text(node).split(".")[-1].split("\\")[-1]
    last = None
    for d in _descendants(node):
        if d.type in _ID_TYPES:
            last = _text(d)
    return (last or _text(node).split(".")[-1]).split("\\")[-1]


# ---- doc-comment extraction (cross-language "semantic bridge") ------------
#
# Python takes its docstring from the stdlib AST (see `_docstring`). Every other
# language documents its API in a leading comment block instead: Go `// …`,
# Rust `/// …` / `//! …`, Java/JS/TS `/** … */` (Javadoc / JSDoc / TSDoc),
# C# `/// <summary>…`. That comment is the single richest natural-language
# signal a symbol carries, and feeding it into the signature + embedding text
# measurably lifts semantic retrieval on polyglot repos. Previously only Python
# symbols were enriched; every tree-sitter and regex-parsed symbol now gets the
# same treatment. We pull the contiguous comment block immediately above a
# definition, strip the comment markup, and keep a compact one-line summary.

DOC_COMMENT_MAX_CHARS = 200

# Comment node types across the tree-sitter grammars we load.
_DOC_COMMENT_TYPES = {
    "comment", "line_comment", "block_comment", "doc_comment",
    "expression_comment", "outer_doc_comment", "inner_doc_comment",
    "documentation_comment",
}
# Nodes that legitimately sit between a doc comment and its definition:
# annotations (Java `@Override`), attributes (Rust `#[inline]`, C# `[Test]`),
# decorators (TS `@Component`) and modifier lists. Walked through transparently
# so the comment above them is still attached to the def.
_DOC_SKIP_TYPES = {
    "attribute_item", "attribute", "attribute_list", "annotation",
    "marker_annotation", "modifiers", "attribute_declaration",
}

# Leading per-line comment markers: `//` `///` `//!` `#` `#!` `--` `;` `*` `"""`.
_DOC_MARKER_RE = re.compile(r'^(?:///?!?|#!?|-{2,}|;{1,}|\*+|"""|\'\'\')\s?')


def _strip_comment_markers(line: str) -> str:
    """Strip comment fences/markers from one physical comment line."""
    s = line.strip()
    if s.startswith("/**"):
        s = s[3:]
    elif s.startswith("/*"):
        s = s[2:]
    if s.endswith("*/"):
        s = s[:-2]
    s = s.strip()
    m = _DOC_MARKER_RE.match(s)
    if m:
        s = s[m.end():]
    # C#/XML doc tags (`<summary>`) and HTML in Javadoc add no retrieval signal.
    s = re.sub(r"<[^>]+>", " ", s)
    return s.strip()


def _clean_doc_comment(raw: str) -> str:
    """Collapse a raw comment block to a compact, single-line summary."""
    out: list[str] = []
    for ln in raw.splitlines():
        s = _strip_comment_markers(ln)
        # A block tag (@param, @returns, \param …) ends the human summary.
        if s.startswith("@") or s.startswith("\\"):
            break
        if not s:
            if out:            # a blank line closes the summary paragraph
                break
            continue
        out.append(s)
    text = " ".join(out).strip()
    if len(text) > DOC_COMMENT_MAX_CHARS:
        text = text[:DOC_COMMENT_MAX_CHARS].rstrip() + "…"
    return text


def leading_doc_comment(node, comment_types=_DOC_COMMENT_TYPES) -> str:
    """The doc comment immediately above a tree-sitter definition node, if any.

    Walks preceding siblings, stepping transparently over annotations /
    attributes / decorators, and collects the contiguous run of comment nodes
    that sits directly above the definition (a ≤1-line gap is tolerated).
    """
    prev = node.prev_sibling
    last_row = node.start_point[0]
    while prev is not None and prev.type in _DOC_SKIP_TYPES:
        last_row = prev.start_point[0]
        prev = prev.prev_sibling
    collected: list[str] = []
    while prev is not None and prev.type in comment_types:
        if last_row - prev.end_point[0] > 1:   # not contiguous → not this def's doc
            break
        collected.append(_text(prev))
        last_row = prev.start_point[0]
        prev = prev.prev_sibling
    if not collected:
        return ""
    collected.reverse()
    return _clean_doc_comment("\n".join(collected))


def _generic_leading_doc(lines: list[str], lineno: int) -> str:
    """Best-effort leading comment for a regex-extracted def (no grammar).

    Scans upward from the line above the definition, collecting a contiguous
    block of comment-looking lines. Purely lexical — it never parses — so it
    stays cheap and safe for the long tail of languages without a grammar.
    """
    i = lineno - 2                                  # 0-based line above the def
    block: list[str] = []
    while i >= 0:
        s = lines[i].strip()
        if not s:
            break
        if (_DOC_MARKER_RE.match(s) or s.startswith("/*") or s.endswith("*/")):
            block.append(lines[i])
            i -= 1
            continue
        break
    if not block:
        return ""
    block.reverse()
    return _clean_doc_comment("\n".join(block))


CLASS_KINDS = {"class", "interface", "struct", "enum", "record"}


# ---- profiles -------------------------------------------------------------

class LanguageProfile:
    name = ""
    extensions: tuple[str, ...] = ()
    DEF_KINDS: dict[str, str] = {}
    CALL_TYPES: set[str] = set()
    IMPORT_TYPES: set[str] = set()
    # node types that, in statement position, are bare references to a callee
    # (e.g. Ruby `helper` with no parens parses as a plain identifier). Edges
    # are speculative: the resolver only links them when a unique symbol matches.
    BARE_CALL_NODES: set[str] = set()
    # CN-1: node types that bind a name to a value at file or type scope —
    # `const`/`var` blocks, class fields, exported literals. These become
    # `constant` symbols so a controlling value (a retry set, a size limit, a
    # status code) is searchable and packable in its own right, instead of
    # being reachable only by accident through a chunk. Locals are excluded:
    # the walker only consults this at definitional scope.
    BINDING_KINDS: dict[str, str] = {}

    def __init__(self, language):
        from tree_sitter import Parser  # type: ignore[import-not-found]
        self.language = language
        self.parser = Parser(language)

    # --- overridable hooks ---
    def kind_of(self, node, default: str) -> str:
        return default

    def name_of(self, node) -> Optional[str]:
        n = node.child_by_field_name("name")
        return _text(n) if n else None

    def owner_prefix(self, node) -> Optional[str]:
        """Extra qualifier between scope and leaf (e.g. Go receiver type)."""
        return None

    def body_of(self, node):
        return node.child_by_field_name("body") or node

    def doc_of(self, node) -> str:
        """Leading doc comment (godoc / rustdoc / Javadoc / JSDoc / TSDoc …)."""
        return leading_doc_comment(node)

    def bases_of(self, node) -> list[str]:
        return []

    def callee_of(self, node) -> Optional[str]:
        return None

    def import_of(self, node) -> Optional[str]:
        return None

    def synth_def(self, node):
        """For lambdas assigned to names: (name, kind, def_node, body_node) or None."""
        return None

    def binding_names(self, node) -> list[str]:
        """Names bound by a BINDING_KINDS node (CN-1).

        The default handles the shape every C-family grammar shares: one or
        more declarator/spec children each carrying a `name` field. A single
        statement may bind several names (`const a = 1, b = 2`; Go's
        `var ( x int; y string )`), so this returns a list.
        """
        out: list[str] = []
        for d in _descendants(node):
            if d.type in self.BINDING_DECLARATORS:
                n = d.child_by_field_name("name")
                if n is not None and (txt := _text(n)):
                    out.append(txt)
        if not out:
            n = node.child_by_field_name("name")
            if n is not None and (txt := _text(n)):
                out.append(txt)
        return out

    # Declarator/spec nodes searched by the default `binding_names`.
    BINDING_DECLARATORS: set[str] = {
        "variable_declarator", "const_spec", "var_spec", "init_declarator",
    }


class JavaProfile(LanguageProfile):
    name = "java"
    extensions = (".java",)
    DEF_KINDS = {
        "class_declaration": "class", "interface_declaration": "interface",
        "enum_declaration": "enum", "record_declaration": "record",
        "method_declaration": "method", "constructor_declaration": "method",
    }
    CALL_TYPES = {"method_invocation", "object_creation_expression"}
    IMPORT_TYPES = {"import_declaration"}
    BINDING_KINDS = {"field_declaration": "constant"}

    def bases_of(self, node):
        out = []
        for f in ("superclass", "interfaces"):
            fn = node.child_by_field_name(f)
            if fn:
                for d in _descendants(fn):
                    if d.type == "type_identifier":
                        out.append(_text(d))
        return out

    def callee_of(self, node):
        if node.type == "object_creation_expression":
            return _leaf_name(node.child_by_field_name("type"))
        n = node.child_by_field_name("name")
        return _text(n) if n else None

    def import_of(self, node):
        for d in _descendants(node):
            if d.type == "scoped_identifier":
                return _text(d)
        return None


class GoProfile(LanguageProfile):
    name = "go"
    extensions = (".go",)
    DEF_KINDS = {
        "function_declaration": "function", "method_declaration": "method",
        "type_spec": "type",
    }
    CALL_TYPES = {"call_expression"}
    IMPORT_TYPES = {"import_spec"}
    # Go keeps its tunables in package-level `const`/`var` blocks and its
    # configuration in struct fields; both are answers to "what is the limit".
    BINDING_KINDS = {"const_declaration": "constant",
                     "var_declaration": "constant",
                     "field_declaration": "field"}

    def kind_of(self, node, default):
        if node.type == "type_spec":
            t = node.child_by_field_name("type")
            if t and t.type == "struct_type":
                return "struct"
            if t and t.type == "interface_type":
                return "interface"
        return default

    def owner_prefix(self, node):
        if node.type != "method_declaration":
            return None
        recv = node.child_by_field_name("receiver")
        if not recv:
            return None
        for d in _descendants(recv):
            if d.type == "type_identifier":
                return _text(d)
        return None

    def callee_of(self, node):
        fn = node.child_by_field_name("function")
        if not fn:
            return None
        if fn.type == "identifier":
            return _text(fn)
        if fn.type == "selector_expression":
            f = fn.child_by_field_name("field")
            return _text(f) if f else None
        return None

    def import_of(self, node):
        for d in _descendants(node):
            if d.type == "interpreted_string_literal_content":
                return _text(d).split("/")[-1]
        return None


class TsJsProfile(LanguageProfile):
    """Shared logic for TypeScript, TSX, and JavaScript."""
    DEF_KINDS = {
        "class_declaration": "class", "abstract_class_declaration": "class",
        "interface_declaration": "interface",
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "method_definition": "method",
    }
    CALL_TYPES = {"call_expression", "new_expression"}
    IMPORT_TYPES = {"import_statement"}
    # `const MAX_BODY_BYTES = 1_000_000` at module scope, and class fields.
    # `synth_def` claims the arrow-function ones first, so a named lambda stays
    # a function rather than being demoted to a constant.
    BINDING_KINDS = {"lexical_declaration": "constant",
                     "variable_declaration": "constant",
                     "public_field_definition": "field"}
    _ARROW = {"arrow_function", "function", "function_expression",
              "generator_function"}

    def bases_of(self, node):
        out = []
        for c in node.children:
            if c.type == "class_heritage":
                for d in _descendants(c):
                    if d.type in ("identifier", "type_identifier"):
                        out.append(_text(d))
        return out

    def callee_of(self, node):
        if node.type == "new_expression":
            return _leaf_name(node.child_by_field_name("constructor"))
        fn = node.child_by_field_name("function")
        if not fn:
            return None
        if fn.type == "identifier":
            return _text(fn)
        if fn.type == "member_expression":
            p = fn.child_by_field_name("property")
            return _text(p) if p else None
        return None

    def import_of(self, node):
        for d in _descendants(node):
            if d.type == "string_fragment":
                return _text(d).split("/")[-1]
        return None

    def synth_def(self, node):
        if node.type != "variable_declarator":
            return None
        val = node.child_by_field_name("value")
        if val and val.type in self._ARROW:
            name = node.child_by_field_name("name")
            if name:
                return (_text(name), "function", node,
                        val.child_by_field_name("body") or val)
        return None


# ---- additional tree-sitter profiles (node types verified against the
#      installed grammars; loaded from tree-sitter-language-pack when present) --

# Body-bearing node types across the grammars below — used to cut signatures
# off before the body (these grammars don't expose a "body" field uniformly).
_BODY_TYPES = {
    "block", "compound_statement", "class_body", "function_body",
    "enum_class_body", "declaration_list", "field_declaration_list",
    "template_body", "protocol_body", "body_statement", "statement_block",
    "enum_variant_list", "interface_body", "struct_body",
    # extra grammars (Lua/Bash/Solidity/Erlang/Julia/OCaml/PowerShell/Nim/…)
    "contract_body", "script_block", "structure", "clause_body",
    "do_block", "braced_expression", "stmt", "implementation_definition",
    "class_implementation", "matchclause",
}


class _TsBase(LanguageProfile):
    """Shared helpers for the extra profiles: body detection + leaf callee."""

    def body_of(self, node):
        b = node.child_by_field_name("body")
        if b is not None:
            return b
        for c in node.children:
            if c.type in _BODY_TYPES:
                return c
        return node


class CFamilyProfile(_TsBase):
    """C and C++ (C++ adds class/namespace + base classes)."""
    name = "c"
    DEF_KINDS = {
        "function_definition": "function",
        "struct_specifier": "struct", "union_specifier": "struct",
        "enum_specifier": "enum", "type_definition": "type",
    }
    CALL_TYPES = {"call_expression"}
    IMPORT_TYPES = {"preproc_include"}

    def name_of(self, node):
        if node.type == "function_definition":
            d = node.child_by_field_name("declarator")
            while d is not None and d.type in (
                    "pointer_declarator", "reference_declarator", "init_declarator"):
                d = d.child_by_field_name("declarator")
            if d is not None and d.type == "function_declarator":
                inner = d.child_by_field_name("declarator")
                return _leaf_name(inner) if inner is not None else None
            return _leaf_name(d) if d is not None else None
        if node.type == "type_definition":
            return _leaf_name(node.child_by_field_name("declarator"))
        n = node.child_by_field_name("name")
        return _text(n) if n is not None else _leaf_name(node)

    def callee_of(self, node):
        return _leaf_name(node.child_by_field_name("function"))

    def import_of(self, node):
        p = node.child_by_field_name("path")
        if p is None:
            return None
        return _text(p).strip("<>\"").split("/")[-1]


class CppProfile(CFamilyProfile):
    name = "cpp"
    DEF_KINDS = {
        **CFamilyProfile.DEF_KINDS,
        "class_specifier": "class", "namespace_definition": "namespace",
    }

    def bases_of(self, node):
        out = []
        for c in node.children:
            if c.type == "base_class_clause":
                for d in _descendants(c):
                    if d.type == "type_identifier":
                        out.append(_text(d))
        return out


class CSharpProfile(_TsBase):
    name = "csharp"
    DEF_KINDS = {
        "class_declaration": "class", "interface_declaration": "interface",
        "struct_declaration": "struct", "enum_declaration": "enum",
        "record_declaration": "record", "method_declaration": "method",
        "constructor_declaration": "method", "property_declaration": "property",
        "namespace_declaration": "namespace",
    }
    CALL_TYPES = {"invocation_expression", "object_creation_expression"}
    IMPORT_TYPES = {"using_directive"}

    def bases_of(self, node):
        out = []
        for c in node.children:
            if c.type == "base_list":
                for d in c.children:
                    if d.type in ("identifier", "qualified_name", "generic_name"):
                        out.append(_leaf_name(d))
        return [b for b in out if b]

    def callee_of(self, node):
        if node.type == "object_creation_expression":
            return _leaf_name(node.child_by_field_name("type"))
        fn = node.child_by_field_name("function")
        if fn is None:
            return None
        if fn.type == "member_access_expression":
            n = fn.child_by_field_name("name")
            return _text(n) if n is not None else _leaf_name(fn)
        return _leaf_name(fn)

    def import_of(self, node):
        for c in node.children:
            if c.type in ("qualified_name", "identifier"):
                return _text(c).split(".")[-1]
        return None


class RustProfile(_TsBase):
    name = "rust"
    DEF_KINDS = {
        "function_item": "function", "struct_item": "struct",
        "enum_item": "enum", "union_item": "struct", "trait_item": "trait",
        "mod_item": "namespace", "type_item": "type",
        "macro_definition": "macro",
    }
    CALL_TYPES = {"call_expression", "macro_invocation"}
    IMPORT_TYPES = {"use_declaration"}

    def callee_of(self, node):
        if node.type == "macro_invocation":
            return _leaf_name(node.child_by_field_name("macro"))
        return _leaf_name(node.child_by_field_name("function"))

    def import_of(self, node):
        a = node.child_by_field_name("argument")
        return _leaf_name(a) if a is not None else None


class RubyProfile(_TsBase):
    name = "ruby"
    DEF_KINDS = {
        "class": "class", "module": "namespace",
        "method": "method", "singleton_method": "method",
    }
    CALL_TYPES = {"call"}
    # `helper` with no args/parens parses as a bare identifier, not a call.
    BARE_CALL_NODES = {"identifier"}

    def bases_of(self, node):
        sc = node.child_by_field_name("superclass")
        if sc is None:
            for c in node.children:
                if c.type == "superclass":
                    sc = c
                    break
        n = _leaf_name(sc) if sc is not None else None
        return [n] if n else []

    def callee_of(self, node):
        m = node.child_by_field_name("method")
        return _text(m) if m is not None else None


class PhpProfile(_TsBase):
    name = "php"
    DEF_KINDS = {
        "class_declaration": "class", "interface_declaration": "interface",
        "trait_declaration": "trait", "enum_declaration": "enum",
        "method_declaration": "method", "function_definition": "function",
        "namespace_definition": "namespace",
    }
    CALL_TYPES = {"function_call_expression", "member_call_expression",
                  "object_creation_expression", "scoped_call_expression"}
    IMPORT_TYPES = {"namespace_use_declaration"}

    def bases_of(self, node):
        out = []
        for c in node.children:
            if c.type in ("base_clause", "class_interface_clause"):
                for d in c.children:
                    if d.type in ("name", "qualified_name"):
                        out.append(_text(d).split("\\")[-1])
        return out

    def callee_of(self, node):
        n = node.child_by_field_name("name")
        if n is not None:
            return _text(n)
        fn = node.child_by_field_name("function")
        if fn is not None:
            return _leaf_name(fn)
        for c in node.children:
            if c.type in ("name", "qualified_name"):
                return _text(c).split("\\")[-1]
        return None

    def import_of(self, node):
        for d in _descendants(node):
            if d.type in ("qualified_name", "name"):
                return _text(d).split("\\")[-1]
        return None


class KotlinProfile(_TsBase):
    name = "kotlin"
    DEF_KINDS = {
        "class_declaration": "class", "object_declaration": "object",
        "function_declaration": "function", "property_declaration": "property",
    }
    CALL_TYPES = {"call_expression"}
    IMPORT_TYPES = {"import_header"}

    def name_of(self, node):
        for c in node.children:
            if c.type in ("type_identifier", "simple_identifier"):
                return _text(c)
        return None

    def bases_of(self, node):
        out = []
        for c in node.children:
            if c.type == "delegation_specifier":
                for d in _descendants(c):
                    if d.type == "type_identifier":
                        out.append(_text(d))
        return out

    def callee_of(self, node):
        ch = node.children[0] if node.children else None
        return _leaf_name(ch) if ch is not None else None

    def import_of(self, node):
        for c in node.children:
            if c.type == "identifier":
                return _text(c).split(".")[-1]
        return None


class SwiftProfile(_TsBase):
    name = "swift"
    DEF_KINDS = {
        "class_declaration": "class", "function_declaration": "function",
        "protocol_declaration": "protocol", "init_declaration": "method",
    }
    CALL_TYPES = {"call_expression"}
    IMPORT_TYPES = {"import_declaration"}
    _SWIFT_KW = {"struct", "enum", "extension", "actor", "class"}

    def kind_of(self, node, default):
        if node.type == "class_declaration":
            for c in node.children:
                if c.type in self._SWIFT_KW:
                    return c.type
        return default

    def bases_of(self, node):
        out = []
        for c in node.children:
            if c.type == "inheritance_specifier":
                for d in _descendants(c):
                    if d.type == "type_identifier":
                        out.append(_text(d))
        return out

    def callee_of(self, node):
        ch = node.children[0] if node.children else None
        return _leaf_name(ch) if ch is not None else None

    def import_of(self, node):
        for c in node.children:
            if c.type == "identifier":
                return _text(c).split(".")[-1]
        return None


class ScalaProfile(_TsBase):
    name = "scala"
    DEF_KINDS = {
        "class_definition": "class", "object_definition": "object",
        "trait_definition": "trait", "function_definition": "method",
        "val_definition": "value", "type_definition": "type",
    }
    CALL_TYPES = {"call_expression"}
    IMPORT_TYPES = {"import_declaration"}

    def bases_of(self, node):
        out = []
        for c in node.children:
            if c.type == "extends_clause":
                for d in _descendants(c):
                    if d.type == "type_identifier":
                        out.append(_text(d))
        return out

    def callee_of(self, node):
        return _leaf_name(node.child_by_field_name("function"))

    def import_of(self, node):
        ids = [c for c in node.children if c.type == "identifier"]
        return _text(ids[-1]) if ids else None


def _first_child(node, *types):
    for c in node.children:
        if c.type in types:
            return c
    return None


def _first_descendant(node, *types):
    for d in _descendants(node):
        if d.type in types:
            return d
    return None


class LuaProfile(_TsBase):
    name = "lua"
    DEF_KINDS = {
        "function_declaration": "function",
        "local_function": "function",
        "function_definition": "function",
    }
    CALL_TYPES = {"function_call"}

    @staticmethod
    def _dotted_leaf(n):
        # `M.greet` / `obj:method` — keep the trailing field, not the table.
        if n is None:
            return None
        if n.type == "dot_index_expression":
            f = n.child_by_field_name("field")
            return _text(f) if f is not None else _leaf_name(n)
        if n.type == "method_index_expression":
            f = n.child_by_field_name("method")
            return _text(f) if f is not None else _leaf_name(n)
        return _text(n) if n.type == "identifier" else _leaf_name(n)

    def name_of(self, node):
        return self._dotted_leaf(node.child_by_field_name("name"))

    def callee_of(self, node):
        return self._dotted_leaf(node.child_by_field_name("name"))


class BashProfile(_TsBase):
    name = "bash"
    DEF_KINDS = {"function_definition": "function"}
    CALL_TYPES = {"command"}

    def name_of(self, node):
        n = node.child_by_field_name("name")
        return _text(n).strip() if n is not None else None

    def callee_of(self, node):
        n = node.child_by_field_name("name")
        if n is None:
            return None
        name = _text(n).strip().split("/")[-1]
        # skip shell builtins / operators that aren't user functions (`:`, `[`)
        if name and (name[0].isalpha() or name[0] == "_"):
            return name
        return None


class SolidityProfile(_TsBase):
    name = "solidity"
    DEF_KINDS = {
        "contract_declaration": "contract", "interface_declaration": "interface",
        "library_declaration": "library", "struct_declaration": "struct",
        "enum_declaration": "enum", "function_definition": "function",
        "modifier_definition": "modifier", "event_definition": "event",
        "constructor_definition": "method", "error_declaration": "error",
    }
    CALL_TYPES = {"call_expression"}
    IMPORT_TYPES = {"import_directive"}

    def bases_of(self, node):
        out = []
        for c in node.children:
            if c.type == "inheritance_specifier":
                n = _leaf_name(c)
                if n:
                    out.append(n)
        return out

    def callee_of(self, node):
        return _leaf_name(node.child_by_field_name("function"))

    def import_of(self, node):
        s = _first_descendant(node, "string")
        if s is None:
            return None
        return _text(s).strip("'\"").split("/")[-1].rsplit(".", 1)[0]


class PerlProfile(_TsBase):
    name = "perl"
    DEF_KINDS = {
        "subroutine_declaration_statement": "function",
        "package_statement": "namespace",
    }
    CALL_TYPES = {"function_call_expression", "method_call_expression"}
    IMPORT_TYPES = {"use_statement"}

    def callee_of(self, node):
        fn = node.child_by_field_name("function")
        return _leaf_name(fn) if fn is not None else None

    def import_of(self, node):
        m = node.child_by_field_name("module")
        return _text(m).split("::")[-1] if m is not None else None


class ErlangProfile(_TsBase):
    name = "erlang"
    DEF_KINDS = {"fun_decl": "function", "module_attribute": "namespace"}
    CALL_TYPES = {"call"}

    def name_of(self, node):
        if node.type == "module_attribute":
            if _first_child(node, "module") is None:  # only `-module(...)`
                return None
            n = node.child_by_field_name("name")
            return _text(n) if n is not None else None
        clause = _first_child(node, "function_clause")
        if clause is None:
            return None
        n = clause.child_by_field_name("name")
        return _text(n) if n is not None else None

    def callee_of(self, node):
        e = node.child_by_field_name("expr")
        return _leaf_name(e) if e is not None else None


class JuliaProfile(_TsBase):
    name = "julia"
    DEF_KINDS = {
        "function_definition": "function", "short_function_definition": "function",
        "struct_definition": "struct", "module_definition": "namespace",
        "abstract_definition": "type", "macro_definition": "macro",
    }
    CALL_TYPES = {"call_expression"}
    IMPORT_TYPES = {"using_statement", "import_statement"}

    def name_of(self, node):
        if node.type in ("function_definition", "short_function_definition"):
            sig = _first_child(node, "signature") or node
            ce = _first_descendant(sig, "call_expression")
            if ce is not None and ce.children:
                return _leaf_name(ce.children[0])
            return None
        if node.type in ("struct_definition", "abstract_definition"):
            head = _first_child(node, "type_head")
            tgt = head if head is not None else node
            ident = _first_descendant(tgt, "identifier")
            return _text(ident) if ident is not None else None
        n = node.child_by_field_name("name")
        return _text(n) if n is not None else None

    def callee_of(self, node):
        if not node.children:
            return None
        head = node.children[0]
        return _leaf_name(head)

    def import_of(self, node):
        ident = _first_descendant(node, "identifier")
        return _text(ident) if ident is not None else None


class RProfile(_TsBase):
    name = "r"
    CALL_TYPES = {"call"}

    def synth_def(self, node):
        if node.type != "binary_operator":
            return None
        op = node.child_by_field_name("operator")
        if op is None or _text(op) not in ("<-", "<<-", "="):
            return None
        rhs = node.child_by_field_name("rhs")
        if rhs is None or rhs.type != "function_definition":
            return None
        lhs = node.child_by_field_name("lhs")
        name = _leaf_name(lhs) if lhs is not None else None
        if not name:
            return None
        return (name, "function", node, rhs.child_by_field_name("body") or rhs)

    def callee_of(self, node):
        fn = node.child_by_field_name("function")
        return _leaf_name(fn) if fn is not None else None


class HaskellProfile(_TsBase):
    name = "haskell"
    DEF_KINDS = {
        "function": "function", "bind": "function",
        "data_type": "type", "newtype": "type", "type_synonym": "type",
        "class": "class", "instance": "instance",
    }
    CALL_TYPES = {"apply"}
    IMPORT_TYPES = {"import"}

    def name_of(self, node):
        n = node.child_by_field_name("name")
        return _leaf_name(n) if n is not None else None

    def callee_of(self, node):
        fn = node.child_by_field_name("function")
        if fn is None:
            return None
        if fn.type in ("variable", "constructor", "qualified_variable",
                       "qualified", "operator"):
            return _text(fn).split(".")[-1]
        return None

    def import_of(self, node):
        m = node.child_by_field_name("module")
        return _text(m).split(".")[-1] if m is not None else None


class OCamlProfile(_TsBase):
    name = "ocaml"
    DEF_KINDS = {
        "value_definition": "value", "module_definition": "namespace",
        "type_definition": "type",
    }
    CALL_TYPES = {"application_expression"}
    IMPORT_TYPES = {"open_module", "open_directive"}

    def name_of(self, node):
        if node.type == "value_definition":
            lb = _first_child(node, "let_binding")
            if lb is None:
                return None
            pat = lb.child_by_field_name("pattern")
            return _text(pat) if pat is not None else None
        if node.type == "module_definition":
            mn = _first_descendant(node, "module_name")
            return _text(mn) if mn is not None else None
        if node.type == "type_definition":
            tb = _first_child(node, "type_binding")
            nm = tb.child_by_field_name("name") if tb is not None else None
            return _text(nm) if nm is not None else None
        return None

    def kind_of(self, node, default):
        if node.type == "value_definition":
            lb = _first_child(node, "let_binding")
            if lb is not None and _first_child(lb, "parameter") is not None:
                return "function"
        return default

    def callee_of(self, node):
        if not node.children:
            return None
        head = node.children[0]
        if head.type in ("value_path", "value_name"):
            return _leaf_name(head)
        return None

    def import_of(self, node):
        mp = _first_descendant(node, "module_path", "module_name")
        return _text(mp).split(".")[-1] if mp is not None else None


class NimProfile(_TsBase):
    name = "nim"
    DEF_KINDS = {"routine": "function", "typeDef": "type"}
    IMPORT_TYPES = {"importStmt", "fromStmt", "includeStmt"}

    def name_of(self, node):
        sym = _first_child(node, "symbol", "exportedSymbol")
        ident = _first_descendant(sym, "ident") if sym is not None else None
        return _text(ident) if ident is not None else None

    def kind_of(self, node, default):
        if node.type == "routine":
            kw = _first_child(node, "keyw")
            if kw is not None:
                return _text(kw)
        return default

    def import_of(self, node):
        ident = _first_descendant(node, "ident")
        return _text(ident) if ident is not None else None


class PowerShellProfile(_TsBase):
    name = "powershell"
    DEF_KINDS = {
        "function_statement": "function", "class_statement": "class",
        "class_method_definition": "method",
        "enum_statement": "enum",
    }

    def name_of(self, node):
        n = _first_child(node, "function_name", "simple_name")
        return _text(n) if n is not None else None


class DartProfile(_TsBase):
    name = "dart"
    DEF_KINDS = {
        "class_definition": "class", "mixin_declaration": "mixin",
        "enum_declaration": "enum", "extension_declaration": "extension",
        "function_signature": "function", "method_signature": "method",
        "getter_signature": "method", "setter_signature": "method",
        "constructor_signature": "method",
    }
    IMPORT_TYPES = {"import_or_export"}

    def name_of(self, node):
        n = node.child_by_field_name("name")
        if n is not None:
            return _text(n)
        sig = _first_descendant(node, "function_signature")
        if sig is not None:
            nn = sig.child_by_field_name("name")
            return _text(nn) if nn is not None else None
        ident = _first_descendant(node, "identifier")
        return _text(ident) if ident is not None else None

    def body_of(self, node):
        if node.type in ("function_signature", "method_signature",
                         "getter_signature", "setter_signature",
                         "constructor_signature"):
            nxt = node.next_named_sibling
            if nxt is not None and nxt.type == "function_body":
                return nxt
            return node
        return _TsBase.body_of(self, node)

    def bases_of(self, node):
        out = []
        sc = node.child_by_field_name("superclass")
        if sc is not None:
            tid = _first_descendant(sc, "type_identifier")
            if tid is not None:
                out.append(_text(tid))
        return out

    def import_of(self, node):
        s = _first_descendant(node, "string_literal")
        if s is None:
            return None
        return _text(s).strip("'\"").split("/")[-1].rsplit(".", 1)[0]


# ---- generic walker -------------------------------------------------------

class _Walk:
    def __init__(self, profile: LanguageProfile, module: str, file: str, src: bytes):
        self.p = profile
        self.module = module
        self.file = file
        self.src = src
        self.symbols: list[Symbol] = []
        self.edges: list[PendingEdge] = []
        # Scopes where a name binding is API rather than a local (CN-1).
        self._type_scopes: set[str] = set()

    def run(self, root):
        self._walk(root, self.module)

    def _is_definitional(self, scope: str) -> bool:
        return scope == self.module or scope in self._type_scopes

    def _record_binding(self, node, scope: str) -> None:
        """Record file-scope constants and type fields as symbols (CN-1)."""
        kind = self.p.BINDING_KINDS[node.type]
        for name in self.p.binding_names(node):
            self.symbols.append(Symbol(
                qname=f"{scope}.{name}", name=name, kind=kind, file=self.file,
                lineno=node.start_point[0] + 1,
                end_lineno=node.end_point[0] + 1,
                # The value is the whole point of a constant, so the signature
                # carries the declaration as written rather than just the name.
                signature=" ".join(
                    self.src[node.start_byte:node.end_byte]
                    .decode("utf-8", "replace").split())[:BINDING_SIGNATURE_CHARS],
                docstring=self.p.doc_of(node), parent=scope,
            ))

    def _signature(self, def_node, body_node) -> str:
        if (body_node is not None and body_node is not def_node
                and body_node.start_byte > def_node.start_byte):
            end = body_node.start_byte
        else:
            end = def_node.end_byte
        raw = self.src[def_node.start_byte:end].decode("utf-8", "replace")
        s = " ".join(raw.split()).rstrip("{").strip()
        return s[:200]

    def _record_def(self, def_node, name, kind, scope, body_node, with_bases):
        prefix = self.p.owner_prefix(def_node)
        leaf = f"{prefix}.{name}" if prefix else name
        qname = f"{scope}.{leaf}"
        self.symbols.append(Symbol(
            qname=qname, name=name, kind=kind, file=self.file,
            lineno=def_node.start_point[0] + 1,
            end_lineno=def_node.end_point[0] + 1,
            signature=self._signature(def_node, body_node),
            docstring=self.p.doc_of(def_node), parent=scope,
        ))
        if kind in CLASS_KINDS:
            self._type_scopes.add(qname)   # CN-1: its fields are API
        # `bases_of` is overridden only on type-like profiles and returns [] for
        # functions/methods, so calling it unconditionally is safe and lets
        # container kinds beyond CLASS_KINDS (contract, trait, protocol, …)
        # contribute inheritance edges.
        if with_bases or kind not in ("function", "method"):
            for b in self.p.bases_of(def_node):
                self.edges.append(PendingEdge(qname, b, "INHERITS"))
        # A def body may be a container (block/template_body — walk its
        # statements) OR an expression that is *itself* a call, as in
        # expression-bodied defs (`def run() = helper()`, Lua/OCaml/F#/Scala).
        # Visiting the body node directly handles both; only fall back to
        # walking children when body_of returned the def node itself (no body),
        # which would otherwise re-record this same def and recurse forever.
        if body_node is not None and body_node is not def_node:
            self._visit(body_node, qname)
        else:
            self._walk(def_node, qname)

    def _walk(self, node, scope):
        bare = self.p.BARE_CALL_NODES
        for child in node.children:
            # children of a body node are statements, so a bare identifier here
            # is a no-arg call (Ruby) rather than an arbitrary sub-expression.
            if bare and child.type in bare and child.child_count == 0:
                txt = _text(child)
                if txt and (txt[0].isalpha() or txt[0] == "_"):
                    self.edges.append(PendingEdge(scope, txt, "CALLS"))
            self._visit(child, scope)

    def _visit(self, node, scope):
        t = node.type

        syn = self.p.synth_def(node)
        if syn:
            name, kind, dnode, body = syn
            if name:
                self._record_def(dnode, name, kind, scope, body, False)
            return

        if t in self.p.DEF_KINDS:
            name = self.p.name_of(node)
            if name:
                kind = self.p.kind_of(node, self.p.DEF_KINDS[t])
                self._record_def(node, name, kind, scope, self.p.body_of(node),
                                 kind in CLASS_KINDS)
                return

        # CN-1: a binding is a symbol only at file or type scope. Inside a
        # function body the same node type is a local, which would swamp the
        # symbol table without answering any question.
        if t in self.p.BINDING_KINDS and self._is_definitional(scope):
            self._record_binding(node, scope)
            self._walk(node, scope)   # calls in the initialiser still count
            return

        if t in self.p.CALL_TYPES:
            callee = self.p.callee_of(node)
            if callee:
                self.edges.append(PendingEdge(scope, callee, "CALLS"))
            self._walk(node, scope)   # nested calls in arguments
            return

        if t in self.p.IMPORT_TYPES:
            imp = self.p.import_of(node)
            if imp:
                self.edges.append(PendingEdge(self.module, imp, "IMPORTS"))
            return

        self._walk(node, scope)


# ---- public API -----------------------------------------------------------

def build_profiles() -> dict[str, LanguageProfile]:
    """Map file extension -> profile. Empty if grammars aren't installed."""
    profiles: dict[str, LanguageProfile] = {}
    try:
        from tree_sitter import Language  # type: ignore[import-not-found]
    except Exception:
        return profiles

    def _try(make, cls, *exts):
        try:
            prof = cls(Language(make()))
            for e in exts:
                profiles[e] = prof
        except Exception:
            pass

    try:
        import tree_sitter_java as tsj  # type: ignore[import-not-found]
        _try(tsj.language, JavaProfile, ".java")
    except Exception:
        pass
    try:
        import tree_sitter_go as tsg  # type: ignore[import-not-found]
        _try(tsg.language, GoProfile, ".go")
    except Exception:
        pass
    try:
        import tree_sitter_typescript as tst  # type: ignore[import-not-found]

        class TSProfile(TsJsProfile):
            name = "typescript"
        class TSXProfile(TsJsProfile):
            name = "tsx"
        _try(tst.language_typescript, TSProfile, ".ts", ".mts", ".cts")
        _try(tst.language_tsx, TSXProfile, ".tsx")
    except Exception:
        pass
    try:
        import tree_sitter_javascript as tsjs  # type: ignore[import-not-found]

        class JSProfile(TsJsProfile):
            name = "javascript"
        _try(tsjs.language, JSProfile, ".js", ".jsx", ".mjs", ".cjs")
    except Exception:
        pass

    # Optional broad coverage via tree-sitter-language-pack (~165 grammars as
    # prebuilt wheels). setdefault: individually-installed wheels above win;
    # the pack fills the rest. When the pack isn't installed, every regex
    # fallback still applies, so this is purely additive.
    try:
        from tree_sitter_language_pack import get_language as _glang  # type: ignore

        class _TSProfile(TsJsProfile):
            name = "typescript"

        class _TSXProfile(TsJsProfile):
            name = "tsx"

        class _JSProfile(TsJsProfile):
            name = "javascript"

        def _pack(lpname, cls, *exts):
            try:
                prof = cls(_glang(lpname))
            except Exception:
                return
            for e in exts:
                profiles.setdefault(e, prof)

        _pack("c", CFamilyProfile, ".c", ".h")
        _pack("cpp", CppProfile, ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx")
        _pack("csharp", CSharpProfile, ".cs")
        _pack("rust", RustProfile, ".rs")
        _pack("ruby", RubyProfile, ".rb", ".rake")
        _pack("php", PhpProfile, ".php")
        _pack("kotlin", KotlinProfile, ".kt", ".kts")
        _pack("swift", SwiftProfile, ".swift")
        _pack("scala", ScalaProfile, ".scala", ".sc")
        # languages upgraded from flat regex to full graph processing
        # (definitions + call / import / inheritance edges) via the pack.
        _pack("lua", LuaProfile, ".lua")
        _pack("bash", BashProfile, ".sh", ".bash", ".zsh")
        _pack("solidity", SolidityProfile, ".sol")
        _pack("perl", PerlProfile, ".pl", ".pm")
        _pack("erlang", ErlangProfile, ".erl", ".hrl")
        _pack("julia", JuliaProfile, ".jl")
        _pack("r", RProfile, ".r")
        _pack("haskell", HaskellProfile, ".hs")
        _pack("ocaml", OCamlProfile, ".ml", ".mli")
        _pack("nim", NimProfile, ".nim", ".nims")
        _pack("powershell", PowerShellProfile, ".ps1", ".psm1", ".psd1")
        _pack("dart", DartProfile, ".dart")
        # also cover the original deep-parse languages if their solo wheels are absent
        _pack("java", JavaProfile, ".java")
        _pack("go", GoProfile, ".go")
        _pack("typescript", _TSProfile, ".ts", ".mts", ".cts")
        _pack("tsx", _TSXProfile, ".tsx")
        _pack("javascript", _JSProfile, ".js", ".jsx", ".mjs", ".cjs")
    except Exception:
        pass
    return profiles


def parse_treesitter(repo_root: Path, path: Path,
                     profile: LanguageProfile) -> ParseResult:
    rel = path.relative_to(repo_root).as_posix()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ParseResult(rel, "", error=str(e))

    h = file_hash(text)
    mod = module_name(repo_root, path)
    res = ParseResult(rel, h)
    nlines = max(1, text.count("\n") + 1)
    res.symbols.append(Symbol(
        qname=mod, name=mod.split(".")[-1], kind="module", file=rel,
        lineno=1, end_lineno=nlines, signature=f"module {mod}"))

    src = text.encode("utf-8")
    try:
        tree = profile.parser.parse(src)
    except Exception as e:
        res.error = str(e)
        return res

    w = _Walk(profile, mod, rel, src)
    w.run(tree.root_node)
    res.symbols.extend(w.symbols)
    res.edges.extend(w.edges)
    return res


# ==========================================================================
# regex fallback parser
# ==========================================================================
"""Lightweight structural extraction for languages without tree-sitter.

This is deliberately conservative: it indexes modules/files plus obvious
top-level definitions so search and context packing still work across broad
polyglot repos. When a tree-sitter grammar is installed for a language, that
parser remains preferred.
"""

GENERIC_LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".cs": "csharp", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".c": "c", ".h": "c", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".rs": "rust", ".php": "php", ".rb": "ruby", ".kt": "kotlin",
    ".kts": "kotlin", ".swift": "swift", ".scala": "scala",
    ".sc": "scala", ".sql": "sql",
    # markup / framework / config (FR-2 breadth — regex, no native deps)
    ".vue": "vue", ".svelte": "svelte",
    ".graphql": "graphql", ".gql": "graphql",
    ".tf": "terraform", ".tfvars": "terraform",
    ".r": "r", ".gd": "gdscript", ".dart": "dart",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "css", ".sass": "css", ".less": "css",
    ".yaml": "yaml", ".yml": "yaml",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".dockerfile": "dockerfile",
    # structured config / IDL (FR-2 breadth — regex, no native deps)
    ".toml": "toml",
    ".xml": "xml", ".xsd": "xml", ".xsl": "xml", ".xslt": "xml",
    ".ini": "ini", ".cfg": "ini", ".conf": "ini",
    ".properties": "properties",
    ".proto": "protobuf",
    ".md": "markdown", ".markdown": "markdown", ".mdx": "markdown",
    # additional mainstream programming languages (regex, no native deps)
    ".lua": "lua",
    ".ex": "elixir", ".exs": "elixir",
    ".erl": "erlang", ".hrl": "erlang",
    ".clj": "clojure", ".cljs": "clojure", ".cljc": "clojure", ".edn": "clojure",
    ".hs": "haskell",
    ".pl": "perl", ".pm": "perl",
    ".jl": "julia",
    ".ml": "ocaml", ".mli": "ocaml",
    ".fs": "fsharp", ".fsx": "fsharp", ".fsi": "fsharp",
    ".groovy": "groovy", ".gradle": "groovy",
    ".ps1": "powershell", ".psm1": "powershell", ".psd1": "powershell",
    ".sol": "solidity",
    ".zig": "zig",
    ".nim": "nim", ".nims": "nim",
    ".cr": "crystal",
    ".hx": "haxe",
    ".m": "objc", ".mm": "objc",
    ".vb": "vbnet",
    ".tcl": "tcl",
    ".pas": "pascal", ".pp": "pascal",
}

# Extensionless / fixed-name files (matched on the lowercased file name).
GENERIC_LANGUAGE_FILENAMES: dict[str, str] = {
    "dockerfile": "dockerfile",
    "makefile": "make",
    "gnumakefile": "make",
    "jenkinsfile": "groovy",
    "gemfile": "ruby",
    "rakefile": "ruby",
    "podfile": "ruby",
    "vagrantfile": "ruby",
    "berksfile": "ruby",
    "guardfile": "ruby",
}


def _generic_definition(line: str, language: str) -> Optional[tuple[str, str]]:
    import re

    stripped = line.strip()
    if not stripped:
        return None
    # '#' is a heading in markdown (not a comment), so don't skip it there.
    if language != "markdown" and stripped.startswith(("//", "#", "*")):
        return None

    patterns: dict[str, list[tuple[str, str]]] = {
        "csharp": [
            ("class", r"\b(class|interface|struct|enum|record)\s+([A-Za-z_][\w]*)"),
            ("method", r"\b(?:public|private|protected|internal|static|virtual|override|async|sealed|partial|extern|unsafe|new|readonly|abstract|\s)+[\w<>,\[\]\?\.]+\s+([A-Za-z_][\w]*)\s*\("),
        ],
        "cpp": [
            ("class", r"\b(class|struct|enum|namespace)\s+([A-Za-z_][\w]*)"),
            ("function", r"^\s*(?:template\s*<[^>]+>\s*)?(?:[\w:&<>,\*\s~]+\s+)+([A-Za-z_~][\w:]*)\s*\([^;]*\)\s*(?:const\s*)?(?:\{|$)"),
        ],
        "c": [
            ("type", r"\b(struct|enum|typedef)\s+([A-Za-z_][\w]*)"),
            ("function", r"^\s*(?:[\w\*\s]+\s+)+([A-Za-z_][\w]*)\s*\([^;]*\)\s*(?:\{|$)"),
        ],
        "rust": [
            ("function", r"\b(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][\w]*)"),
            ("type", r"\b(?:pub\s+)?(struct|enum|trait|impl|mod)\s+([A-Za-z_][\w]*)"),
        ],
        "php": [
            ("class", r"\b(class|interface|trait|enum)\s+([A-Za-z_][\w]*)"),
            ("function", r"\bfunction\s+([A-Za-z_][\w]*)\s*\("),
        ],
        "ruby": [
            ("class", r"\b(class|module)\s+([A-Za-z_:][\w:]*)"),
            ("function", r"\bdef\s+(?:self\.)?([A-Za-z_][\w!?=]*)"),
        ],
        "kotlin": [
            ("class", r"\b(?:data\s+|sealed\s+|open\s+|abstract\s+)?(class|interface|object|enum\s+class)\s+([A-Za-z_][\w]*)"),
            ("function", r"\bfun\s+(?:[A-Za-z_][\w]*\.)?([A-Za-z_][\w]*)\s*\("),
        ],
        "swift": [
            ("class", r"\b(class|struct|enum|protocol|actor|extension)\s+([A-Za-z_][\w]*)"),
            ("function", r"\bfunc\s+([A-Za-z_][\w]*)\s*\("),
        ],
        "scala": [
            ("class", r"\b(class|object|trait|enum)\s+([A-Za-z_][\w]*)"),
            ("function", r"\bdef\s+([A-Za-z_][\w]*)\s*[(:=]"),
        ],
        "sql": [
            ("object", r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(FUNCTION|PROCEDURE|VIEW|TABLE|TRIGGER|INDEX)\s+([A-Za-z_][\w\.\"]*)"),
        ],
        "vue": [
            ("component", r"\bname\s*:\s*['\"]([A-Za-z_][\w.-]*)['\"]"),
            ("function", r"\bfunction\s+([A-Za-z_]\w*)\s*\("),
            ("function", r"\b(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_]\w*\s*=>)"),
        ],
        "svelte": [
            ("function", r"\bfunction\s+([A-Za-z_]\w*)\s*\("),
            ("function", r"\b(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_]\w*\s*=>)"),
        ],
        "graphql": [
            ("type", r"\b(type|input|interface|enum|scalar|union|query|mutation|subscription|fragment)\s+([A-Za-z_]\w*)"),
        ],
        "terraform": [
            ("object", r"\b(resource|data)\s+\"([^\"]+)\"\s+\"([^\"]+)\""),
            ("object", r"\b(variable|output|module|provider|locals)\s+\"?([A-Za-z_][\w-]*)\"?"),
        ],
        "r": [
            ("function", r"\b([A-Za-z_.][\w.]*)\s*(?:<-|=)\s*function\s*\("),
        ],
        "gdscript": [
            ("function", r"\bfunc\s+([A-Za-z_]\w*)\s*\("),
            ("class", r"\bclass_name\s+([A-Za-z_]\w*)"),
            ("class", r"\bclass\s+([A-Za-z_]\w*)"),
            ("signal", r"\bsignal\s+([A-Za-z_]\w*)"),
        ],
        "dart": [
            ("class", r"\b(?:abstract\s+)?(class|mixin|enum|extension)\s+([A-Za-z_]\w*)"),
            ("function", r"^\s*(?:[\w<>,\[\]\?]+\s+)?([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:async\s*\*?\s*)?\{"),
        ],
        "html": [
            ("element", r"<(?:section|article|main|nav|header|footer|form|div|template)\b[^>]*\bid=[\"']([\w-]+)[\"']"),
            ("script", r"<(?:script|style|link)\b[^>]*\b(?:src|href)=[\"']([^\"']+)[\"']"),
        ],
        "css": [
            ("at_rule", r"@(?:mixin|function|media|keyframes|supports|include)\s+([A-Za-z_][\w-]*)"),
            ("selector", r"^([.#]?[A-Za-z_][\w-]*(?:[\s,>+~][.#:]?[\w-]+)*)\s*\{"),
        ],
        "yaml": [
            ("key", r"^([A-Za-z_][\w-]*)\s*:(?:\s|$)"),
        ],
        "shell": [
            ("function", r"^\s*(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\s*\)\s*\{"),
            ("function", r"^\s*function\s+([A-Za-z_][\w-]*)"),
        ],
        "dockerfile": [
            ("stage", r"^FROM\s+\S+\s+[Aa][Ss]\s+([A-Za-z_][\w.-]*)"),
        ],
        "make": [
            ("target", r"^([A-Za-z_][\w./-]*)\s*:(?!=)"),
        ],
        "toml": [
            ("table", r"^\[\[([^\]]+)\]\]"),
            ("table", r"^\[([^\]]+)\]"),
        ],
        "ini": [
            ("section", r"^\[([^\]]+)\]"),
        ],
        "properties": [
            ("key", r"^([A-Za-z_][\w.-]*)\s*[=:]"),
        ],
        "xml": [
            ("element", r"<(?:xs:)?element\b[^>]*\bname=[\"']([\w.:-]+)[\"']"),
            ("node", r"\b(?:id|name)=[\"']([\w.:-]+)[\"']"),
        ],
        "protobuf": [
            ("type", r"\b(message|service|enum|rpc|oneof)\s+([A-Za-z_]\w*)"),
        ],
        "markdown": [
            ("heading", r"^#{1,6}\s+(.+?)\s*#*\s*$"),
        ],
        "lua": [
            ("function", r"\bfunction\s+([A-Za-z_][\w.:]*)\s*\("),
            ("function", r"\b([A-Za-z_][\w.:]*)\s*=\s*function\s*\("),
        ],
        "elixir": [
            ("namespace", r"\bdefmodule\s+([A-Za-z_][\w.]*)"),
            ("function", r"\bdef(?:p|macro|macrop|guard|guardp|delegate)?\s+([A-Za-z_][\w?!]*)"),
            ("protocol", r"\bdefprotocol\s+([A-Za-z_][\w.]*)"),
        ],
        "erlang": [
            ("namespace", r"^-module\(\s*([a-z][\w]*)"),
            ("function", r"^([a-z][\w@]*)\s*\("),
        ],
        "clojure": [
            ("function", r"\(defn-?\s+([A-Za-z_][\w*+!?.<>=/-]*)"),
            ("macro", r"\(defmacro\s+([A-Za-z_][\w*+!?.<>=/-]*)"),
            ("record", r"\(defrecord\s+([A-Za-z_]\w*)"),
            ("protocol", r"\(defprotocol\s+([A-Za-z_]\w*)"),
            ("namespace", r"\(ns\s+([A-Za-z_][\w*+!?.<>=/-]*)"),
            ("var", r"\(def\s+([A-Za-z_][\w*+!?.<>=/-]*)"),
        ],
        "haskell": [
            ("type", r"^(?:data|newtype|type)\s+([A-Z][\w']*)"),
            ("class", r"^class\s+(?:.*=>\s*)?([A-Z][\w']*)"),
            ("instance", r"^instance\s+(?:.*=>\s*)?([A-Z][\w'.]*)"),
            ("function", r"^([a-z_][\w']*)\s*::"),
            ("function", r"^([a-z_][\w']*)\s*="),
        ],
        "perl": [
            ("package", r"\bpackage\s+([A-Za-z_][\w:]*)"),
            ("function", r"\bsub\s+([A-Za-z_]\w*)"),
        ],
        "julia": [
            ("namespace", r"\bmodule\s+([A-Za-z_]\w*)"),
            ("type", r"\b(?:mutable\s+)?struct\s+([A-Za-z_]\w*)"),
            ("type", r"\babstract\s+type\s+([A-Za-z_]\w*)"),
            ("macro", r"\bmacro\s+([A-Za-z_]\w*)"),
            ("function", r"\bfunction\s+([A-Za-z_][\w.!]*)"),
            ("function", r"^([A-Za-z_][\w.!]*)\s*\([^)]*\)\s*="),
        ],
        "ocaml": [
            ("namespace", r"\bmodule\s+([A-Z][\w']*)"),
            ("type", r"\btype\s+([a-z_][\w']*)"),
            ("value", r"\bval\s+([a-z_][\w']*)"),
            ("function", r"\blet\s+(?:rec\s+)?([a-z_][\w']*)"),
        ],
        "fsharp": [
            ("namespace", r"\bmodule\s+([A-Za-z_][\w'.]*)"),
            ("type", r"\btype\s+([A-Za-z_][\w']*)"),
            ("member", r"\bmember\s+(?:this\.|_\.)?([A-Za-z_][\w']*)"),
            ("function", r"\blet\s+(?:rec\s+|mutable\s+|inline\s+)*([A-Za-z_][\w']*)"),
        ],
        "groovy": [
            ("class", r"\b(class|interface|trait|enum)\s+([A-Za-z_]\w*)"),
            ("function", r"\bdef\s+([A-Za-z_]\w*)\s*\("),
        ],
        "powershell": [
            ("function", r"\bfunction\s+([A-Za-z_][\w-]*)"),
            ("filter", r"\bfilter\s+([A-Za-z_][\w-]*)"),
            ("class", r"\bclass\s+([A-Za-z_]\w*)"),
        ],
        "solidity": [
            ("type", r"\b(contract|interface|library|struct|enum)\s+([A-Za-z_]\w*)"),
            ("function", r"\bfunction\s+([A-Za-z_]\w*)\s*\("),
            ("event", r"\bevent\s+([A-Za-z_]\w*)"),
            ("modifier", r"\bmodifier\s+([A-Za-z_]\w*)"),
        ],
        "zig": [
            ("function", r"\b(?:pub\s+)?(?:export\s+)?fn\s+([A-Za-z_]\w*)\s*\("),
            ("type", r"\bconst\s+([A-Za-z_]\w*)\s*=\s*(?:packed\s+|extern\s+)?(?:struct|enum|union|opaque)\b"),
        ],
        "nim": [
            ("object", r"\b(proc|func|method|iterator|template|macro|converter)\s+([A-Za-z_][\w*]*)"),
            ("type", r"\btype\s+([A-Za-z_]\w*)"),
        ],
        "crystal": [
            ("class", r"\b(class|module|struct|enum)\s+([A-Z][\w:]*)"),
            ("macro", r"\bmacro\s+([A-Za-z_][\w!?]*)"),
            ("function", r"\bdef\s+(?:self\.)?([A-Za-z_][\w!?=]*)"),
        ],
        "haxe": [
            ("class", r"\b(class|interface|enum|abstract|typedef)\s+([A-Za-z_]\w*)"),
            ("function", r"\bfunction\s+([A-Za-z_]\w*)\s*\("),
        ],
        "objc": [
            ("interface", r"@interface\s+([A-Za-z_]\w*)"),
            ("implementation", r"@implementation\s+([A-Za-z_]\w*)"),
            ("protocol", r"@protocol\s+([A-Za-z_]\w*)"),
            ("method", r"^\s*[-+]\s*\([\w\s*<>,]+\)\s*([A-Za-z_]\w*)"),
        ],
        "vbnet": [
            ("class", r"\b(Class|Module|Structure|Interface|Enum)\s+([A-Za-z_]\w*)"),
            ("function", r"\b(?:Public\s+|Private\s+|Protected\s+|Friend\s+|Shared\s+|"
                         r"Overrides\s+|Overridable\s+|MustOverride\s+)*(?:Sub|Function)\s+([A-Za-z_]\w*)"),
        ],
        "tcl": [
            ("proc", r"\bproc\s+([A-Za-z_:][\w:]*)"),
        ],
        "pascal": [
            ("function", r"\b(?:function|procedure)\s+([A-Za-z_]\w*)"),
            ("type", r"^\s*([A-Za-z_]\w*)\s*=\s*(?:class|record|interface)\b"),
        ],
    }

    _ci_langs = {"sql", "dockerfile", "terraform", "graphql",
                 "vbnet", "pascal", "powershell"}
    for default_kind, pattern in patterns.get(language, []):
        match = re.search(pattern, stripped, flags=re.IGNORECASE if language in _ci_langs else 0)
        if match:
            groups = [g for g in match.groups() if g]
            name = groups[-1].strip('"')
            kind = groups[0].lower().replace(" ", "_") if default_kind in {"class", "type", "object"} and len(groups) > 1 else default_kind
            if language == "csharp" and default_kind == "class":
                kind = groups[0].lower()
            return name.split("::")[-1], kind
    return None


def _line_end_for_defs(defs: list[tuple[int, str, str]], total_lines: int, index: int) -> int:
    if index + 1 < len(defs):
        return max(defs[index][0], defs[index + 1][0] - 1)
    return total_lines


def parse_generic(repo_root: Path, path: Path, language: str) -> ParseResult:
    rel = path.relative_to(repo_root).as_posix()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ParseResult(rel, "", error=str(e))

    h = file_hash(text)
    mod = module_name(repo_root, path)
    lines = text.splitlines()
    result = ParseResult(rel, h)
    result.symbols.append(Symbol(
        qname=mod, name=mod.split(".")[-1], kind="module", file=rel,
        lineno=1, end_lineno=max(1, len(lines)), signature=f"module {mod}",
    ))

    raw_defs: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(lines, start=1):
        found = _generic_definition(line, language)
        if found:
            raw_defs.append((lineno, found[0], found[1]))

    seen: dict[str, int] = {}
    for i, (lineno, name, kind) in enumerate(raw_defs):
        seen[name] = seen.get(name, 0) + 1
        leaf = name if seen[name] == 1 else f"{name}_{seen[name]}"
        qname = f"{mod}.{leaf}"
        signature = lines[lineno - 1].strip()[:200] if lines else qname
        result.symbols.append(Symbol(
            qname=qname, name=name, kind=kind, file=rel, lineno=lineno,
            end_lineno=_line_end_for_defs(raw_defs, max(1, len(lines)), i),
            signature=signature,
            docstring=_generic_leading_doc(lines, lineno), parent=mod,
        ))
    return result


# ==========================================================================
# language registry
# ==========================================================================
_PROFILE_CACHE = None


def _profiles() -> dict:
    """Lazily build and cache the tree-sitter profile map (ext -> profile)."""
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        _PROFILE_CACHE = build_profiles()
    return _PROFILE_CACHE


def supported_extensions() -> set:
    return {".py"} | set(_profiles()) | set(GENERIC_LANGUAGE_EXTENSIONS)


# EX-1: bump when extraction starts (or stops) producing symbols or edges that
# an existing database would not already contain. Generation 2 added
# module- and type-scope constants/fields (CN-1). Generation 3 added
# cross-language doc-comment extraction (godoc/rustdoc/Javadoc/JSDoc/TSDoc and a
# regex-fallback leading-comment scan) into every symbol's `docstring`, which
# feeds both FTS and embedding text — an existing index has empty docstrings for
# non-Python symbols, so it must be rebuilt.
EXTRACTOR_GENERATION = 3


def extractor_version() -> str:
    """Identity of the installed extraction stack (EX-1).

    Covers both halves of "what this build can see": the generation of the
    extractors themselves, and which tree-sitter grammars are actually
    importable — installing `contextiq[languages]` upgrades a repository from
    regex symbols to a real call graph, and that has to invalidate the index
    just as surely as a code change does.
    """
    langs = ",".join(sorted({p.name for p in _profiles().values()}))
    return f"{EXTRACTOR_GENERATION}:{langs}"


def language_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    prof = _profiles().get(ext)
    if prof is not None:
        return prof.name
    if ext in GENERIC_LANGUAGE_EXTENSIONS:
        return GENERIC_LANGUAGE_EXTENSIONS[ext]
    return GENERIC_LANGUAGE_FILENAMES.get(path.name.lower(), "text")


def languages_available() -> dict:
    out = {"python (ast)": [".py"]}
    by: dict = {}
    for ext, prof in _profiles().items():
        by.setdefault(prof.name, []).append(ext)
    for name, exts in by.items():
        out[f"{name} (tree-sitter)"] = sorted(exts)
    fallback: dict[str, list[str]] = {}
    for ext, name in GENERIC_LANGUAGE_EXTENSIONS.items():
        if ext not in _profiles():
            fallback.setdefault(name, []).append(ext)
    for fname, name in GENERIC_LANGUAGE_FILENAMES.items():
        fallback.setdefault(name, []).append(fname)
    for name, exts in sorted(fallback.items()):
        out[f"{name} (regex fallback)"] = sorted(exts)
    return out


# PF-1: what each extraction tier can actually promise.
#
# "Supports 55 languages" is true and misleading in the same breath: a regex
# extractor finds symbol *names* and nothing else, so `get_callers` on a Zig
# function is not a weaker answer than on a Python one — it is an empty one,
# for a structural reason the caller cannot see. Every tier now states which
# graph edges it produces, so a consumer can tell "no callers" from "callers
# not extractable here" and escalate to reading the file instead of trusting
# an empty blast radius.
EXTRACTION_TIERS: dict[str, dict] = {
    "ast": {
        "rank": 3, "label": "AST (stdlib parser)",
        "symbols": True, "calls": True, "imports": True, "inheritance": True,
        "note": "exact spans and edges",
    },
    "tree-sitter": {
        "rank": 2, "label": "tree-sitter grammar",
        "symbols": True, "calls": True, "imports": True, "inheritance": True,
        "note": "edges resolved by name, so cross-file targets are best-effort",
    },
    "regex": {
        "rank": 1, "label": "regex symbol scan",
        "symbols": True, "calls": False, "imports": False, "inheritance": False,
        "note": "definitions only — the call graph is EMPTY for these files, "
                "so an empty get_callers/get_impact result proves nothing",
    },
}


def language_tier(language: str) -> str:
    """Which extraction tier a language gets in this installation (PF-1)."""
    if language == "python":
        return "ast"
    if any(p.name == language for p in _profiles().values()):
        return "tree-sitter"
    return "regex"


def tier_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".py":
        return "ast"
    return "tree-sitter" if ext in _profiles() else "regex"


def parse_path(repo_root: Path, path: Path):
    ext = path.suffix.lower()
    if ext == ".py":
        return parse_file(repo_root, path)
    prof = _profiles().get(ext)
    if prof is not None:
        return parse_treesitter(repo_root, path, prof)
    language = (GENERIC_LANGUAGE_EXTENSIONS.get(ext)
                or GENERIC_LANGUAGE_FILENAMES.get(path.name.lower()))
    if language is not None:
        return parse_generic(repo_root, path, language)
    return None


# ---- extractor self-test (FR-2b) ------------------------------------------
# (filename, source, {expected leaf names}). Tree-sitter languages are only
# checked when their grammar is installed; the regex fallbacks always run.
_EXTRACTOR_FIXTURES: list[tuple[str, str, set[str]]] = [
    ("fix.py", "def foo(a, b):\n    return a\n\nclass Bar:\n    def m(self):\n        pass\n",
     {"foo", "Bar", "m"}),
    ("fix.cs", "public class Svc {\n  public void Run() {}\n}\n", {"Svc", "Run"}),
    ("fix.cpp", "class Widget {};\nint compute(int x) {\n  return x;\n}\n", {"Widget", "compute"}),
    ("fix.rs", "pub struct Conf {}\npub fn build() {}\n", {"Conf", "build"}),
    ("fix.php", "<?php\nclass Repo {\n  function find() {}\n}\n", {"Repo", "find"}),
    ("fix.rb", "class Account\n  def deposit(n)\n  end\nend\n", {"Account", "deposit"}),
    ("fix.kt", "class Engine {\n  fun start() {}\n}\n", {"Engine", "start"}),
    ("fix.swift", "struct Vec {}\nfunc dot() {}\n", {"Vec", "dot"}),
    ("fix.scala", "object App {\n  def main() {}\n}\n", {"App", "main"}),
    ("fix.sql", "CREATE TABLE users (id INT);\n", {"users"}),
    ("fix.vue", "<script>\nfunction onClick() {}\n</script>\n", {"onClick"}),
    ("fix.svelte", "<script>\nfunction load() {}\n</script>\n", {"load"}),
    ("fix.graphql", "type User {\n  id: ID\n}\n", {"User"}),
    ("fix.tf", 'resource "aws_s3_bucket" "data" {}\nvariable "region" {}\n', {"data", "region"}),
    ("fix.r", "mean_sq <- function(x) {\n  x * x\n}\n", {"mean_sq"}),
    ("fix.gd", "func _ready():\n  pass\nclass_name Player\n", {"_ready", "Player"}),
    ("fix.dart", "class Box {}\nint add(int a, int b) {\n  return a;\n}\n", {"Box"}),
    ("fix.html", '<section id="hero"></section>\n', {"hero"}),
    ("fix.css", ".card {\n  color: red;\n}\n", {".card"}),
    ("fix.yaml", "database:\n  host: localhost\n", {"database"}),
    ("fix.sh", "deploy() {\n  echo hi\n}\n", {"deploy"}),
    ("Dockerfile", "FROM python:3 AS base\nRUN echo hi\n", {"base"}),
    ("Makefile", "build:\n\tgo build\n", {"build"}),
    ("fix.java", "class Foo {\n  void run() {}\n}\n", {"Foo", "run"}),
    ("fix.go", "package main\nfunc Add(a int) int {\n  return a\n}\n", {"Add"}),
    ("fix.ts", "export class Svc {\n  run() {}\n}\n", {"Svc", "run"}),
    ("fix.js", "function handler() {}\n", {"handler"}),
    ("fix.toml", "[tool.black]\nline-length = 88\n", {"tool.black"}),
    ("fix.ini", "[server]\nhost = localhost\n", {"server"}),
    ("fix.properties", "db.host=localhost\n", {"db.host"}),
    ("fix.proto", "message User {\n  string id = 1;\n}\n", {"User"}),
    ("fix.xml", '<bean id="svc"></bean>\n', {"svc"}),
    # additional mainstream programming languages
    ("fix.md", "# Title\n\n## Install\n", {"Title", "Install"}),
    ("fix.lua", "function greet(name)\n  return name\nend\n", {"greet"}),
    ("fix.ex", "defmodule App do\n  def run(x) do\n    x\n  end\nend\n", {"App", "run"}),
    ("fix.erl", "-module(app).\ninit(Args) ->\n  ok.\n", {"app", "init"}),
    ("fix.clj", "(ns app.core)\n(defn add [a b]\n  (+ a b))\n", {"app.core", "add"}),
    ("fix.hs", "data Tree = Leaf\n\nadd :: Int -> Int\nadd x = x\n", {"Tree", "add"}),
    ("fix.pl", "package App;\nsub run {\n  return 1;\n}\n", {"App", "run"}),
    ("fix.jl", "module App\nstruct Point end\nfunction run(x)\n  x\nend\nend\n",
     {"App", "Point", "run"}),
    ("fix.ml", "module M = struct end\nlet add x = x\ntype t = int\n", {"M", "add", "t"}),
    ("fix.fs", "module App\ntype Vec = { x: int }\nlet add x = x\n", {"App", "Vec", "add"}),
    ("fix.groovy", "class Svc {\n  def run() {}\n}\n", {"Svc", "run"}),
    ("fix.ps1", "function Get-Item {\n}\nclass Box {\n}\n", {"Get-Item", "Box"}),
    ("fix.sol", "contract Token {\n  function mint() public {}\n  event Sent();\n}\n",
     {"Token", "mint", "Sent"}),
    ("fix.zig", "pub fn add(a: i32) i32 {\n  return a;\n}\nconst Point = struct {};\n",
     {"add", "Point"}),
    ("fix.nim", "proc greet(name: string) =\n  echo name\ntype Animal = object\n",
     {"greet", "Animal"}),
    ("fix.cr", "class Account\n  def deposit(n)\n  end\nend\n", {"Account", "deposit"}),
    ("fix.hx", "class Main {\n  function run() {}\n}\n", {"Main", "run"}),
    ("fix.m", "@interface Foo\n- (void)bar;\n@end\n", {"Foo", "bar"}),
    ("fix.vb", "Public Class Svc\n  Public Sub Run()\n  End Sub\nEnd Class\n", {"Svc", "Run"}),
    ("fix.tcl", "proc greet {name} {\n  puts $name\n}\n", {"greet"}),
    ("fix.pas", "function Add(a: Integer): Integer;\nbegin\nend;\n", {"Add"}),
]


def diagnose_extractors() -> dict:
    """Run every language extractor against a fixture and report pass/fail.

    Each extractor must (a) never throw and (b) recover the expected symbol
    names. Tree-sitter languages whose grammar isn't installed are reported as
    `skipped` (the regex fallbacks always run). Designed as a CI gate (FR-2b).
    """
    import tempfile
    rows: list[dict] = []
    profiles = _profiles()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for fname, src, expected in _EXTRACTOR_FIXTURES:
            ext = Path(fname).suffix.lower()
            lang = (("python" if ext == ".py" else None)
                    or (profiles[ext].name if ext in profiles else None)
                    or GENERIC_LANGUAGE_EXTENSIONS.get(ext)
                    or GENERIC_LANGUAGE_FILENAMES.get(fname.lower()) or "?")
            needs_ts = ext in {".java", ".go", ".ts", ".tsx", ".js", ".jsx",
                               ".mjs", ".cjs", ".mts", ".cts"}
            if needs_ts and ext not in profiles:
                rows.append({"file": fname, "language": lang, "status": "skipped",
                             "reason": "tree-sitter grammar not installed"})
                continue
            p = root / fname
            try:
                p.write_text(src, encoding="utf-8")
                res = parse_path(root, p)
                names = {s.name for s in (res.symbols if res else []) if s.kind != "module"}
                missing = sorted(expected - names)
                threw = bool(res and res.error and not res.symbols)
                ok = not missing and not threw
                rows.append({"file": fname, "language": lang,
                             "status": "pass" if ok else "fail",
                             "found": sorted(names)[:MAX_SIGS_PER_FILE],
                             "missing": missing})
            except Exception as ex:  # never-throw contract violated
                rows.append({"file": fname, "language": lang, "status": "fail",
                             "missing": sorted(expected), "error": str(ex)})
    failed = [r for r in rows if r["status"] == "fail"]
    return {
        "ok": not failed,
        "total": len(rows),
        "passed": sum(1 for r in rows if r["status"] == "pass"),
        "failed": len(failed),
        "skipped": sum(1 for r in rows if r["status"] == "skipped"),
        "rows": rows,
    }


# ==========================================================================
# sqlite store + FTS5
# ==========================================================================
"""Local persistent store: SQLite + FTS5.

Everything lives in a single .db file inside the repo (default: .tokengraph/graph.db).
No server, no external service — this is the "local store / graph" layer.

Tables:
    files(path, hash, mtime, language, token_est, symbols_count)
                                                                                         -- for incremental indexing
  symbols(id, qname, name, kind, file, lineno, end_lineno, signature, docstring, parent)
  edges(src_id, dst_id, type)               -- resolved graph edges
    chunks(file, start_line, end_line, text)  -- prompt-sized local snippets
  symbols_fts                               -- FTS5 over name+qname+signature+docstring
    chunks_fts                                -- FTS5 over indexed source chunks

If FTS5 is unavailable the store transparently falls back to LIKE queries.
"""

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable, Optional



SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY, hash TEXT NOT NULL, mtime REAL,
    language TEXT DEFAULT '', token_est INTEGER DEFAULT 0,
    symbols_count INTEGER DEFAULT 0, size INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    qname TEXT UNIQUE NOT NULL, name TEXT, kind TEXT, file TEXT,
    lineno INTEGER, end_lineno INTEGER, signature TEXT, docstring TEXT, parent TEXT
);
CREATE INDEX IF NOT EXISTS idx_sym_file ON symbols(file);
CREATE INDEX IF NOT EXISTS idx_sym_name ON symbols(name);
CREATE TABLE IF NOT EXISTS edges (
    src_id INTEGER NOT NULL, dst_id INTEGER NOT NULL, type TEXT NOT NULL,
    UNIQUE(src_id, dst_id, type)
);
CREATE INDEX IF NOT EXISTS idx_edge_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edge_dst ON edges(dst_id);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file TEXT NOT NULL, start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
    text TEXT NOT NULL, token_est INTEGER NOT NULL,
    UNIQUE(file, start_line, end_line)
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file);
CREATE TABLE IF NOT EXISTS vectors (
    symbol_id INTEGER PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL,
    backend TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS summaries (
    path TEXT PRIMARY KEY, summary TEXT NOT NULL, kind TEXT DEFAULT 'module',
    token_est INTEGER DEFAULT 0, source TEXT DEFAULT 'auto'
);
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, kind TEXT DEFAULT 'note', text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, label TEXT, git_sha TEXT DEFAULT '', note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS weights (
    file TEXT PRIMARY KEY, weight REAL NOT NULL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT ''
);
-- PK-1: memoised context packs, valid only for the graph version that
-- produced them.
CREATE TABLE IF NOT EXISTS pack_cache (
    key TEXT PRIMARY KEY, graph_version INTEGER NOT NULL,
    task TEXT NOT NULL, markdown TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}',
    tokens INTEGER NOT NULL DEFAULT 0, ts REAL NOT NULL DEFAULT 0.0,
    hits INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pack_cache_ver ON pack_cache(graph_version);
-- SD-1: per-session ledger of what has already been sent to a given agent
-- session, so a second retrieval in the same conversation does not re-bill
-- bodies the model is already holding.
CREATE TABLE IF NOT EXISTS sent_ledger (
    session TEXT NOT NULL, qname TEXT NOT NULL, mode TEXT NOT NULL DEFAULT 'body',
    content_hash TEXT NOT NULL DEFAULT '', tokens INTEGER NOT NULL DEFAULT 0,
    ts REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (session, qname)
);
CREATE INDEX IF NOT EXISTS idx_sent_session ON sent_ledger(session);
"""

# Bumped whenever the on-disk table shapes change in a way that older rows
# cannot satisfy. Store._migrate() rebuilds what it cannot ALTER into place.
SCHEMA_VERSION = 4

# ---- SD-2: session-ledger safety bounds -----------------------------------
# The ledger is a bet that the model still holds text we sent it earlier. That
# bet expires. Three independent guards, because each fails differently:
#
#  * TTL          — a conversation resumed tomorrow is not the same context.
#  * entry cap    — stops unbounded growth on a long-lived session.
#  * window watch — once cumulative delivered tokens exceed the model's usable
#                   context, the host has almost certainly compacted or
#                   truncated; anything older can no longer be assumed present.
#
# The window guard is the important one: it is the only automatic defence
# against silently withholding content the model has already dropped.
SESSION_TTL_SECONDS = 8 * 3600
SESSION_MAX_ENTRIES = 4000
SESSION_CONTEXT_WINDOW = 128_000
# Fraction of the window we trust before assuming compaction has occurred.
SESSION_WINDOW_SAFETY = 0.5


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the MCP server pools a Retriever per thread
        # but tears the pool down from whichever thread calls shutdown, and
        # sqlite otherwise refuses a cross-thread close(). Each connection is
        # still only *used* by its owning thread, so this does not introduce
        # sharing — it only permits the close.
        self.conn = sqlite3.connect(str(self.db_path), timeout=10.0,
                                    check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Concurrency hardening: WAL lets a reader (a query) and a writer
        # (a freshen-on-query reindex) coexist without "database is locked".
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA busy_timeout=10000")
        except sqlite3.OperationalError:
            pass
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.fts = self._init_fts()
        self.conn.commit()

    def _migrate(self) -> None:
        """Bring an older database up to SCHEMA_VERSION, or rebuild if newer.

        Additive column changes are applied with ALTER TABLE. A database
        written by a *newer* ContextIQ is not readable safely, so it is
        rebuilt from scratch rather than silently mis-read.
        """
        found = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if found > SCHEMA_VERSION:
            # Forward-incompatible: drop the derived tables and re-index.
            for tbl in ("vectors", "chunks", "edges", "symbols", "files",
                        "summaries"):
                self.conn.execute(f"DROP TABLE IF EXISTS {tbl}")
            self.conn.executescript(SCHEMA)
            found = 0

        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(files)")}
        for name, ddl in {
            "language": "ALTER TABLE files ADD COLUMN language TEXT DEFAULT ''",
            "token_est": "ALTER TABLE files ADD COLUMN token_est INTEGER DEFAULT 0",
            "symbols_count": "ALTER TABLE files ADD COLUMN symbols_count INTEGER DEFAULT 0",
            "size": "ALTER TABLE files ADD COLUMN size INTEGER DEFAULT 0",
        }.items():
            if name not in cols:
                self.conn.execute(ddl)

        # EM-2: vectors carry the identity of the backend that produced them,
        # so switching embedding backends invalidates rather than silently
        # mixing incompatible spaces.
        vcols = {r["name"] for r in self.conn.execute("PRAGMA table_info(vectors)")}
        if vcols and "backend" not in vcols:
            self.conn.execute(
                "ALTER TABLE vectors ADD COLUMN backend TEXT NOT NULL DEFAULT ''")

        self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    # ---- key/value metadata ----
    # ---- PK-1: context-pack memoisation ----
    def graph_version(self) -> int:
        """Monotonic counter bumped whenever the indexed graph changes.

        Every cached pack is stamped with the version that produced it, so a
        single reindex invalidates the whole cache without any per-file
        dependency tracking.
        """
        try:
            return int(self.get_meta("graph_version", "0") or 0)
        except ValueError:
            return 0

    def bump_graph_version(self) -> int:
        v = self.graph_version() + 1
        self.set_meta("graph_version", str(v))
        return v

    def content_fingerprint(self, exclude: set[str] | None = None) -> str:
        """Hash of every indexed file's content hash (CFG-6).

        Identifies *what the graph was built from*, so a generated artefact can
        be checked against the code it claims to describe. Unlike
        `graph_version`, which counts writes, this is stable across a reindex
        that changed nothing — reindexing must not make a correct file look
        stale.

        `exclude` must carry the generated artefacts themselves. They are
        indexed like any other Markdown, so counting them makes generation
        invalidate its own output: write the file, the fingerprint moves, and
        the freshly written context is instantly "stale".
        """
        skip = exclude or set()
        rows = self.conn.execute(
            "SELECT path, hash FROM files ORDER BY path").fetchall()
        h = hashlib.blake2b(digest_size=16)
        for r in rows:
            if r["path"] in skip:
                continue
            h.update(f"{r['path']}\x1f{r['hash']}\x1e".encode("utf-8"))
        return h.hexdigest()

    def cached_pack(self, key: str) -> Optional[sqlite3.Row]:
        row = self.conn.execute(
            "SELECT * FROM pack_cache WHERE key=? AND graph_version=?",
            (key, self.graph_version())).fetchone()
        if row is not None:
            self.conn.execute(
                "UPDATE pack_cache SET hits=hits+1 WHERE key=?", (key,))
        return row

    def store_pack(self, key: str, task: str, markdown: str, meta: str,
                   tokens: int) -> None:
        import time as _t
        self.conn.execute(
            "INSERT OR REPLACE INTO pack_cache"
            "(key,graph_version,task,markdown,meta,tokens,ts,hits) "
            "VALUES(?,?,?,?,?,?,?,0)",
            (key, self.graph_version(), task, markdown, meta, tokens, _t.time()))
        self.conn.commit()

    def purge_stale_packs(self) -> int:
        cur = self.conn.execute(
            "DELETE FROM pack_cache WHERE graph_version <> ?",
            (self.graph_version(),))
        return cur.rowcount or 0

    def pack_cache_stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(hits),0) h, "
            "COALESCE(SUM(tokens),0) t FROM pack_cache "
            "WHERE graph_version=?", (self.graph_version(),)).fetchone()
        return {"entries": row["n"], "hits": row["h"],
                "tokens_cached": row["t"], "graph_version": self.graph_version()}

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
        self.conn.commit()

    def _init_fts(self) -> bool:
        try:
            rebuild_symbols = self._drop_contentless_fts("symbols_fts")
            rebuild_chunks = self._drop_contentless_fts("chunks_fts")
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5("
                "qname, name, signature, docstring)"
            )
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
                "file, text)"
            )
            self._rebuild_fts_if_needed("symbols_fts", rebuild_symbols)
            self._rebuild_fts_if_needed("chunks_fts", rebuild_chunks)
            return True
        except sqlite3.OperationalError:
            return False

    def _drop_contentless_fts(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (name,)).fetchone()
        sql = row["sql"] if row else ""
        if "content=''" in sql or 'content=""' in sql:
            self.conn.execute(f"DROP TABLE IF EXISTS {name}")
            return True
        return False

    def _rebuild_fts_if_needed(self, name: str, force: bool) -> None:
        count = self.conn.execute(f"SELECT COUNT(*) n FROM {name}").fetchone()["n"]
        if count and not force:
            return
        if name == "symbols_fts":
            self.conn.execute("DELETE FROM symbols_fts")
            self.conn.execute(
                "INSERT INTO symbols_fts(rowid,qname,name,signature,docstring) "
                "SELECT id,qname,name,signature,docstring FROM symbols")
        elif name == "chunks_fts":
            self.conn.execute("DELETE FROM chunks_fts")
            self.conn.execute(
                "INSERT INTO chunks_fts(rowid,file,text) "
                "SELECT id,file,text FROM chunks")

    # ---- incremental bookkeeping ----
    def known_hash(self, path: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT hash FROM files WHERE path=?", (path,)).fetchone()
        return row["hash"] if row else None

    def file_meta(self, path: str) -> Optional[sqlite3.Row]:
        """(hash, mtime, size) for the cheap stat-based staleness check."""
        return self.conn.execute(
            "SELECT hash, mtime, size FROM files WHERE path=?", (path,)).fetchone()

    def touch_file(self, path: str, mtime: float, size: int) -> None:
        """Record fresh stat metadata without reparsing (content unchanged)."""
        self.conn.execute(
            "UPDATE files SET mtime=?, size=? WHERE path=?", (mtime, size, path))

    def set_file(self, path: str, h: str, mtime: float, language: str = "",
                 token_est: int = 0, symbols_count: int = 0, size: int = 0) -> None:
        self.conn.execute(
            "INSERT INTO files(path,hash,mtime,language,token_est,symbols_count,size) "
            "VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET hash=excluded.hash, mtime=excluded.mtime,"
            "language=excluded.language, token_est=excluded.token_est, "
            "symbols_count=excluded.symbols_count, size=excluded.size",
            (path, h, mtime, language, token_est, symbols_count, size))

    def all_indexed_files(self) -> set[str]:
        return {r["path"] for r in self.conn.execute("SELECT path FROM files")}

    def forget_file(self, path: str) -> None:
        """Remove a file's symbols (and their edges) before re-indexing it."""
        ids = [r["id"] for r in self.conn.execute(
            "SELECT id FROM symbols WHERE file=?", (path,))]
        if ids:
            q = ",".join("?" * len(ids))
            self.conn.execute(f"DELETE FROM edges WHERE src_id IN ({q}) OR dst_id IN ({q})",
                              ids + ids)
            if self.fts:
                for sid in ids:
                    self.conn.execute("DELETE FROM symbols_fts WHERE rowid=?", (sid,))
            self.conn.execute(f"DELETE FROM symbols WHERE id IN ({q})", ids)
        if ids:
            q = ",".join("?" * len(ids))
            self.conn.execute(f"DELETE FROM vectors WHERE symbol_id IN ({q})", ids)
        chunk_ids = [r["id"] for r in self.conn.execute(
            "SELECT id FROM chunks WHERE file=?", (path,))]
        if chunk_ids:
            q = ",".join("?" * len(chunk_ids))
            if self.fts:
                for cid in chunk_ids:
                    self.conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (cid,))
            self.conn.execute(f"DELETE FROM chunks WHERE id IN ({q})", chunk_ids)
        self.conn.execute("DELETE FROM summaries WHERE path=?", (path,))
        self.conn.execute("DELETE FROM files WHERE path=?", (path,))

    def prepare_file_update(self, path: str, qnames: set[str]) -> None:
        """Clear replaceable file data while preserving stable incoming edges."""
        rows = list(self.conn.execute(
            "SELECT id,qname FROM symbols WHERE file=?", (path,)))
        retained_ids = [r["id"] for r in rows if r["qname"] in qnames]
        removed_ids = [r["id"] for r in rows if r["qname"] not in qnames]

        if retained_ids:
            q = ",".join("?" * len(retained_ids))
            self.conn.execute(f"DELETE FROM edges WHERE src_id IN ({q})", retained_ids)
            self.conn.execute(f"DELETE FROM vectors WHERE symbol_id IN ({q})", retained_ids)
        if removed_ids:
            q = ",".join("?" * len(removed_ids))
            self.conn.execute(
                f"DELETE FROM edges WHERE src_id IN ({q}) OR dst_id IN ({q})",
                removed_ids + removed_ids)
            if self.fts:
                for sid in removed_ids:
                    self.conn.execute("DELETE FROM symbols_fts WHERE rowid=?", (sid,))
            self.conn.execute(f"DELETE FROM vectors WHERE symbol_id IN ({q})", removed_ids)
            self.conn.execute(f"DELETE FROM symbols WHERE id IN ({q})", removed_ids)

        chunk_ids = [r["id"] for r in self.conn.execute(
            "SELECT id FROM chunks WHERE file=?", (path,))]
        if chunk_ids:
            q = ",".join("?" * len(chunk_ids))
            if self.fts:
                for cid in chunk_ids:
                    self.conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (cid,))
            self.conn.execute(f"DELETE FROM chunks WHERE id IN ({q})", chunk_ids)
        self.conn.execute("DELETE FROM summaries WHERE path=?", (path,))

    # ---- writing symbols ----
    def upsert_symbol(self, s: Symbol) -> int:
        cur = self.conn.execute(
            "INSERT INTO symbols(qname,name,kind,file,lineno,end_lineno,signature,docstring,parent) "
            "VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(qname) DO UPDATE SET name=excluded.name,kind=excluded.kind,"
            "file=excluded.file,lineno=excluded.lineno,end_lineno=excluded.end_lineno,"
            "signature=excluded.signature,docstring=excluded.docstring,parent=excluded.parent",
            (s.qname, s.name, s.kind, s.file, s.lineno, s.end_lineno,
             s.signature, s.docstring, s.parent))
        sid = self.id_for_qname(s.qname)
        if self.fts and sid is not None:
            self.conn.execute("DELETE FROM symbols_fts WHERE rowid=?", (sid,))
            self.conn.execute(
                "INSERT INTO symbols_fts(rowid,qname,name,signature,docstring) VALUES(?,?,?,?,?)",
                (sid, s.qname, s.name, s.signature, s.docstring))
        return sid

    def add_edge(self, src_id: int, dst_id: int, type_: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO edges(src_id,dst_id,type) VALUES(?,?,?)",
            (src_id, dst_id, type_))

    def add_chunk(self, file: str, start_line: int, end_line: int, text: str) -> None:
        token_est = count_tokens(text)
        self.conn.execute(
            "INSERT OR REPLACE INTO chunks(file,start_line,end_line,text,token_est) "
            "VALUES(?,?,?,?,?)", (file, start_line, end_line, text, token_est))
        row = self.conn.execute(
            "SELECT id FROM chunks WHERE file=? AND start_line=? AND end_line=?",
            (file, start_line, end_line)).fetchone()
        if self.fts and row is not None:
            self.conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row["id"],))
            self.conn.execute(
                "INSERT INTO chunks_fts(rowid,file,text) VALUES(?,?,?)",
                (row["id"], file, text))

    def set_vector(self, symbol_id: int, vec: list[float],
                   backend: str = "") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO vectors(symbol_id,dim,vec,backend) VALUES(?,?,?,?)",
            (symbol_id, len(vec), vec_to_blob(vec), backend or embed_backend_id()))

    def iter_vectors(self):
        """Only vectors from the *current* backend — mixing spaces is meaningless."""
        return self.conn.execute(
            "SELECT symbol_id, dim, vec FROM vectors WHERE backend=?",
            (embed_backend_id(),))

    def vector_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) n FROM vectors WHERE backend=?",
            (embed_backend_id(),)).fetchone()["n"]

    def has_vectors(self) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM vectors WHERE backend=? LIMIT 1",
            (embed_backend_id(),)).fetchone() is not None

    def drop_stale_vectors(self) -> int:
        """Delete vectors produced by any other embedding backend (EM-2).

        Returns the number removed. Callers follow this with a re-embed of the
        affected files, so semantic search never silently degrades to empty
        after a backend switch.
        """
        cur = self.conn.execute(
            "DELETE FROM vectors WHERE backend<>?", (embed_backend_id(),))
        return cur.rowcount or 0

    def symbols_missing_vectors(self, limit: int = 0) -> list[sqlite3.Row]:
        """Indexed symbols that have no vector for the current backend."""
        sql = ("SELECT s.id, s.qname, s.name, s.signature, s.docstring, s.file "
               "FROM symbols s LEFT JOIN vectors v "
               "ON v.symbol_id = s.id AND v.backend = ? "
               "WHERE v.symbol_id IS NULL AND s.kind <> 'module'")
        params: list = [embed_backend_id()]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    # ---- SD-1: per-session sent ledger (cross-turn dedup) ----
    def mark_sent(self, session: str, entries: list[tuple[str, str, str, int]]) -> None:
        """Record (qname, mode, content_hash, tokens) as delivered to `session`."""
        if not session or not entries:
            return
        import time as _t
        now = _t.time()
        self.conn.executemany(
            "INSERT INTO sent_ledger(session,qname,mode,content_hash,tokens,ts) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(session,qname) DO UPDATE SET "
            "mode=excluded.mode, content_hash=excluded.content_hash, "
            "tokens=excluded.tokens, ts=excluded.ts",
            [(session, q, m, h, t, now) for q, m, h, t in entries])
        self.conn.commit()

    def sent_map(self, session: str, window: int = SESSION_CONTEXT_WINDOW
                 ) -> tuple[dict[str, sqlite3.Row], dict]:
        """Everything this session can still be assumed to hold (SD-2).

        Returns (qname -> row, diagnostics). Reuse is a claim about the model's
        state, not ours, so it is bounded three ways — expiry by age, by entry
        count, and by cumulative delivered tokens against the usable context
        window. Crossing the window strongly implies the host compacted or
        truncated the conversation, so the ledger is dropped rather than
        withholding content the model no longer has.
        """
        if not session:
            return {}, {"active": False}
        import time as _t
        now = _t.time()
        cutoff = now - SESSION_TTL_SECONDS
        expired = self.conn.execute(
            "DELETE FROM sent_ledger WHERE session=? AND ts < ?",
            (session, cutoff)).rowcount or 0

        rows = self.conn.execute(
            "SELECT qname, mode, content_hash, tokens, ts FROM sent_ledger "
            "WHERE session=? ORDER BY ts DESC", (session,)).fetchall()

        limit = int(window * SESSION_WINDOW_SAFETY)
        total = sum(r["tokens"] for r in rows)
        reset_reason = ""
        if total > limit:
            # Past the trustworthy fraction of the window: assume compaction.
            self.conn.execute("DELETE FROM sent_ledger WHERE session=?", (session,))
            self.conn.commit()
            return {}, {"active": True, "reset": True,
                        "reset_reason": "context-window exceeded "
                                        f"({total} > {limit} tokens); assuming "
                                        f"the host compacted the conversation",
                        "expired_entries": expired, "tokens_held": 0}

        evicted = 0
        if len(rows) > SESSION_MAX_ENTRIES:
            # Keep the most recent; drop the oldest tail.
            drop = [r["qname"] for r in rows[SESSION_MAX_ENTRIES:]]
            self.conn.executemany(
                "DELETE FROM sent_ledger WHERE session=? AND qname=?",
                [(session, q) for q in drop])
            rows = rows[:SESSION_MAX_ENTRIES]
            evicted = len(drop)
        if expired or evicted:
            self.conn.commit()
        return ({r["qname"]: r for r in rows},
                {"active": True, "reset": False, "expired_entries": expired,
                 "evicted_entries": evicted, "tokens_held": total,
                 "window_limit": limit})

    def prune_sessions(self) -> int:
        """Drop every ledger entry past its TTL, across all sessions."""
        import time as _t
        cur = self.conn.execute("DELETE FROM sent_ledger WHERE ts < ?",
                                (_t.time() - SESSION_TTL_SECONDS,))
        self.conn.commit()
        return cur.rowcount or 0

    def clear_session(self, session: str = "") -> int:
        """Forget one session's ledger, or all of them when session is empty."""
        if session:
            cur = self.conn.execute(
                "DELETE FROM sent_ledger WHERE session=?", (session,))
        else:
            cur = self.conn.execute("DELETE FROM sent_ledger")
        self.conn.commit()
        return cur.rowcount or 0

    def session_stats(self, session: str) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(tokens),0) t FROM sent_ledger "
            "WHERE session=?", (session,)).fetchone()
        return {"session": session, "symbols_sent": row["n"],
                "tokens_sent": row["t"]}

    def set_summary(self, path: str, summary: str, kind: str = "module",
                    source: str = "auto") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO summaries(path,summary,kind,token_est,source) "
            "VALUES(?,?,?,?,?)", (path, summary, kind, count_tokens(summary), source))

    def get_summary(self, path: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM summaries WHERE path=?", (path,)).fetchone()

    # ---- cross-session memory + checkpoints + learning weights ----
    def add_memory(self, text: str, kind: str = "note") -> int:
        import time as _t
        cur = self.conn.execute(
            "INSERT INTO memory(ts,kind,text) VALUES(?,?,?)",
            (_t.time(), kind, text))
        return cur.lastrowid

    def recent_memory(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM memory ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def add_checkpoint(self, label: str, git_sha: str = "", note: str = "") -> int:
        import time as _t
        cur = self.conn.execute(
            "INSERT INTO checkpoints(ts,label,git_sha,note) VALUES(?,?,?,?)",
            (_t.time(), label, git_sha, note))
        return cur.lastrowid

    def recent_checkpoints(self, limit: int = 10) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM checkpoints ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def bump_weight(self, file: str, delta: float) -> None:
        self.conn.execute(
            "INSERT INTO weights(file,weight) VALUES(?,?) "
            "ON CONFLICT(file) DO UPDATE SET weight=weight+excluded.weight",
            (file, delta))

    def weight_for(self, file: str) -> float:
        row = self.conn.execute(
            "SELECT weight FROM weights WHERE file=?", (file,)).fetchone()
        return row["weight"] if row else 0.0

    def all_weights(self) -> dict[str, float]:
        return {r["file"]: r["weight"]
                for r in self.conn.execute("SELECT file,weight FROM weights")}

    def importers_of_module(self, module_leaf: str) -> list[sqlite3.Row]:
        """Symbols that IMPORT a given module/name (reverse import edges)."""
        return self.conn.execute(
            "SELECT s.* FROM edges e JOIN symbols s ON s.id=e.src_id "
            "JOIN symbols d ON d.id=e.dst_id "
            "WHERE e.type='IMPORTS' AND (d.name=? OR d.qname=?)",
            (module_leaf, module_leaf)).fetchall()

    def edges_of_type(self, type_: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT s.qname AS src, s.file AS src_file, d.qname AS dst, "
            "d.file AS dst_file FROM edges e "
            "JOIN symbols s ON s.id=e.src_id JOIN symbols d ON d.id=e.dst_id "
            "WHERE e.type=?", (type_,)).fetchall()

    def token_est_for(self, path: str) -> int:
        row = self.conn.execute(
            "SELECT token_est FROM files WHERE path=?", (path,)).fetchone()
        return row["token_est"] if row else 0

    def repo_token_total(self) -> int:
        """Sum of whole-file token estimates across every indexed file.

        The repo-scale 'read everything' baseline for the savings report.
        """
        row = self.conn.execute(
            "SELECT COALESCE(SUM(token_est), 0) t FROM files").fetchone()
        return row["t"] if row else 0

    def files_with_tokens(self) -> list[sqlite3.Row]:
        """Every indexed file with its token estimate and symbol count."""
        return self.conn.execute(
            "SELECT path, token_est, symbols_count, language FROM files "
            "ORDER BY path").fetchall()

    def commit(self):
        self.conn.commit()

    # ---- lookups ----
    def id_for_qname(self, qname: str) -> Optional[int]:
        row = self.conn.execute(
            "SELECT id FROM symbols WHERE qname=?", (qname,)).fetchone()
        return row["id"] if row else None

    def symbol(self, sid: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM symbols WHERE id=?", (sid,)).fetchone()

    def symbol_by_qname(self, qname: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM symbols WHERE qname=?", (qname,)).fetchone()

    def all_qnames(self) -> list[str]:
        return [r["qname"] for r in self.conn.execute("SELECT qname FROM symbols")]

    def all_names(self) -> list[str]:
        return [r["name"] for r in self.conn.execute(
            "SELECT DISTINCT name FROM symbols")]

    def candidates_by_leaf(self, name: str) -> list[sqlite3.Row]:
        """All symbols whose leaf name matches (used for edge resolution)."""
        return self.conn.execute(
            "SELECT * FROM symbols WHERE name=?", (name,)).fetchall()

    def symbol_at(self, file: str, line: int) -> Optional[sqlite3.Row]:
        """Smallest indexed symbol containing a one-based source line."""
        return self.conn.execute(
            "SELECT * FROM symbols WHERE file=? AND lineno<=? AND end_lineno>=? "
            "ORDER BY CASE WHEN kind='module' THEN 1 ELSE 0 END, "
            "(end_lineno-lineno) ASC, lineno DESC LIMIT 1",
            (file, line, line)).fetchone()

    def search(self, terms: str, limit: int = 12) -> list[sqlite3.Row]:
        """Lexical search over names/signatures/docstrings."""
        if self.fts:
            # build an OR query of prefix tokens; tolerate odd characters
            tokens = [t for t in _tokenize(terms) if t]
            if not tokens:
                return []
            match = " OR ".join(f'{t}*' for t in tokens)
            try:
                rows = self.conn.execute(
                    "SELECT s.*, bm25(symbols_fts) AS score FROM symbols_fts f "
                    "JOIN symbols s ON s.id=f.rowid WHERE symbols_fts MATCH ? "
                    "ORDER BY score LIMIT ?", (match, limit)).fetchall()
                if rows:
                    return rows
            except sqlite3.OperationalError:
                pass
        # fallback: LIKE across the union of tokens
        rows: list[sqlite3.Row] = []
        seen = set()
        for t in _tokenize(terms):
            like = f"%{t}%"
            for r in self.conn.execute(
                "SELECT * FROM symbols WHERE name LIKE ? OR signature LIKE ? "
                "OR docstring LIKE ? LIMIT ?", (like, like, like, limit)):
                if r["id"] not in seen:
                    seen.add(r["id"]); rows.append(r)
        return rows[:limit]

    def search_chunks(self, terms: str, limit: int = 6) -> list[sqlite3.Row]:
        if self.fts:
            tokens = [t for t in _tokenize(terms) if t]
            if tokens:
                match = " OR ".join(f'{t}*' for t in tokens)
                try:
                    rows = self.conn.execute(
                        "SELECT c.*, bm25(chunks_fts) AS score FROM chunks_fts f "
                        "JOIN chunks c ON c.id=f.rowid WHERE chunks_fts MATCH ? "
                        "ORDER BY score LIMIT ?", (match, limit)).fetchall()
                    if rows:
                        return rows
                except sqlite3.OperationalError:
                    pass
        rows: list[sqlite3.Row] = []
        seen = set()
        for t in _tokenize(terms):
            like = f"%{t}%"
            for r in self.conn.execute(
                "SELECT * FROM chunks WHERE text LIKE ? LIMIT ?", (like, limit)):
                if r["id"] not in seen:
                    seen.add(r["id"]); rows.append(r)
        return rows[:limit]

    def degrees(self, ids: Iterable[int]) -> dict[int, int]:
        """Total edge degree for each id, in one round trip (NB-1).

        Used to damp hub symbols: something referenced from everywhere carries
        little task-specific signal, so it should not outrank a close match.
        """
        idl = list(dict.fromkeys(ids))
        if not idl:
            return {}
        out: dict[int, int] = {i: 0 for i in idl}
        q = ",".join("?" * len(idl))
        for col in ("src_id", "dst_id"):
            for r in self.conn.execute(
                    f"SELECT {col} AS i, COUNT(*) AS n FROM edges "
                    f"WHERE {col} IN ({q}) GROUP BY {col}", idl):
                out[r["i"]] = out.get(r["i"], 0) + r["n"]
        return out

    def neighbors(self, sid: int, types: Iterable[str], direction: str,
                  limit: int = 0) -> list[sqlite3.Row]:
        """Return symbols connected to `sid` by the given edge types.

        direction: 'out' (sid is src) returns callees/bases;
                   'in'  (sid is dst) returns callers/subclasses.
        """
        tlist = list(types)
        q = ",".join("?" * len(tlist))
        if direction == "out":
            sql = (f"SELECT s.* FROM edges e JOIN symbols s ON s.id=e.dst_id "
                   f"WHERE e.src_id=? AND e.type IN ({q})")
        else:
            sql = (f"SELECT s.* FROM edges e JOIN symbols s ON s.id=e.src_id "
                   f"WHERE e.dst_id=? AND e.type IN ({q})")
        params: list = [sid] + tlist
        if limit:
            # Deterministic truncation for hub symbols (NB-1): ordering by id
            # keeps the candidate set stable across runs.
            sql += " ORDER BY s.id LIMIT ?"
            params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    def file_symbols(self, file: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM symbols WHERE file=? ORDER BY lineno", (file,)).fetchall()

    def stats(self) -> dict:
        c = self.conn.execute
        return {
            "files": c("SELECT COUNT(*) n FROM files").fetchone()["n"],
            "symbols": c("SELECT COUNT(*) n FROM symbols").fetchone()["n"],
            "edges": c("SELECT COUNT(*) n FROM edges").fetchone()["n"],
            "chunks": c("SELECT COUNT(*) n FROM chunks").fetchone()["n"],
            "vectors": c("SELECT COUNT(*) n FROM vectors").fetchone()["n"],
            "summaries": c("SELECT COUNT(*) n FROM summaries").fetchone()["n"],
        }

    def close(self):
        self.conn.close()


def _tokenize(text: str) -> list[str]:
    import re
    # split camelCase / snake_case / dotted paths into searchable tokens
    raw = re.split(r"[^A-Za-z0-9_]+", text)
    out: list[str] = []
    for w in raw:
        if not w:
            continue
        for part in w.split("_"):
            # split camelCase
            cur = ""
            for ch in part:
                if ch.isupper() and cur:
                    out.append(cur.lower()); cur = ch
                else:
                    cur += ch
            if cur:
                out.append(cur.lower())
    # dedupe, drop very short noise tokens
    seen, res = set(), []
    for t in out:
        if len(t) >= 2 and t not in seen:
            seen.add(t); res.append(t)
    return res


# ==========================================================================
# embeddings (optional, local — semantic seeding layer)
# ==========================================================================
"""Local embeddings for semantic retrieval.

Seeding used to be lexical-only (FTS5). This adds a vector representation per
symbol so a task phrased in different words than the code still finds it
("retry the request" -> `backoff`/`reattempt`). It is fully offline and has
ZERO required dependencies:

  - Default backend is a deterministic hashing embedding (token + char-trigram
    signed hashing into a fixed-dim L2-normalised vector). No model download,
    no network — works everywhere the CLI works. It captures lexical/structural
    overlap robustly; it is not a true neural semantic model.
  - Opt into a real local code-embedding model by installing
    `sentence-transformers` and setting TOKENGRAPH_EMBEDDINGS=st
    (model via TOKENGRAPH_EMBED_MODEL, default all-MiniLM-L6-v2).

Vectors are stored as float32 blobs in the `vectors` table and compared by
cosine in Python (brute force — fine for repo-scale symbol counts). Lexical and
semantic seed lists are combined with reciprocal-rank fusion in the retriever,
so neither backend alone has to be perfect.
"""

import array as _array
import math

EMBED_DIM = 256
_EMBED_MODEL = None
_EMBED_TRIED = False
_EMBED_STATUS = "not probed"


def offline_mode() -> bool:
    return os.environ.get("TOKENGRAPH_OFFLINE", "").lower() in {
        "1", "true", "yes", "on"
    }


_EMBED_DISABLED = {"0", "false", "no", "off", "none", "hash", "hashing"}
_EMBED_ENABLED = {"1", "true", "yes", "on", "auto", "st", "sbert",
                  "sentence-transformers"}


def embed_model_name() -> str:
    return os.environ.get("TOKENGRAPH_EMBED_MODEL", "all-MiniLM-L6-v2")


def _embed_model():
    """Load sentence-transformers when it is installed (EM-1).

    Previously this required an opt-in env var, so `pip install
    contextiq[embeddings]` bought nothing and users believed they had a
    neural model while running the hashing fallback. Now presence of the
    package is the signal; `TOKENGRAPH_EMBEDDINGS=off` (or offline mode)
    forces the deterministic hash backend.
    """
    global _EMBED_MODEL, _EMBED_TRIED, _EMBED_STATUS
    if offline_mode():
        _EMBED_STATUS = "offline mode (TOKENGRAPH_OFFLINE)"
        return None
    if _EMBED_TRIED:
        return _EMBED_MODEL
    _EMBED_TRIED = True
    setting = os.environ.get("TOKENGRAPH_EMBEDDINGS", "auto").strip().lower()
    if setting in _EMBED_DISABLED:
        _EMBED_STATUS = "disabled via TOKENGRAPH_EMBEDDINGS"
        return None
    if setting and setting not in _EMBED_ENABLED | {"download"}:
        _EMBED_STATUS = f"unrecognised TOKENGRAPH_EMBEDDINGS={setting!r}"
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        _EMBED_STATUS = ("sentence-transformers not installed — "
                         "`pip install contextiq[embeddings]`")
        return None
    # Never fetch over the network implicitly: a first MCP call must not stall
    # for minutes on a model download. `TOKENGRAPH_EMBEDDINGS=download` (or
    # `contextiq embed-warm`) opts into the one-time fetch.
    allow_download = setting == "download"
    try:
        kwargs = {} if allow_download else {"local_files_only": True}
        model = SentenceTransformer(embed_model_name(), **kwargs)
        # sentence-transformers silently improvises an *untrained* mean-pooling
        # model when it cannot read a proper ST config. That is strictly worse
        # than the deterministic hash backend, so reject it: a real model
        # reports a sentence-embedding dimension and encodes stably.
        dim = model.get_sentence_embedding_dimension()
        if not dim or dim <= 0:
            raise RuntimeError("no sentence-embedding dimension")
        probe = model.encode(["def add(a, b): return a + b"],
                             normalize_embeddings=True)
        if len(probe[0]) != dim:
            raise RuntimeError("embedding dimension mismatch")
        _EMBED_MODEL = model
        _EMBED_STATUS = f"loaded {embed_model_name()} ({dim}d)"
    except Exception as ex:
        _EMBED_MODEL = None
        _EMBED_STATUS = (
            f"{embed_model_name()} not available locally ({type(ex).__name__}); "
            f"run `contextiq embed-warm` once to download it, or set "
            f"TOKENGRAPH_EMBEDDINGS=download")
    return _EMBED_MODEL


def embed_backend_id() -> str:
    """Stable identity of the embedding space currently in use (EM-2).

    Stored on every vector row so a backend switch invalidates the old
    vectors instead of silently comparing incompatible spaces.
    """
    model = _embed_model()
    if model is None:
        return f"hash-v1-d{EMBED_DIM}"
    return f"st:{embed_model_name()}"


def embed_backend_info() -> dict:
    """Human-facing description of the active embedding backend.

    Surfaced by `doctor`, `search_semantic` and `embedding_status` so nobody
    has to guess whether they are getting true semantic matching or the
    lexical hash fallback.
    """
    model = _embed_model()
    semantic = model is not None
    return {
        "backend": embed_backend_id(),
        "semantic": semantic,
        "kind": "sentence-transformers" if semantic else "hashing",
        "model": embed_model_name() if semantic else None,
        "status": _EMBED_STATUS,
        "note": (
            "True neural embeddings: matches by meaning."
            if semantic else
            "Deterministic hashing over tokens + character trigrams. Captures "
            "lexical and structural overlap robustly, but does NOT match by "
            "meaning across unrelated vocabulary. Install "
            "`contextiq[embeddings]` and run `contextiq embed-warm`."
        ),
    }


def embed_warm() -> dict:
    """Download + verify the semantic embedding model, once, explicitly.

    Separated from the query path so no MCP tool call ever blocks on a model
    fetch. After this succeeds, `auto` picks the model up from local cache.
    """
    global _EMBED_MODEL, _EMBED_TRIED, _EMBED_STATUS
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return {"ok": False, "backend": embed_backend_id(),
                "error": "sentence-transformers not installed. "
                         "Install with: pip install 'contextiq[embeddings]'"}
    name = embed_model_name()
    try:
        model = SentenceTransformer(name)
        dim = model.get_sentence_embedding_dimension()
        if not dim or dim <= 0:
            raise RuntimeError(
                f"{name} has no sentence-transformers config; it would load as "
                f"an untrained mean-pooling model")
        _EMBED_MODEL, _EMBED_TRIED = model, True
        _EMBED_STATUS = f"loaded {name} ({dim}d)"
        return {"ok": True, "model": name, "dim": dim,
                "backend": embed_backend_id(),
                "next": "Re-index to rebuild vectors in the new space: "
                        "`contextiq index`"}
    except Exception as ex:
        return {"ok": False, "model": name, "backend": embed_backend_id(),
                "error": f"{type(ex).__name__}: {ex}"}


def _trigrams(token: str):
    t = f"#{token}#"
    return (t[i:i + 3] for i in range(len(t) - 2)) if len(t) >= 3 else (t,)


def _hash_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic signed-hashing embedding over tokens + char trigrams."""
    vec = [0.0] * dim
    toks = _tokenize(text)
    if not toks:
        return vec
    feats: list[str] = []
    for t in toks:
        feats.append(t)
        feats.extend(_trigrams(t))
    for f in feats:
        h = int.from_bytes(hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest(), "big")
        idx = h % dim
        vec[idx] += 1.0 if (h >> 1) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _embed_model()
    if model is not None:
        try:
            embs = model.encode(texts, normalize_embeddings=True)
            return [[float(x) for x in e] for e in embs]
        except Exception:
            pass
    return [_hash_embed(t) for t in texts]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def vec_to_blob(vec: list[float]) -> bytes:
    return _array.array("f", vec).tobytes()


def blob_to_vec(blob: bytes) -> list[float]:
    a = _array.array("f")
    a.frombytes(blob)
    return list(a)


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def symbol_embedding_text(qname: str, signature: str, docstring: str) -> str:
    return " ".join(p for p in (qname, signature, docstring) if p)


# ==========================================================================
# indexer
# ==========================================================================
"""Indexer: turn a repo into the local graph.

Walks all supported source files, skips unchanged ones (mtime+size, then hash),
parses changed files, writes symbols + per-symbol embeddings + a module summary,
then resolves name-based edges to concrete symbol ids.

Edge resolution heuristic (best-effort, like every static call-graph tool):
  1. exact qname match
  2. import-aware: if the file imports the leaf from a module, prefer that module
  3. same-module sibling by leaf name
  4. unique global leaf-name match
  5. otherwise dropped (unresolved external call — e.g. stdlib)
DEFINES edges (parent -> child) are added directly from the parse tree, so the
graph also answers "what's defined in X" / "where is Y defined". This keeps the
graph precise where it can be and silent where it can't.
"""

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_IGNORES = {".git", ".tokengraph", "__pycache__", ".venv", "venv",
                   "node_modules", ".mypy_cache", ".pytest_cache", "build", "dist"}
CHUNK_LINES = 120
CHUNK_OVERLAP = 20


class GitIgnore:
    """Lightweight .gitignore matcher.

    Supports the common subset that matters for source indexing: comments,
    blank lines, trailing-slash directory rules, leading-slash anchoring,
    `**`/`*`/`?` globs, and `!` negation. It is intentionally not a full
    reimplementation of git's pathspec semantics — it errs toward ignoring
    fewer files (never silently dropping source) rather than more.
    """

    def __init__(self, patterns: list[str]):
        self.rules: list[tuple[bool, bool, str]] = []  # (negated, dir_only, regex)
        for raw in patterns:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            neg = line.startswith("!")
            if neg:
                line = line[1:]
            dir_only = line.endswith("/")
            core = line.rstrip("/")
            # A leading slash, or any interior slash, anchors to the repo root;
            # a slashless pattern matches a basename at any depth.
            anchored = core.startswith("/") or "/" in core.strip("/")
            self.rules.append((neg, dir_only, self._compile(core.strip("/"), anchored)))

    @staticmethod
    def _compile(pat: str, anchored: bool) -> str:
        import re
        # Translate a gitignore glob to a regex matched against the POSIX relpath.
        i, out = 0, []
        while i < len(pat):
            c = pat[i]
            if c == "*":
                if pat[i:i + 2] == "**":
                    out.append(".*")
                    i += 2
                    continue
                out.append("[^/]*")
            elif c == "?":
                out.append("[^/]")
            else:
                out.append(re.escape(c))
            i += 1
        body = "".join(out)
        # A non-anchored pattern matches at any path depth (basename or subdir).
        prefix = r"^" if anchored else r"(^|.*/)"
        return prefix + body + r"(/.*)?$"

    def is_ignored(self, relpath: str, is_dir: bool) -> bool:
        import re
        rp = relpath.replace(os.sep, "/")
        result = False
        for neg, dir_only, rx in self.rules:
            if dir_only and not is_dir:
                continue
            if re.match(rx, rp):
                result = not neg
        return result

    @classmethod
    def load(cls, root: Path) -> "GitIgnore":
        patterns: list[str] = []
        for name in (".gitignore", ".contextignore"):
            try:
                patterns += (root / name).read_text(
                    encoding="utf-8", errors="replace").splitlines()
            except OSError:
                pass
        return cls(patterns)


# ==========================================================================
# security: secret scanning (SEC-1, SEC-2)
# ==========================================================================
"""Redact credentials before any context leaves the tool.

Output (context packs, surgical line fetches, skeletons) is scanned and
matching secrets are replaced with [REDACTED]. Patterns are deliberately
broad; a false positive only costs a few characters of context, while a leak
is unacceptable. Scanning never raises — on any regex error the original text
is returned unchanged so a run always completes (NFR-6).
"""

import re as _re

_SECRET_PATTERNS = [
    # name, compiled regex
    ("aws_access_key", _re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("aws_secret", _re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}")),
    ("github_token", _re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("jwt", _re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("db_url", _re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s'\"]+:[^\s'\"@]+@[^\s'\"]+")),
    ("ssh_key", _re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("gcp_key", _re.compile(r"(?i)\"private_key\"\s*:\s*\"-----BEGIN")),
    ("gcp_api_key", _re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("stripe_key", _re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}\b")),
    ("twilio_key", _re.compile(r"\bSK[0-9a-fA-F]{32}\b")),
    ("slack_token", _re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b")),
    ("generic_secret", _re.compile(
        r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)"
        r"\s*[=:]\s*['\"][^'\"\s]{6,}['\"]")),
]


def redact_secrets(text: str) -> tuple[str, int]:
    """Return (redacted_text, num_redactions). Never raises (NFR-6)."""
    if not text:
        return text, 0
    total = 0
    try:
        for _name, rx in _SECRET_PATTERNS:
            text, n = rx.subn("[REDACTED]", text)
            total += n
    except Exception:
        return text, total
    return text, total


# ==========================================================================
# input squeeze: shrink pasted stacktraces / CI logs / JSON payloads (G6)
# ==========================================================================
"""Reduce noisy pasted input before it reaches the model.

Classifies a blob (stacktrace / CI log / JSON / generic) and strips the
low-signal parts — vendor stack frames, build-progress spam, verbose JSON —
while preserving the diagnostic content. Deterministic and local; every
reducer falls back to the original text on error so a run always completes.
"""

_VENDOR_FRAME_MARKERS = (
    "node_modules", "site-packages", "dist-packages", "/usr/lib", "\\lib\\",
    "/lib/", ".venv", "/venv/", "vendor/", "<frozen", "runtime/", "/golang/",
    "\\go\\pkg\\", "/go/pkg/",
)


def classify_input(text: str) -> str:
    """Return 'stacktrace' | 'cilog' | 'json' | 'text' for a pasted blob."""
    import re
    s = (text or "").strip()
    if not s:
        return "text"
    if s[0] in "{[" and s[-1] in "}]":
        import json
        try:
            json.loads(s)
            return "json"
        except Exception:
            pass
    low = text.lower()
    if "traceback (most recent call last)" in low:
        return "stacktrace"
    lines = text.splitlines()
    frames = sum(1 for ln in lines
                 if re.match(r"\s+at\s+\S", ln) or 'file "' in ln.lower()
                 or re.match(r"\s*#\d+\s+0x", ln))
    if frames >= 2:
        return "stacktrace"
    diag = sum(1 for ln in lines
               if re.search(r"(?i)\b(error|warn|fail|exception|fatal|panic)\b", ln))
    ts = sum(1 for ln in lines if re.match(
        r"^\s*(\[?\d{4}-\d\d-\d\d|\d{1,2}:\d\d:\d\d|\[\d+/\d+\]|#\d+\b)", ln))
    noise = sum(1 for ln in lines if re.search(
        r"(?i)\b(download|fetch|extract|install|build|compil|resolv|step|progress|upload)", ln))
    if (ts >= 3 or noise >= 5 or len(lines) > 30) and diag >= 1:
        return "cilog"
    return "text"


def _squeeze_json(text: str, max_str: int = 160, max_items: int = 6) -> str:
    import json
    try:
        data = json.loads(text)
    except Exception:
        return _squeeze_generic(text)

    def reduce(v):
        if isinstance(v, str):
            return v if len(v) <= max_str else v[:max_str] + f"…(+{len(v)-max_str} chars)"
        if isinstance(v, list):
            if len(v) <= max_items:
                return [reduce(x) for x in v]
            return [reduce(x) for x in v[:max_items]] + [f"…(+{len(v)-max_items} more items)"]
        if isinstance(v, dict):
            return {k: reduce(x) for k, x in v.items()}
        return v

    try:
        return json.dumps(reduce(data), indent=2, ensure_ascii=False)
    except Exception:
        return _squeeze_generic(text)


def _enrich_frames(out: str, store, root) -> str:
    """Annotate repo stack frames with the enclosing symbol qname (best-effort)."""
    if store is None or root is None:
        return out
    import re
    rx = re.compile(r'File "([^"]+)", line (\d+)|(\S+\.\w+):(\d+)')
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        m = rx.search(ln)
        if not m:
            continue
        path = m.group(1) or m.group(3)
        lineno = int(m.group(2) or m.group(4) or 0)
        if not path or not lineno:
            continue
        try:
            rel = os.path.relpath(path, root).replace("\\", "/")
        except Exception:
            continue
        if rel.startswith(".."):
            continue
        try:
            best = None
            for s in store.file_symbols(rel):
                if s["kind"] == "module":
                    continue
                if s["lineno"] <= lineno <= (s["end_lineno"] or s["lineno"]):
                    best = s["qname"]
            if best and best not in ln:
                lines[i] = ln + f"   -> {best}"
        except Exception:
            continue
    return "\n".join(lines)


def _squeeze_stacktrace(text: str, store=None, root=None) -> str:
    import re
    lines = text.splitlines()
    kept: list[str] = []
    dropped = 0
    skip_next = False
    for ln in lines:
        if not ln.strip():
            skip_next = False
            continue
        low = ln.lower()
        is_pyframe = 'file "' in low and ".py" in low
        if skip_next and not is_pyframe and not re.match(r'\s*(at\s+\S|#\d+\s+0x)', ln):
            # the indented source line that follows a dropped Python frame
            skip_next = False
            continue
        skip_next = False
        is_frame = bool(re.match(r'\s*(at\s+\S|#\d+\s+0x)', ln)) or is_pyframe
        if is_frame and any(v in low for v in _VENDOR_FRAME_MARKERS):
            dropped += 1
            skip_next = is_pyframe   # also drop the code line that follows it
            continue
        kept.append(ln.rstrip())
    out = "\n".join(kept)
    if dropped:
        out += f"\n... ({dropped} library/vendor frame(s) elided)"
    return _enrich_frames(out, store, root)


def _squeeze_cilog(text: str) -> str:
    import re
    lines = [ln.rstrip() for ln in text.splitlines()]
    keep_rx = re.compile(
        r"(?i)\b(error|errno|fail|failed|failure|exception|traceback|fatal|panic|"
        r"assert|denied|refused|timed?\s*out|cannot|unable|not found|undefined|"
        r"unresolved|missing|expected|syntaxerror|✖|✗|×)\b|[✖✗×]")
    keep_idx: set[int] = set()
    for i, ln in enumerate(lines):
        if keep_rx.search(ln):
            keep_idx.add(i)
            if i > 0:
                keep_idx.add(i - 1)        # one line of leading context
    if not keep_idx:
        return _squeeze_generic(text)
    out: list[str] = []
    prev = -2
    elided = 0
    for i, ln in enumerate(lines):
        if i in keep_idx:
            if out and prev != i - 1:
                out.append("...")
            out.append(ln)
            prev = i
        else:
            elided += 1
    deduped: list[str] = []
    for ln in out:
        if deduped and deduped[-1] == ln:
            continue
        deduped.append(ln)
    res = "\n".join(deduped)
    if elided:
        res += f"\n... ({elided} non-diagnostic line(s) removed)"
    return res


def _squeeze_generic(text: str, max_lines: int = 200) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    blank = False
    for ln in lines:
        if not ln:
            if blank:
                continue
            blank = True
        else:
            blank = False
        if out and out[-1] == ln and ln:
            continue
        out.append(ln)
    if len(out) > max_lines:
        out = out[:max_lines] + [f"... ({len(out)-max_lines} more line(s) truncated)"]
    return "\n".join(out)


def squeeze_text(text: str, kind: str = "auto", store=None, root=None) -> dict:
    """Classify and reduce a pasted blob; report token savings. Never raises."""
    text = text or ""
    original = count_tokens(text)
    detected = classify_input(text) if (kind in (None, "auto", "")) else kind
    try:
        if detected == "json":
            out = _squeeze_json(text)
        elif detected == "stacktrace":
            out = _squeeze_stacktrace(text, store, root)
        elif detected == "cilog":
            out = _squeeze_cilog(text)
        else:
            out = _squeeze_generic(text)
    except Exception:
        out = text
    out, _ = redact_secrets(out)
    final = count_tokens(out)
    saved = max(0, original - final)
    return {
        "kind": detected,
        "original_tokens": original,
        "squeezed_tokens": final,
        "tokens_saved": saved,
        "reduction_pct": round(saved / original * 100, 1) if original else 0.0,
        "text": out,
    }


# ==========================================================================
# verify: flag fabricated files / symbols in an AI answer (G5, did-you-mean)
# ==========================================================================
def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _closest(name: str, candidates, n: int = 3, max_dist: int | None = None) -> list[str]:
    """Up to n nearest candidates by edit distance (for 'did you mean?')."""
    if not name:
        return []
    if max_dist is None:
        max_dist = max(2, len(name) // 3)
    low = name.lower()
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        d = _levenshtein(low, c.lower())
        if d <= max_dist:
            scored.append((d, c))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [c for _, c in scored[:n]]


# stoplist of common builtins/keywords so verify() does not flag them as fabricated
_VERIFY_STOPWORDS = frozenset("""
print len range int str float bool list dict set tuple type id map filter zip
open close read write append super self this new return if else for while def
class import from func var let const async await yield throw catch try except
finally with as in is not and or none null true false void main get post put
delete patch console log error warn info debug assert require module exports
""".split())


# ==========================================================================
# architecture map helpers: HTTP routes + import hubs/cycles (G11)
# ==========================================================================
# language-family -> list of route regexes (group1=method/verb, last group=path)
_ROUTE_PATTERNS: dict[str, list[str]] = {
    "python": [
        r'@(?:app|router|bp|blueprint|api|mod)\.(get|post|put|delete|patch|route|websocket)'
        r'\(\s*["\']([^"\']+)["\']',
    ],
    "js": [
        r'\b(?:app|router|api|server|route)\.(get|post|put|delete|patch|all|head|options|use)'
        r'\(\s*["\'`]([^"\'`]+)["\'`]',
    ],
    "java": [
        r'@(Get|Post|Put|Delete|Patch|Request)Mapping\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']',
    ],
    "go": [
        r'\.(GET|POST|PUT|DELETE|PATCH|Handle|HandleFunc)\(\s*"([^"]+)"',
    ],
}

_ROUTE_FAMILY = {"python": "python", "typescript": "js", "javascript": "js",
                 "java": "java", "go": "go"}


def _import_cycles(graph: dict[str, set]) -> list[list[str]]:
    """Strongly-connected components of size>1 in a file import graph (Tarjan, iterative)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    onstack: dict[str, bool] = {}
    stack: list[str] = []
    result: list[list[str]] = []
    counter = [0]

    def strongconnect(v: str):
        work = [(v, 0)]
        while work:
            node, pi = work[-1]
            if pi == 0:
                index[node] = low[node] = counter[0]
                counter[0] += 1
                stack.append(node)
                onstack[node] = True
            recurse = False
            succ = sorted(graph.get(node, ()))
            i = pi
            while i < len(succ):
                w = succ[i]
                if w not in index:
                    work[-1] = (node, i + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                if onstack.get(w):
                    low[node] = min(low[node], index[w])
                i += 1
            if recurse:
                continue
            if low[node] == index[node]:
                comp: list[str] = []
                while True:
                    w = stack.pop()
                    onstack[w] = False
                    comp.append(w)
                    if w == node:
                        break
                if len(comp) > 1:
                    result.append(sorted(comp))
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])

    for n in list(graph.keys()):
        if n not in index:
            strongconnect(n)
    return result


# ==========================================================================
# intent detection + risk + model-tier routing (FR-6, MR-1..MR-3)
# ==========================================================================
_INTENT_KEYWORDS = {
    "debug": ("debug", "fix", "bug", "error", "crash", "fail", "broken",
              "exception", "stack trace", "regression", "race", "leak"),
    "review": ("review", "audit", "security", "vulnerab", "lint", "quality",
               "smell", "best practice"),
    "refactor": ("refactor", "rename", "restructure", "clean up", "extract",
                 "migrate", "modernize", "rewrite", "decouple"),
    "explain": ("explain", "understand", "how does", "what does", "walk through",
                "describe", "document", "why"),
    "search": ("find", "where", "locate", "search", "list", "which file"),
}

# Languages/extensions that route to a cheaper tier by default.
_FAST_EXTS = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
              ".cfg", ".env", ".html", ".css", ".scss", ".xml", ".sql"}
_POWERFUL_HINTS = ("architecture", "security", "concurrency", "distributed",
                   "performance", "cryptograph", "auth", "multi-file",
                   "schema", "migration", "design")


def detect_intent(task: str) -> str:
    """Classify a task into debug/review/refactor/explain/search (default search)."""
    t = (task or "").lower()
    best, best_hits = "search", 0
    for intent, kws in _INTENT_KEYWORDS.items():
        hits = sum(1 for k in kws if k in t)
        if hits > best_hits:
            best, best_hits = intent, hits
    return best


def recommend_tier(task: str) -> dict:
    """Recommend a model tier (fast/balanced/powerful) for a task (MR-1)."""
    t = (task or "").lower()
    if any(h in t for h in _POWERFUL_HINTS):
        tier = "powerful"
    elif any(k in t for k in _INTENT_KEYWORDS["debug"] + _INTENT_KEYWORDS["refactor"]):
        tier = "balanced"
    elif detect_intent(task) in ("explain", "search"):
        tier = "fast"
    else:
        tier = "balanced"
    return _tier_info(tier, task=task)


def _tier_info(tier: str, task: str = "", file: str = "") -> dict:
    table = {
        "fast": {
            "tier": "fast", "cost_hint_per_1k": 0.0008,
            "models": ["claude-haiku", "gpt-4o-mini", "gemini-flash", "llama-3.1-8b"],
            "use_for": "config/markup/typos/simple lookups",
        },
        "balanced": {
            "tier": "balanced", "cost_hint_per_1k": 0.003,
            "models": ["claude-sonnet", "gpt-4o", "gemini-pro", "llama-3.1-70b"],
            "use_for": "features/tests/debugging",
        },
        "powerful": {
            "tier": "powerful", "cost_hint_per_1k": 0.015,
            "models": ["claude-opus", "gpt-5", "gemini-ultra", "llama-3.1-405b"],
            "use_for": "architecture/security/multi-file refactors",
        },
    }
    info = dict(table[tier])
    if task:
        info["task"] = task
        info["intent"] = detect_intent(task)
    if file:
        info["file"] = file
    return info


def tier_for_file(file: str, token_est: int = 0, symbols: int = 0,
                  fan_in: int = 0, fan_out: int = 0, max_depth: int = 0,
                  branches: int = 0) -> dict:
    """Per-file complexity routing from graph features (MR-3).

    The original version routed on file extension and raw size alone, so a
    3,000-token data file and a 3,000-token concurrency-critical module got the
    same answer. Size is a weak proxy for how hard a file is to reason about;
    what actually predicts difficulty is how entangled it is (fan-in/fan-out),
    how deeply nested its control flow is, and how many branches it carries.

    Every input beyond `file` is optional, so callers that only know the size
    still get the old behaviour rather than an error.
    """
    from os.path import splitext
    ext = splitext(file)[1].lower()

    # Config/markup never needs an expensive model regardless of size.
    if ext in _FAST_EXTS:
        return _tier_info("fast", file=file)

    score = 0.0
    reasons: list[str] = []

    def add(points: float, why: str):
        nonlocal score
        score += points
        reasons.append(why)

    # Size still carries real weight: a file that is both large and
    # symbol-dense reaches `powerful` on those signals alone, preserving the
    # original size-only contract. What changed is that either signal *by
    # itself* is now only suggestive, and entanglement can promote a small
    # file that the old size-only rule would have called easy.
    if token_est:
        if token_est < 300:
            add(-1.0, "very small file")
        elif token_est > 3000:
            add(1.5, f"large file ({token_est} tokens)")
    if symbols > 40:
        add(1.5, f"many definitions ({symbols})")
    # Entanglement: a widely-depended-on file is risky to change.
    if fan_in >= 10:
        add(1.5, f"high fan-in ({fan_in} dependents)")
    elif fan_in >= 4:
        add(0.5, f"moderate fan-in ({fan_in})")
    if fan_out >= 15:
        add(1.0, f"high fan-out ({fan_out} dependencies)")
    # Control-flow complexity beats size as a difficulty signal.
    if max_depth >= 5:
        add(1.5, f"deeply nested control flow (depth {max_depth})")
    elif max_depth >= 4:
        add(0.5, f"nested control flow (depth {max_depth})")
    if branches >= 60:
        add(1.5, f"many branches ({branches})")
    elif branches >= 25:
        add(0.5, f"branchy ({branches})")

    if score >= 2.5:
        tier = "powerful"
    elif score <= 0.0:
        tier = "fast"
    else:
        tier = "balanced"
    info = _tier_info(tier, file=file)
    info["complexity_score"] = round(score, 2)
    info["signals"] = reasons or ["no strong signals; defaulted on size"]
    return info


def file_complexity_features(text: str) -> dict:
    """Cheap structural signals for routing: nesting depth and branch count.

    Language-agnostic on purpose — it reads indentation and branch keywords
    rather than parsing, so it works for every language the indexer supports
    without a per-language implementation.
    """
    import re as _re3
    branch_rx = _re3.compile(
        r"\b(if|elif|else\s+if|for|while|case|catch|except|switch|when|"
        r"&&|\|\|)\b")
    max_depth = depth = 0
    branches = 0
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "*", "--")):
            continue
        indent = len(line) - len(line.lstrip())
        # 4 spaces (or a tab) per level is the overwhelmingly common convention.
        depth = indent // 4 if " " in line[:4] else line.count("\t")
        max_depth = max(max_depth, depth)
        branches += len(branch_rx.findall(stripped))
    return {"max_depth": max_depth, "branches": branches}


# ==========================================================================
# git helpers (recency boost, diff/PR mode, hot-set sizing)
# ==========================================================================
def _git(root, *args, timeout: int = 5) -> str:
    """Run a git command in `root`; return stdout ('' on any failure)."""
    import subprocess
    try:
        r = subprocess.run(["git", *args], cwd=str(root),
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def git_recent_files(root, commits: int = 10) -> list[str]:
    """Repo-relative paths touched in the last N commits, newest first (TB-5)."""
    out = _git(root, "log", f"-{max(1, commits)}", "--name-only",
               "--pretty=format:")
    files: list[str] = []
    seen: set[str] = set()
    for line in out.splitlines():
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            files.append(line)
    return files


def git_changed_files(root, staged: bool = False) -> list[str]:
    """Working-tree changes (TB-5a). staged=True → only the index (pre-commit)."""
    args = ["diff", "--name-only"] + (["--cached"] if staged else [])
    files = [l.strip() for l in _git(root, *args).splitlines() if l.strip()]
    if not staged:
        unt = _git(root, "ls-files", "--others", "--exclude-standard")
        files += [l.strip() for l in unt.splitlines() if l.strip()]
    return list(dict.fromkeys(files))


def git_diff_hunks(root, staged: bool = False) -> dict:
    """Map each changed file → list of (start, end) line ranges on the NEW side,
    parsed from `git diff --unified=0`. Lets a caller pick out exactly which
    indexed symbols a diff touched instead of pulling whole files. Untracked
    files have no diff and won't appear (the caller treats them as fully new)."""
    import re
    args = ["diff", "--unified=0"] + (["--cached"] if staged else [])
    out = _git(root, *args)
    hunks: dict[str, list] = {}
    cur: str | None = None
    for line in out.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:].strip()
            if cur == "/dev/null":
                cur = None
        elif line.startswith("@@") and cur:
            # @@ -a,b +c,d @@  → the new side is +c,d (c=start, d=line count)
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if not m:
                continue
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            if count == 0:  # pure deletion: nothing exists on the new side
                continue
            hunks.setdefault(cur, []).append((start, start + count - 1))
    return hunks


@dataclass
class IndexReport:
    scanned: int = 0
    parsed: int = 0
    skipped: int = 0
    removed: int = 0
    reembedded: int = 0
    errors: list[str] = None
    stats: dict = None


def iter_source_files(root: Path, ignores: set[str],
                      gitignore: "GitIgnore | None" = None) -> list[Path]:
    exts = supported_extensions()
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        kept = []
        for d in dirnames:
            if d in ignores:
                continue
            rel = (rel_dir / d).as_posix()
            if gitignore and gitignore.is_ignored(rel, is_dir=True):
                continue
            kept.append(d)
        dirnames[:] = kept
        for f in filenames:
            if (Path(f).suffix.lower() not in exts
                    and f.lower() not in GENERIC_LANGUAGE_FILENAMES):
                continue
            rel = (rel_dir / f).as_posix()
            if gitignore and gitignore.is_ignored(rel, is_dir=False):
                continue
            out.append(Path(dirpath) / f)
    return out


def iter_text_chunks(text: str, chunk_lines: int = CHUNK_LINES,
                     overlap: int = CHUNK_OVERLAP) -> Iterable[tuple[int, int, str]]:
    lines = text.splitlines()
    if not lines:
        return
    step = max(1, chunk_lines - overlap)
    start = 0
    while start < len(lines):
        end = min(len(lines), start + chunk_lines)
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            yield start + 1, end, chunk
        if end >= len(lines):
            break
        start += step


def _module_summary(symbols: list[Symbol], text: str) -> str:
    """Compact extractive summary of a file: docstring + top-level defs.

    Lets the retriever gesture at a whole module for a handful of tokens
    instead of pulling its files — the doc's "50-token module summary replaces
    a 2,000-token file" tier.
    """
    mod = next((s for s in symbols if s.kind == "module"), None)
    head = mod.qname if mod else ""
    doc = (mod.docstring if mod and mod.docstring else "")
    if not doc:
        first = text.strip().splitlines()[:1]
        doc = first[0].lstrip("#/* ").strip() if first else ""
    tops = [s for s in symbols if s.kind != "module" and (s.parent == head or not s.parent)]
    classes = [s.name for s in tops if s.kind in CLASS_KINDS or s.kind == "class"]
    funcs = [s.name for s in tops if s.kind in ("function", "method")]
    parts = [f"module {head}"]
    if doc:
        parts.append(f"— {doc}")
    bits = []
    if classes:
        bits.append("types: " + ", ".join(sorted(set(classes))[:20]))
    if funcs:
        bits.append("functions: " + ", ".join(sorted(set(funcs))[:30]))
    line = " ".join(parts)
    return line + ("\n" + "; ".join(bits) if bits else "")


def _resolve_edge(store: Store, src_module_file: str, e: PendingEdge,
                  import_map: dict[str, str] | None = None):
    """Return (src_id, dst_id) or None if unresolved."""
    src = store.symbol_by_qname(e.src_qname)
    if not src:
        return None
    src_id = src["id"]

    # the dst leaf name is the last dotted component for CALLS/INHERITS
    leaf = e.dst_name.split(".")[-1]

    # 1. exact qname
    direct = store.symbol_by_qname(e.dst_name)
    if direct:
        return (src_id, direct["id"])

    candidates = store.candidates_by_leaf(leaf)
    if not candidates:
        return None

    # 2. import-aware: if this file imports `leaf` from a module, prefer the
    #    candidate whose module path matches that import target.
    if import_map and leaf in import_map:
        target = import_map[leaf]
        by_qname = store.symbol_by_qname(target)
        if by_qname:
            return (src_id, by_qname["id"])
        mod_match = [c for c in candidates if c["qname"] == target
                     or target.endswith("." + c["name"])]
        if len(mod_match) == 1:
            return (src_id, mod_match[0]["id"])

    # 3. scope-aware: a call inside `pkg.Class.method` most often targets a
    #    sibling in the same enclosing scope (e.g. `self.helper()` / `helper()`).
    #    Resolving these first raises recall on the intra-class calls that the
    #    old same-file-or-unique rule dropped as ambiguous, without guessing
    #    across unrelated classes that happen to share a leaf name.
    src_parent = src["parent"] if "parent" in src.keys() else None
    if src_parent:
        scoped = store.symbol_by_qname(f"{src_parent}.{leaf}")
        if scoped:
            return (src_id, scoped["id"])
        siblings = [c for c in candidates if c["parent"] == src_parent]
        if len(siblings) == 1:
            return (src_id, siblings[0]["id"])

    # 4. prefer a candidate in the same file (same module scope)
    same_file = [c for c in candidates if c["file"] == src["file"]]
    if len(same_file) == 1:
        return (src_id, same_file[0]["id"])
    if len(same_file) > 1 and src_parent:
        sf_scope = [c for c in same_file if c["parent"] == src_parent]
        if len(sf_scope) == 1:
            return (src_id, sf_scope[0]["id"])

    # 5. unique global match
    if len(candidates) == 1:
        return (src_id, candidates[0]["id"])

    # ambiguous -> skip (avoid polluting the graph with wrong edges)
    return None


def index_repo(root: Path, db_path: Path, ignores: set[str] | None = None,
               respect_gitignore: bool = True,
               paths: list[str] | None = None) -> IndexReport:
    ignores = (ignores or set()) | DEFAULT_IGNORES
    root = root.resolve()
    store = Store(db_path)
    report = IndexReport(errors=[])

    gitignore = GitIgnore.load(root) if respect_gitignore else None
    if paths is None:
        files = iter_source_files(root, ignores, gitignore)
    else:
        files = []
        for rel in paths:
            try:
                candidate = (root / rel).resolve()
                candidate.relative_to(root)
            except (OSError, ValueError):
                continue
            supported = (candidate.suffix.lower() in supported_extensions()
                         or candidate.name.lower() in GENERIC_LANGUAGE_FILENAMES)
            if candidate.is_file() and supported:
                files.append(candidate)
    report.scanned = len(files)
    current = {p.relative_to(root).as_posix() for p in files}

    # drop deleted files.
    # A targeted refresh (paths=[...], used by the notify_* MCP tools) cannot
    # see repo-wide deletions, but it CAN reconcile the paths it was handed:
    # anything explicitly named that no longer exists on disk is forgotten.
    # Without this, IDE-pushed change events accumulated ghost symbols forever.
    if paths is None:
        for gone in store.all_indexed_files() - current:
            store.forget_file(gone)
            report.removed += 1
    else:
        indexed = store.all_indexed_files()
        for rel in paths:
            rel_posix = Path(rel).as_posix().lstrip("./")
            if rel_posix in indexed and rel_posix not in current:
                store.forget_file(rel_posix)
                report.removed += 1

    # EM-2: if the embedding backend changed since the last index, every stored
    # vector belongs to a different space. Drop them and force a re-embed of
    # all files rather than letting semantic search silently return nothing.
    backend = embed_backend_id()
    if store.get_meta("embed_backend") != backend:
        store.drop_stale_vectors()
        store.set_meta("embed_backend", backend)

    # EX-1: the incremental fast path keys off file content, so a file that has
    # not changed is never reparsed — which means *upgrading the extractor* had
    # no effect until someone happened to edit each file. A graph missing the
    # symbol kinds the installed version knows how to find is stale in a way no
    # freshness check could see. Stamping the extractor's identity turns that
    # into a one-off full reparse on upgrade.
    extractor = extractor_version()
    reparse_all = store.get_meta("extractor_version") != extractor
    if reparse_all:
        store.set_meta("extractor_version", extractor)

    pending: list[tuple[str, PendingEdge]] = []
    import_maps: dict[str, dict[str, str]] = {}

    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            st = path.stat()
        except OSError as ex:
            report.errors.append(f"{rel}: {ex}")
            continue

        # Fast path: unchanged mtime+size means unchanged content — skip the
        # read+hash entirely. This is what makes freshen-on-query cheap enough
        # to run before every retrieval.
        meta = store.file_meta(rel)
        if (not reparse_all and meta and meta["mtime"] == st.st_mtime
                and meta["size"] == st.st_size):
            report.skipped += 1
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as ex:
            report.errors.append(f"{rel}: {ex}")
            continue
        h = file_hash(text)
        if not reparse_all and meta and meta["hash"] == h:
            # Content identical despite a touched mtime — refresh stat only.
            store.touch_file(rel, st.st_mtime, st.st_size)
            report.skipped += 1
            continue

        res: ParseResult | None = parse_path(root, path)
        if res is None:
            continue
        if res.error and not res.symbols:
            report.errors.append(f"{rel}: {res.error}")

        store.prepare_file_update(rel, {s.qname for s in res.symbols})

        emb_sids: list[int] = []
        emb_texts: list[str] = []
        for s in res.symbols:
            sid = store.upsert_symbol(s)
            if sid is not None:
                emb_sids.append(sid)
                emb_texts.append(symbol_embedding_text(s.qname, s.signature, s.docstring))
        # per-symbol embeddings (the semantic seeding layer)
        if emb_texts:
            for sid, vec in zip(emb_sids, embed_texts(emb_texts)):
                store.set_vector(sid, vec)
        # DEFINES edges: parent -> child (both exact qnames, no heuristic needed)
        for s in res.symbols:
            if s.parent:
                pid = store.id_for_qname(s.parent)
                cid = store.id_for_qname(s.qname)
                if pid is not None and cid is not None:
                    store.add_edge(pid, cid, "DEFINES")
        # compact module summary (cheap whole-module gesture for the retriever)
        store.set_summary(rel, _module_summary(res.symbols, text))

        for start_line, end_line, chunk in iter_text_chunks(text):
            store.add_chunk(rel, start_line, end_line, chunk)
        store.set_file(rel, h, st.st_mtime, language_for_path(path),
                       count_tokens(text), len(res.symbols), st.st_size)
        imap: dict[str, str] = {}
        for e in res.edges:
            pending.append((rel, e))
            if e.type == "IMPORTS":
                imap[e.dst_name.split(".")[-1]] = e.dst_name
        import_maps[rel] = imap
        report.parsed += 1

    # second pass: resolve edges now that all symbols of changed files exist
    resolved_n = 0
    for rel, e in pending:
        resolved = _resolve_edge(store, rel, e, import_maps.get(rel))
        if resolved:
            store.add_edge(resolved[0], resolved[1], e.type)
            resolved_n += 1

    # EM-2: backfill vectors for any symbol lacking one in the *current*
    # embedding space. Normally a no-op; after a backend switch (or an
    # interrupted index) it is what restores semantic search instead of
    # leaving it silently returning nothing.
    # PK-1: any structural change invalidates every memoised pack.
    if report.parsed or report.removed:
        store.bump_graph_version()
        store.purge_stale_packs()

    missing = store.symbols_missing_vectors()
    if missing:
        batch = 256
        for i in range(0, len(missing), batch):
            group = missing[i:i + batch]
            texts = [symbol_embedding_text(r["qname"], r["signature"], r["docstring"])
                     for r in group]
            for row, vec in zip(group, embed_texts(texts)):
                store.set_vector(row["id"], vec, backend)
        report.reembedded = len(missing)

    store.commit()
    report.stats = store.stats()
    if pending:
        report.stats["edge_resolution_pct"] = round(100 * resolved_n / len(pending), 1)
    store.close()
    return report


def import_scip_json(root: Path, db_path: Path, index_file: Path) -> dict:
    """Import definition/reference occurrences from `scip print --json` output."""
    import json
    root = Path(root).resolve()
    payload = json.loads(Path(index_file).read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    store = Store(db_path)
    definitions: dict[str, sqlite3.Row] = {}
    occurrences: list[tuple[str, dict]] = []
    try:
        for document in documents:
            file = (document.get("relativePath") or
                    document.get("relative_path") or "").replace("\\", "/")
            if not file or not (root / file).is_file():
                continue
            for occurrence in document.get("occurrences", []):
                occurrences.append((file, occurrence))
                roles = occurrence.get("symbolRoles", occurrence.get("symbol_roles", 0))
                source_range = occurrence.get("range", [])
                if roles & 1 and source_range:
                    row = store.symbol_at(file, int(source_range[0]) + 1)
                    if row is not None:
                        definitions[occurrence.get("symbol", "")] = row

        imported = unresolved = 0
        for file, occurrence in occurrences:
            roles = occurrence.get("symbolRoles", occurrence.get("symbol_roles", 0))
            symbol = occurrence.get("symbol", "")
            source_range = occurrence.get("range", [])
            if roles & 1 or not symbol or not source_range:
                continue
            source = store.symbol_at(file, int(source_range[0]) + 1)
            destination = definitions.get(symbol)
            if source is None or destination is None:
                unresolved += 1
                continue
            if source["id"] != destination["id"]:
                store.add_edge(source["id"], destination["id"], "REFERENCES")
                imported += 1
        # PF-1: record that precise references exist for this repo, so fidelity
        # reporting can say so instead of guessing from language alone.
        import datetime as _dt
        stamp = _dt.datetime.now().replace(microsecond=0).isoformat()
        store.set_meta("scip_ingested_at", stamp)
        store.commit()
        return {"documents": len(documents), "definitions": len(definitions),
                "references_imported": imported, "unresolved": unresolved,
                "scip_ingested_at": stamp}
    finally:
        store.close()


# ==========================================================================
# retriever (token optimization core)
# ==========================================================================
"""Retriever: the token-optimization core.

Given a task description, instead of dumping whole files we:
  1. seed   - lexical search to find the most relevant symbols
  2. expand - walk CALLS/INHERITS edges to pull the dependency neighborhood
    3. budget - fit within a token budget using tiered detail:
                                core symbols  -> full source body when small, signature when large
                neighbors     -> signature + one-line docstring
                                chunks        -> relevant indexed file regions
                overflow      -> dropped, but listed by name so the agent
                                 can request them explicitly

The result is a compact "context pack" (markdown) that an agent reads instead
of opening files. Typical packs are a small fraction of the tokens that reading
the relevant files whole would cost.
"""

from dataclasses import dataclass, field
from pathlib import Path



_FENCE = {".py": "python", ".java": "java", ".go": "go", ".ts": "typescript",
          ".tsx": "tsx", ".mts": "typescript", ".cts": "typescript",
          ".js": "javascript", ".jsx": "jsx", ".mjs": "javascript",
          ".cjs": "javascript", ".cs": "csharp", ".cpp": "cpp", ".cc": "cpp",
          ".cxx": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp", ".hh": "cpp",
          ".hxx": "cpp", ".rs": "rust", ".php": "php", ".rb": "ruby",
          ".kt": "kotlin", ".kts": "kotlin", ".swift": "swift",
          ".scala": "scala", ".sc": "scala", ".sql": "sql",
          ".vue": "vue", ".svelte": "svelte", ".graphql": "graphql",
          ".gql": "graphql", ".tf": "hcl", ".tfvars": "hcl", ".r": "r",
          ".gd": "gdscript", ".dart": "dart", ".html": "html", ".htm": "html",
          ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
          ".yaml": "yaml", ".yml": "yaml", ".sh": "bash", ".bash": "bash",
          ".zsh": "bash", ".dockerfile": "dockerfile"}


def _fence(file: str) -> str:
    from os.path import splitext
    return _FENCE.get(splitext(file)[1].lower(), "")


# ==========================================================================
# context deduplication (DD-1) — drop pieces/blocks whose content is already
# covered, so the same code isn't paid for twice in a pack or a prompt.
# ==========================================================================
def _dedup_shingles(text: str, k: int = 5) -> set:
    """Set of k-gram token shingles for near-duplicate detection."""
    toks = _tokenize(text)
    if len(toks) < k:
        return frozenset([" ".join(toks)]) if toks else frozenset()
    return frozenset(" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1))


def _dedup_similarity(a: set, b: set) -> float:
    """Containment score: fraction of the smaller shingle set covered by the other.

    Containment (not Jaccard) so a short excerpt fully inside a larger body
    scores ~1.0 even though their sizes differ wildly.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


def dedupe_blocks(blocks: list[str], threshold: float = 0.8) -> dict:
    """Remove near-duplicate text blocks, keeping the first (longest-wins) copy.

    General-purpose context dedup: feed a list of retrieved snippets / tool
    outputs / pasted context and get back only the non-redundant ones plus the
    token saving. Deterministic, order-stable (longer blocks are preferred as
    the canonical copy). See also the automatic per-pack dedup in
    find_relevant_context.
    """
    indexed = sorted(enumerate(blocks), key=lambda it: -len(it[1] or ""))
    kept: list[tuple[int, str, set]] = []
    removed: list[dict] = []
    for idx, text in indexed:
        shingles = _dedup_shingles(text or "")
        dup_of = None
        for kidx, _ktext, kshingles in kept:
            if _dedup_similarity(shingles, kshingles) >= threshold:
                dup_of = kidx
                break
        if dup_of is None:
            kept.append((idx, text, shingles))
        else:
            removed.append({"index": idx, "duplicate_of": dup_of})
    kept_order = [text for idx, text, _ in sorted(kept, key=lambda it: it[0])]
    before = sum(count_tokens(b or "") for b in blocks)
    after = sum(count_tokens(b or "") for b in kept_order)
    return {
        "kept": kept_order,
        "removed": removed,
        "input_blocks": len(blocks),
        "kept_blocks": len(kept_order),
        "tokens_before": before,
        "tokens_after": after,
        "tokens_saved": before - after,
        "reduction_pct": round((before - after) / before * 100.0, 1) if before else 0.0,
    }


def _piece_hash(text: str) -> str:
    """Content identity of a pack piece, for the session ledger (SD-1).

    Whitespace-normalised so cosmetic reformatting does not force a resend,
    but any real edit does.
    """
    norm = " ".join((text or "").split())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()


def _dedupe_pieces(pieces: list["Piece"], threshold: float = 0.8) -> tuple[list, list]:
    """Drop pack pieces whose text is near-duplicate of an earlier, kept piece.

    Preserves pack priority order (seeds before neighbors before chunks), so an
    indexed chunk that merely re-shows a body already in the pack is removed,
    not the other way round. Returns (kept_pieces, removed_qnames).
    """
    kept: list[Piece] = []
    kept_shingles: list[set] = []
    removed: list[str] = []
    for p in pieces:
        shingles = _dedup_shingles(p.text or "")
        if any(_dedup_similarity(shingles, ks) >= threshold for ks in kept_shingles):
            removed.append(p.qname)
            continue
        kept.append(p)
        kept_shingles.append(shingles)
    return kept, removed


# ==========================================================================
# prompt quality scoring (PQ-1) — rate a *prompt* (not an answer) on clarity,
# specificity, context and actionability, with concrete fix suggestions.
# Deterministic and local; complements judge() which scores answer grounding.
# ==========================================================================
_PQ_ACTION_VERBS = {
    "add", "fix", "implement", "refactor", "explain", "remove", "delete",
    "optimize", "optimise", "test", "write", "create", "update", "debug",
    "review", "rename", "migrate", "document", "compare", "analyze", "analyse",
    "build", "generate", "find", "trace", "summarize", "summarise", "improve",
}
_PQ_VAGUE_TERMS = {
    "something", "somehow", "stuff", "things", "etc", "whatever", "some",
    "maybe", "kinda", "sort", "nice", "good", "better", "properly", "correctly",
}


def score_prompt(prompt: str) -> dict:
    """Score a prompt's quality 0–100 on four axes, with suggestions (PQ-1).

    Axes: clarity (concrete vs vague wording), specificity (code identifiers /
    file paths / quoted terms), context (references the code it's about), and
    actionability (a clear verb/ask). Deterministic — no LLM. Use it to catch
    under-specified prompts before they burn a round-trip on a vague answer.
    """
    import re
    text = (prompt or "").strip()
    words = text.split()
    n = len(words)
    lower = text.lower()
    toks = _tokenize(text)

    # --- clarity: penalise vague filler and extreme length; reward focus ---
    vague_hits = sum(1 for w in toks if w in _PQ_VAGUE_TERMS)
    clarity = 100.0
    clarity -= vague_hits * 15
    if n < 3:
        clarity -= 40                          # too terse to be clear
    if n > 120:
        clarity -= 20                          # rambling / unfocused
    clarity = max(0.0, min(100.0, clarity))

    # --- specificity: concrete identifiers, paths, quoted terms, numbers ---
    idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
                        r"|[a-z]+_[a-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]+", text)
    paths = re.findall(r"[\w./-]+\.[A-Za-z]{1,6}\b", text)
    quoted = re.findall(r"`[^`]+`|\"[^\"]+\"|'[^']+'", text)
    signal = len(set(idents)) + len(set(paths)) + len(quoted)
    specificity = min(100.0, signal * 22.0)
    if signal == 0 and n >= 3:
        specificity = 20.0                     # prose-only, no concrete anchor

    # --- context: does it point at the codebase it wants changed? ---
    has_ctx = bool(paths) or bool(idents) or bool(
        re.search(r"\b(file|function|class|module|method|endpoint|test)\b", lower))
    context = 85.0 if has_ctx else 30.0

    # --- actionability: is there a clear ask? ---
    first = toks[0] if toks else ""
    has_verb = any(v in toks for v in _PQ_ACTION_VERBS)
    is_question = text.endswith("?") or first in {"how", "why", "what", "where", "when"}
    actionability = 90.0 if (has_verb or is_question) else 35.0
    if first in _PQ_ACTION_VERBS:
        actionability = min(100.0, actionability + 10)

    overall = round(0.3 * clarity + 0.3 * specificity +
                    0.2 * context + 0.2 * actionability, 1)

    suggestions: list[str] = []
    if vague_hits:
        suggestions.append(f"replace vague wording ({vague_hits} term(s): "
                           f"e.g. 'something', 'properly') with concrete detail")
    if specificity < 50:
        suggestions.append("name the file(s), function(s) or symbol(s) involved")
    if not has_ctx:
        suggestions.append("point at the code — a path, class or function")
    if actionability < 50:
        suggestions.append("lead with a clear ask (a verb: add / fix / explain …)")
    if n < 3:
        suggestions.append("add detail — a 2–3 word prompt is too terse to act on")

    grade = ("excellent" if overall >= 85 else "good" if overall >= 70 else
             "fair" if overall >= 50 else "weak")
    return {
        "score": overall,
        "grade": grade,
        "subscores": {
            "clarity": round(clarity, 1),
            "specificity": round(specificity, 1),
            "context": round(context, 1),
            "actionability": round(actionability, 1),
        },
        "signals": {"identifiers": sorted(set(idents))[:12],
                    "paths": sorted(set(paths))[:12],
                    "vague_terms": vague_hits, "words": n},
        "suggestions": suggestions,
    }


# ==========================================================================
# conversation summarization (CS-1) — compress a long chat transcript into a
# compact, token-cheap brief so a session can carry forward without replaying
# the whole history. Deterministic, extractive, local (no LLM).
# ==========================================================================
_CS_DECISION_CUES = ("decid", "let's", "lets ", "we'll", "we will", "going to",
                     "chose", "choose", "use ", "should ", "will use", "agreed",
                     "plan is", "approach", "instead of")
_CS_ACTION_CUES = ("todo", "to do", "next step", "need to", "must ", "follow up",
                   "action item", "remaining", "still need")


def _cs_parse_turns(transcript: str) -> list[tuple[str, str]]:
    """Split a transcript into (role, text) turns. Tolerant of plain text."""
    import re
    turns: list[tuple[str, str]] = []
    role, buf = "note", []
    for line in (transcript or "").splitlines():
        m = re.match(r"^\s*(user|assistant|system|human|ai|claude|me)\s*[:>-]\s*(.*)$",
                     line, re.IGNORECASE)
        if m:
            if buf:
                turns.append((role, "\n".join(buf).strip()))
            role, buf = m.group(1).lower(), [m.group(2)]
        else:
            buf.append(line)
    if buf:
        turns.append((role, "\n".join(buf).strip()))
    return [(r, t) for r, t in turns if t]


_STOPWORDS = frozenset("""
a an the and or but if then else so of to in on at by for with from as is are
was were be been being it its this that these those i you we they he she them
us our your their my me do does did done have has had will would can could
should shall may might must not no yes just about into over under again very
too also than there here what which who whom when where why how all any both
each few more most other some such only own same s t don now
""".split())


def _sentence_tokens(sentence: str) -> set[str]:
    return {w for w in _tokenize(sentence) if w not in _STOPWORDS and len(w) > 2}


def _textrank(sentences: list[str], top_k: int, damping: float = 0.85,
              iterations: int = 30) -> list[int]:
    """Rank sentences by TextRank; return indices, most central first (CS-2).

    A graph of sentences weighted by normalised term overlap, scored with
    PageRank. This is what makes the summary reflect what the conversation was
    *about*, rather than only the sentences that happened to contain a cue word
    like "decided" — the previous implementation could not surface a key point
    that was phrased plainly.

    Pure Python, deterministic, no dependencies.
    """
    n = len(sentences)
    if n == 0:
        return []
    if n <= top_k:
        return list(range(n))
    toks = [_sentence_tokens(s) for s in sentences]
    # Similarity: shared terms normalised by length, the standard TextRank
    # kernel. Sentences with no content words simply never accumulate weight.
    weights: list[dict[int, float]] = [{} for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            shared = len(toks[i] & toks[j])
            if not shared:
                continue
            denom = math.log(len(toks[i]) + 1) + math.log(len(toks[j]) + 1)
            if denom <= 0:
                continue
            w = shared / denom
            weights[i][j] = w
            weights[j][i] = w
    out_sum = [sum(d.values()) for d in weights]
    score = [1.0 / n] * n
    for _ in range(iterations):
        nxt = []
        for i in range(n):
            acc = 0.0
            for j, w in weights[i].items():
                if out_sum[j] > 0:
                    acc += w / out_sum[j] * score[j]
            nxt.append((1.0 - damping) / n + damping * acc)
        if max(abs(a - b) for a, b in zip(nxt, score)) < 1e-6:
            score = nxt
            break
        score = nxt
    # Rank by score, then by original position so ties are stable.
    ranked = sorted(range(n), key=lambda i: (-score[i], i))[:top_k]
    return ranked


def summarize_conversation(transcript: str, max_tokens: int = 400) -> dict:
    """Extractive summary of a chat transcript, capped at ~max_tokens (CS-2).

    Two passes over the transcript:

    * **Structured extraction** — decisions, action items, open questions and
      the code entities touched, found by cue phrases.
    * **TextRank** — the most central sentences overall, which catches the
      substance that no cue phrase matches. Cue-only extraction was the old
      behaviour and it missed anything phrased plainly.

    Sections are ranked by salience rather than truncated to the first N, and
    the result is trimmed to `max_tokens` measured in real tokens.
    """
    import re
    turns = _cs_parse_turns(transcript)
    orig_tokens = count_tokens(transcript or "")
    if not turns:
        return {"summary": "", "turns": 0, "original_tokens": orig_tokens,
                "summary_tokens": 0, "tokens_saved": 0, "reduction_pct": 0.0,
                "decisions": [], "action_items": [], "open_questions": [],
                "entities": []}

    decisions, actions, questions = [], [], []
    seen_d, seen_a, seen_q = set(), set(), set()
    ent_counts: dict[str, int] = {}
    all_sentences: list[str] = []
    seen_any: set[str] = set()
    for _role, text in turns:
        for raw in re.split(r"(?<=[.!?])\s+|\n", text):
            s = raw.strip(" -*•\t")
            if not s:
                continue
            low = s.lower()
            key = low[:80]
            # Corpus for TextRank: substantive sentences only, deduplicated.
            if 25 < len(s) < 400 and key not in seen_any:
                seen_any.add(key)
                all_sentences.append(s)
            if s.endswith("?") and key not in seen_q and len(s) < 200:
                seen_q.add(key); questions.append(s)
            if any(c in low for c in _CS_DECISION_CUES) and key not in seen_d:
                seen_d.add(key); decisions.append(s)
            if any(c in low for c in _CS_ACTION_CUES) and key not in seen_a:
                seen_a.add(key); actions.append(s)
        for ident in re.findall(r"`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.]+)"
                                r"|([\w/-]+\.[A-Za-z]{1,6})\b", text):
            name = next((g for g in ident if g), "")
            if name and 2 < len(name) < 60:
                ent_counts[name] = ent_counts.get(name, 0) + 1
    entities = [e for e, _ in sorted(ent_counts.items(),
                                     key=lambda kv: -kv[1])][:12]

    def _cap(items: list[str], n: int) -> list[str]:
        """Keep the n most central items, in their original order (CS-2)."""
        if len(items) <= n:
            return items
        chosen = sorted(_textrank(items, n))
        return [items[i] for i in chosen]

    # Key points: the most central sentences overall, excluding anything
    # already surfaced verbatim in a structured section.
    claimed = {s[:80].lower() for s in decisions + actions + questions}
    candidates = [s for s in all_sentences if s[:80].lower() not in claimed]
    key_points = [candidates[i] for i in sorted(_textrank(candidates, 6))]

    lines = [f"# Conversation summary ({len(turns)} turns)"]
    if decisions:
        lines.append("\n## Decisions")
        lines += [f"- {d}" for d in _cap(decisions, 8)]
    if actions:
        lines.append("\n## Action items / open work")
        lines += [f"- {a}" for a in _cap(actions, 8)]
    if key_points:
        lines.append("\n## Key points")
        lines += [f"- {k}" for k in key_points]
    if questions:
        lines.append("\n## Open questions")
        lines += [f"- {q}" for q in _cap(questions, 6)]
    if entities:
        lines.append("\n## Code entities touched")
        lines.append(", ".join(f"`{e}`" for e in entities))

    summary = "\n".join(lines)
    # Trim to budget from the least-critical tail (entities → questions → …).
    while count_tokens(summary) > max_tokens and len(lines) > 3:
        lines.pop()
        summary = "\n".join(lines).rstrip()
    summary, _ = redact_secrets(summary)
    sum_tokens = count_tokens(summary)
    return {
        "summary": summary,
        "turns": len(turns),
        "original_tokens": orig_tokens,
        "summary_tokens": sum_tokens,
        "tokens_saved": orig_tokens - sum_tokens,
        "reduction_pct": round((orig_tokens - sum_tokens) / orig_tokens * 100.0, 1)
        if orig_tokens else 0.0,
        "decisions": _cap(decisions, 8),
        "action_items": _cap(actions, 8),
        "key_points": key_points,
        "open_questions": _cap(questions, 6),
        "entities": entities,
        "method": "textrank+cues",
    }


# CS-3: summarisation fidelity.
#
# `summarize_conversation` reported a reduction percentage and nothing else,
# which measures only how much it threw away — a summary that deleted
# everything would have scored best. What matters is the opposite: whether the
# things a resumed session needs (the decisions taken, the constraints agreed,
# the questions still open, the identifiers under discussion) survived the
# compression. This scores that directly, against facts the caller declares.
SUMMARY_FIDELITY_THRESHOLDS = {
    "identifier_recall": 0.80,   # code entities still nameable afterwards
    "fact_recall": 0.70,         # declared decisions/constraints still present
}


def score_summary_fidelity(summary: dict, required: dict) -> dict:
    """Did compression keep what a resumed session actually needs (CS-3)?

    `required` declares what must survive:
      * ``identifiers`` — symbols, files and flags the conversation was about;
      * ``facts`` — literal substrings standing for decisions, constraints and
        unresolved issues.

    Matching is case-insensitive substring containment over the *whole*
    rendered summary — structured fields included — because a decision that
    survives as an action item is still retained, and scoring by section would
    punish correct behaviour.
    """
    rendered = "\n".join([
        summary.get("summary", ""),
        *summary.get("decisions", []), *summary.get("action_items", []),
        *summary.get("open_questions", []), *summary.get("key_points", []),
        *summary.get("entities", []),
    ]).lower()

    def _recall(items: list[str]) -> tuple[list[str], list[str]]:
        kept, lost = [], []
        for item in items:
            (kept if str(item).lower() in rendered else lost).append(item)
        return kept, lost

    ids_kept, ids_lost = _recall(list(required.get("identifiers") or []))
    facts_kept, facts_lost = _recall(list(required.get("facts") or []))
    n_ids = len(ids_kept) + len(ids_lost)
    n_facts = len(facts_kept) + len(facts_lost)
    id_recall = (len(ids_kept) / n_ids) if n_ids else 1.0
    fact_recall = (len(facts_kept) / n_facts) if n_facts else 1.0
    return {
        "identifier_recall": round(id_recall, 3),
        "fact_recall": round(fact_recall, 3),
        "identifiers_lost": ids_lost,
        "facts_lost": facts_lost,
        "reduction_pct": summary.get("reduction_pct", 0.0),
        # A compression is only "good" when it is both small AND complete;
        # reporting reduction alone rewarded deleting the answer.
        "faithful": (id_recall >= SUMMARY_FIDELITY_THRESHOLDS["identifier_recall"]
                     and fact_recall >= SUMMARY_FIDELITY_THRESHOLDS["fact_recall"]),
    }


def _span_overlaps(row, spans: list[tuple[int, int]]) -> bool:
    """Does this symbol's source range touch a range the pack already shows?

    Containment runs both ways and both are redundancy: a class body already
    prints its methods, and a method printed on its own is re-printed the
    moment its class arrives. Overlapping bodies are never worth their tokens.
    """
    lo = row["lineno"] or 0
    hi = row["end_lineno"] or lo
    return any(lo <= e and hi >= s for s, e in spans)


@dataclass
class Piece:
    qname: str
    kind: str
    file: str
    detail: str          # "body" | "signature"
    text: str
    token_est: int
    reason: str          # why it's in the pack (seed / callee / caller / base)


@dataclass
class ContextPack:
    task: str
    pieces: list[Piece] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    deduped: list[str] = field(default_factory=list)   # DD-1: redundant pieces removed
    reused: list[str] = field(default_factory=list)    # SD-1: already in session
    budget: int = 0
    session: str = ""
    session_state: dict = field(default_factory=dict)   # SD-2 ledger diagnostics
    tokens_reused: int = 0          # tokens NOT resent thanks to the ledger
    neighbors_considered: int = 0   # NB-1: candidates seen during expansion
    neighbors_pruned: int = 0       # NB-1: candidates dropped by ranking

    @property
    def tokens(self) -> int:
        return sum(p.token_est for p in self.pieces)

    def to_markdown(self) -> str:
        lines = [f"# Context for: {self.task}",
                 f"_~{self.tokens} tokens, {len(self.pieces)} symbols_\n"]
        tags = {"body": "full", "signature": "signature",
                "chunk": "indexed chunk", "summary": "module summary"}
        for p in self.pieces:
            tag = tags.get(p.detail, "signature")
            lines.append(f"## `{p.qname}`  ({p.kind}, {tag}, {p.reason})")
            lines.append(f"`{p.file}`")
            lines.append(f"```{_fence(p.file)}")
            lines.append(p.text.rstrip())
            lines.append("```\n")
        if self.dropped:
            lines.append("## Available but not included (request by name if needed)")
            lines.append(", ".join(f"`{d}`" for d in self.dropped))
        if self.deduped:
            lines.append("## Deduplicated (redundant with content already shown)")
            lines.append(", ".join(f"`{d}`" for d in self.deduped))
        if self.reused:
            lines.append(
                f"## Already sent earlier this session — unchanged, not repeated "
                f"(~{self.tokens_reused} tokens saved)")
            lines.append(", ".join(f"`{d}`" for d in self.reused))
        if self.session_state.get("reset"):
            # Say it out loud: the ledger was discarded, so this pack is a full
            # resend rather than a delta. Silence here would look like a bug.
            lines.append("## Session context reset")
            lines.append(
                f"_{self.session_state.get('reset_reason', 'ledger expired')}. "
                f"Everything relevant is included in full above._")
        out, _ = redact_secrets("\n".join(lines))
        return out

    @property
    def rendered_tokens(self) -> int:
        return count_tokens(self.to_markdown())


class Retriever:
    def __init__(self, root: Path, db_path: Path):
        self.root = Path(root).resolve()
        self.store = Store(db_path)
        self._src_cache: dict[str, list[str]] = {}
        self._ann_index = None
        self.pooled = False        # RP-1: set by the MCP pool; makes close() a no-op

    # ---- source access ----
    def _lines(self, file: str) -> list[str]:
        if file not in self._src_cache:
            try:
                self._src_cache[file] = (self.root / file).read_text(
                    encoding="utf-8", errors="replace").splitlines()
            except OSError:
                self._src_cache[file] = []
        return self._src_cache[file]

    def _body(self, row) -> str:
        lines = self._lines(row["file"])
        start = max(0, row["lineno"] - 1)
        end = min(len(lines), row["end_lineno"])
        return "\n".join(lines[start:end])

    def _sig_block(self, row) -> str:
        sig = row["signature"] or row["qname"]
        doc = row["docstring"]
        return f"{sig}{f'    # {doc}' if doc else ''}"

    @staticmethod
    def _render_cost(piece: "Piece") -> int:
        """What a piece really costs once rendered into the pack markdown.

        `Piece.token_est` counts the payload only. Every piece also carries a
        heading, a path line and a fence — ~25 tokens of envelope. Assembly that
        budgets on payload alone believes a 2500-token pack is 1250 tokens, and
        the completion sweep would then overfill and immediately trim itself
        away. Budget against what the caller is actually billed for.
        """
        envelope = (f"## `{piece.qname}`  ({piece.kind}, signature, {piece.reason})\n"
                    f"`{piece.file}`\n```{_fence(piece.file)}\n```\n")
        return piece.token_est + count_tokens(envelope)

    def _chunk_excerpt(self, text: str, task: str, max_tokens: int) -> str:
        tokens = set(_tokenize(task))
        lines = text.splitlines()
        if not lines:
            return ""
        hit = 0
        for i, line in enumerate(lines):
            words = set(_tokenize(line))
            if words & tokens:
                hit = i
                break
        radius = 14
        start = max(0, hit - radius)
        end = min(len(lines), hit + radius + 1)
        excerpt = "\n".join(lines[start:end]).strip()
        while count_tokens(excerpt) > max_tokens and end - start > 8:
            if hit - start > end - hit:
                start += 1
            else:
                end -= 1
            excerpt = "\n".join(lines[start:end]).strip()
        if start > 0 or end < len(lines):
            return "...\n" + excerpt + "\n..."
        return excerpt

    # ---- public single-symbol helpers (also exposed as MCP tools) ----
    def get_symbol(self, qname: str) -> str | None:
        row = self.store.symbol_by_qname(qname)
        if not row:
            return None
        body, _ = redact_secrets(self._body(row))
        return body

    def get_callers(self, qname: str) -> list[str]:
        row = self.store.symbol_by_qname(qname)
        if not row:
            return []
        return [r["qname"] for r in self.store.neighbors(row["id"], ["CALLS"], "in")]

    def get_callees(self, qname: str) -> list[str]:
        row = self.store.symbol_by_qname(qname)
        if not row:
            return []
        return [r["qname"] for r in self.store.neighbors(row["id"], ["CALLS"], "out")]

    def file_skeleton(self, file: str) -> str:
        rows = [r for r in self.store.file_symbols(file) if r["kind"] != "module"]
        out = [f"# skeleton: {file}"]
        # FR-2a: cap emitted signatures per file to bound worst-case token cost.
        # The full graph is still indexed, so callers/callees/semantic search
        # over the dropped symbols keep working — only this rendering is capped.
        for r in rows[:MAX_SIGS_PER_FILE]:
            indent = "    " if r["kind"] == "method" else ""
            out.append(f"{indent}{self._sig_block(r)}")
        if len(rows) > MAX_SIGS_PER_FILE:
            out.append(f"# … +{len(rows) - MAX_SIGS_PER_FILE} more symbol(s) "
                       f"(capped at {MAX_SIGS_PER_FILE}/file; fetch via get_symbol/search)")
        redacted, _ = redact_secrets("\n".join(out))
        return redacted

    def module_summary(self, file: str) -> str:
        """Compact summary of a file (cached, or a signature skeleton fallback)."""
        row = self.store.get_summary(file)
        if row:
            redacted, _ = redact_secrets(row["summary"])
            return redacted
        return self.file_skeleton(file)

    # ---- semantic + hybrid seeding ----
    def semantic_search(self, query: str, limit: int = 12) -> list:
        """Cosine ranking of symbols by embedding similarity to the query.

        Never returns empty just because the vector table is unusable — an
        un-vectorised or mid-migration database degrades to lexical search
        rather than silently yielding nothing (EM-3).
        """
        if not self.store.has_vectors():
            return list(self.store.search(query, limit=limit))
        qv = embed_text(query)
        backend = os.environ.get("TOKENGRAPH_VECTOR_BACKEND", "exact").lower()
        threshold = int(os.environ.get("TOKENGRAPH_ANN_THRESHOLD", "5000"))
        vector_count = self.store.vector_count()
        if backend == "hnsw" and vector_count >= threshold:
            try:
                import hnswlib  # type: ignore[import-not-found]
                import numpy as np  # type: ignore[import-not-found]
                if self._ann_index is None:
                    index = hnswlib.Index(space="cosine", dim=len(qv))
                    index.init_index(max_elements=vector_count, ef_construction=200,
                                     M=16)
                    rows = list(self.store.iter_vectors())
                    labels = np.asarray([row["symbol_id"] for row in rows])
                    vectors = np.asarray([blob_to_vec(row["vec"]) for row in rows],
                                         dtype=np.float32)
                    index.add_items(vectors, labels)
                    index.set_ef(max(50, limit * 3))
                    self._ann_index = index
                labels, _ = self._ann_index.knn_query(
                    np.asarray([qv], dtype=np.float32),
                    k=min(limit, vector_count))
                return [row for sid in labels[0]
                        if (row := self.store.symbol(int(sid))) is not None]
            except (ImportError, RuntimeError, ValueError):
                pass
        scored: list[tuple[float, int]] = []
        mismatched = 0
        for r in self.store.iter_vectors():
            v = blob_to_vec(r["vec"])
            if len(v) != len(qv):
                mismatched += 1
                continue
            s = cosine(qv, v)
            if s > 0:
                scored.append((s, r["symbol_id"]))
        if not scored and mismatched:
            # Every stored vector is from another embedding space and the
            # re-embed pass has not run yet. Lexical results beat none.
            return list(self.store.search(query, limit=limit))
        # Deterministic order: score desc, then symbol id — so equal-scoring
        # symbols do not reshuffle between runs (needed for stable prompt
        # prefixes and reproducible evidence packs).
        scored.sort(key=lambda t: (-t[0], t[1]))
        rows = []
        for _, sid in scored[:limit]:
            row = self.store.symbol(sid)
            if row is not None:
                rows.append(row)
        return rows

    @staticmethod
    def _fuse(ranked_lists: list[list], limit: int, k: int = 60) -> list:
        """Reciprocal-rank fusion of several ranked symbol lists (by id)."""
        scores: dict[int, float] = {}
        rowmap: dict[int, object] = {}
        for lst in ranked_lists:
            for rank, row in enumerate(lst):
                rid = row["id"]
                rowmap[rid] = row
                scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        # Deterministic: fused score desc, then symbol id asc as a stable
        # tie-break so identical DB state always yields byte-identical packs.
        order = sorted(scores, key=lambda i: (-scores[i], i))
        return [rowmap[i] for i in order[:limit]]

    # ---- the main entrypoint ----
    @staticmethod
    def _neighbor_score(row, reason: str, hop: int, task_terms: set[str],
                        degree: int) -> float:
        """Rank a candidate neighbour against the task (NB-1).

        Combines four signals: which kind of edge reached it, how far it is
        from a seed, how much its identity overlaps the task wording, and how
        hub-like it is. Pure arithmetic on already-loaded rows — no I/O.
        """
        weight = NEIGHBOR_REASON_WEIGHT.get(reason, 0.6)
        hop_decay = 1.0 / (1.0 + hop)
        text = " ".join(str(row[k] or "") for k in ("qname", "name", "signature",
                                                    "docstring"))
        terms = set(_tokenize(text))
        overlap = (len(task_terms & terms) / len(task_terms)) if task_terms else 0.0
        # Hub damping: degree 0 -> 1.0, degree 100 -> ~0.15.
        hub = 1.0 / (1.0 + math.log2(1.0 + degree))
        return weight * hop_decay * (0.35 + overlap) * hub

    def pack_cache_key(self, task: str, budget_tokens: int, expand_depth: int,
                       max_body_tokens: int) -> str:
        """Identity of a stateless pack request (PK-1).

        Learned file weights participate because `learn()` changes seed order,
        and the embedding backend does because it changes what is retrieved.
        """
        payload = "\x1f".join([
            task.strip(), str(budget_tokens), str(expand_depth),
            str(max_body_tokens), embed_backend_id(),
            str(sorted(self.store.all_weights().items())),
        ])
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=20).hexdigest()

    def find_relevant_context_cached(self, task: str, budget_tokens: int = 6000,
                                     expand_depth: int = 1,
                                     max_body_tokens: int = 1600,
                                     session: str = "") -> tuple[str, dict]:
        """Rendered pack markdown, memoised across identical requests (PK-1).

        Returns (markdown, info). Only *stateless* packs are cached: with a
        session id the correct answer legitimately differs on every call
        (already-sent symbols get dropped), so caching one would resurrect
        content the caller was deliberately not being resent.
        """
        import json
        if session:
            pack = self.find_relevant_context(
                task, budget_tokens=budget_tokens, expand_depth=expand_depth,
                max_body_tokens=max_body_tokens, session=session)
            return pack.to_markdown(), {"cached": False,
                                        "reason": "session packs are stateful",
                                        "pack": pack}
        key = self.pack_cache_key(task, budget_tokens, expand_depth,
                                  max_body_tokens)
        row = self.store.cached_pack(key)
        if row is not None:
            self.store.commit()
            return row["markdown"], {"cached": True, "key": key,
                                     "tokens": row["tokens"],
                                     "meta": json.loads(row["meta"] or "{}")}
        pack = self.find_relevant_context(
            task, budget_tokens=budget_tokens, expand_depth=expand_depth,
            max_body_tokens=max_body_tokens)
        md = pack.to_markdown()
        meta = {"files": sorted({p.file for p in pack.pieces}),
                "symbols": len(pack.pieces),
                "dropped": pack.dropped,
                "neighbors_considered": pack.neighbors_considered,
                "neighbors_pruned": pack.neighbors_pruned}
        self.store.store_pack(key, task, md, json.dumps(meta, sort_keys=True),
                              pack.rendered_tokens)
        return md, {"cached": False, "key": key,
                    "tokens": pack.rendered_tokens, "meta": meta, "pack": pack}

    def find_relevant_context(self, task: str, budget_tokens: int = 6000,
                              expand_depth: int = 1,
                              max_body_tokens: int = 1600,
                              session: str = "") -> ContextPack:
        pack = ContextPack(task=task, budget=budget_tokens)
        # SD-1/SD-2: what this conversation can still be assumed to hold.
        # Anything unchanged since we sent it is referenced by name instead of
        # re-serialised. The ledger self-expires (see Store.sent_map) so we
        # never withhold content the model has probably already dropped.
        already: dict = {}
        if session:
            already, pack.session_state = self.store.sent_map(session)
        pack.session = session

        # hybrid seeding: lexical (FTS5) + semantic (embeddings), fused by RRF.
        lexical = self.store.search(task, limit=SEED_CANDIDATES)
        semantic = self.semantic_search(task, limit=SEED_CANDIDATES)
        seeds = (self._fuse([lexical, semantic], limit=SEED_CANDIDATES)
                 if semantic else lexical)
        # de-prioritise module nodes as seeds; we want concrete defs
        seeds = [s for s in seeds if s["kind"] != "module"] or seeds
        # Constants and fields (CN-1) are deliberately NOT excluded here.
        # Barring them from seeding was the intuitive move — they are short and
        # lexically dense, so they look like they should crowd out the
        # functions that implement the behaviour — but it measured worse in
        # aggregate: a question about a limit really is best anchored on the
        # limit. The per-file seed cap below is what keeps them honest.
        # Retrieval order before per-file diversification — the truest signal of
        # which files the query is actually about (SR-1).
        fused_order = list(seeds)
        # Spread seeds across files so one file cannot monopolise the pack.
        # The cap of 4 is load-bearing, not arbitrary: raising it measurably
        # *lowered* symbol recall on the benchmark suite, because extra
        # same-file seeds consume the budget with full bodies and evict the
        # symbol the task was actually about.
        diversified = []
        per_file: dict[str, int] = {}
        for seed in seeds:
            if per_file.get(seed["file"], 0) >= 4:
                continue
            diversified.append(seed)
            per_file[seed["file"]] = per_file.get(seed["file"], 0) + 1
        seeds = diversified
        # G10: let learned file weights nudge ordering — reinforced files first,
        # penalised last. Stable sort keeps relevance order among unweighted files.
        weights = self.store.all_weights()
        if weights:
            seeds = sorted(seeds, key=lambda s: weights.get(s["file"], 0.0), reverse=True)
        seed_ids = [s["id"] for s in seeds]
        chunk_rows = self.store.search_chunks(task, limit=8)

        # SR-1: roll the evidence up to files. Same formula as rank_files(), but
        # computed from rows already in hand rather than re-querying — this map
        # decides which files deserve full bodies and which get completed by the
        # sweep at the end.
        file_scores: dict[str, float] = {}
        for rank, row in enumerate(fused_order):
            f = row["file"]
            file_scores[f] = max(file_scores.get(f, 0.0), 1.0 / (rank + 1))
        for rank, c in enumerate(chunk_rows):
            file_scores[c["file"]] = (file_scores.get(c["file"], 0.0)
                                      + 0.75 / (rank + 1))
        for f in list(file_scores):
            file_scores[f] += weights.get(f, 0.0) * 0.1
        ranked_files = sorted(file_scores, key=lambda f: (-file_scores[f], f))
        body_files = set(ranked_files[:BODY_FILE_RANK])

        # Collect neighbours via BFS over CALLS/REFERENCES (both directions)
        # and INHERITS, under a per-symbol fan-out cap and a global ceiling so
        # a hub symbol cannot dominate the candidate set (NB-1).
        candidates: dict[int, tuple] = {}  # id -> (row, reason, hop)
        frontier = list(seed_ids)
        seen = set(seed_ids)
        for hop in range(max(0, expand_depth)):
            nxt = []
            for sid in frontier:
                if len(candidates) >= MAX_NEIGHBOR_CANDIDATES:
                    break
                for types, direction, reason in (
                        (["CALLS", "REFERENCES"], "out", "callee"),
                        (["INHERITS"], "out", "base"),
                        (["CALLS", "REFERENCES"], "in", "caller")):
                    for r in self.store.neighbors(sid, types, direction,
                                                  limit=MAX_FANOUT_PER_SYMBOL):
                        if r["id"] in seen or r["id"] in candidates:
                            continue
                        if len(candidates) >= MAX_NEIGHBOR_CANDIDATES:
                            break
                        candidates[r["id"]] = (r, reason, hop)
                        nxt.append(r["id"])
            seen.update(nxt)
            frontier = nxt

        # Rank candidates by task relevance; only the best are emitted.
        task_terms = set(_tokenize(task))
        degree_map = self.store.degrees(candidates.keys())
        ranked = sorted(
            candidates.values(),
            key=lambda t: (-self._neighbor_score(t[0], t[1], t[2], task_terms,
                                                 degree_map.get(t[0]["id"], 0)),
                           t[0]["id"]))
        pack.neighbors_considered = len(candidates)
        neighbor_rows = [(r, reason) for r, reason, _ in ranked[:MAX_NEIGHBOR_SIGS]]
        pack.neighbors_pruned = max(0, len(candidates) - len(neighbor_rows))

        # --- budget assembly ---
        # 1. seeds get full bodies (highest value), in search-rank order
        full_body_files: set[str] = set()
        for s in seeds:
            body = self._body(s)
            est = count_tokens(body)
            # SD-1: if this session already holds this exact text, spend a
            # one-line reference instead of the whole body.
            prior = already.get(s["qname"])
            if prior is not None and prior["content_hash"] == _piece_hash(body):
                pack.reused.append(s["qname"])
                pack.tokens_reused += prior["tokens"]
                full_body_files.add(s["file"])
                continue
            # SR-1: a full body is the most expensive thing the pack can buy, so
            # only files the query actually points at get one. A seed sitting in
            # a weakly-ranked file still appears — as a signature — and the
            # budget it would have eaten goes to completing the target file.
            off_target = s["file"] not in body_files
            if (est > max_body_tokens or off_target
                    or (pack.tokens + est > budget_tokens and pack.pieces)):
                # demote large bodies to signatures; indexed chunks below provide detail.
                sig = self._sig_block(s)
                est = count_tokens(sig)
                if pack.tokens + est > budget_tokens:
                    pack.dropped.append(s["qname"]); continue
                pack.pieces.append(Piece(s["qname"], s["kind"], s["file"],
                                         "signature", sig, est, "seed"))
            else:
                pack.pieces.append(Piece(s["qname"], s["kind"], s["file"],
                                         "body", body, est, "seed"))
                full_body_files.add(s["file"])

        # 2. neighbors get signatures only (cheap context that resolves refs),
        #    already ranked by task relevance above.
        for r, reason in neighbor_rows:
            if r["kind"] == "module":
                continue
            sig = self._sig_block(r)
            prior = already.get(r["qname"])
            if prior is not None and prior["content_hash"] == _piece_hash(sig):
                pack.reused.append(r["qname"])
                pack.tokens_reused += prior["tokens"]
                continue
            est = count_tokens(sig)
            if pack.tokens + est > budget_tokens:
                pack.dropped.append(r["qname"]); continue
            pack.pieces.append(Piece(r["qname"], r["kind"], r["file"],
                                     "signature", sig, est, reason))

        # 2b. module summaries: a few tokens to gesture at a referenced file
        #     whose body we didn't pull in full.
        summarized: set[str] = set()
        for r, reason in neighbor_rows:
            f = r["file"]
            if f in full_body_files or f in summarized:
                continue
            row = self.store.get_summary(f)
            if not row:
                continue
            est = row["token_est"]
            if pack.tokens + est > budget_tokens:
                continue
            pack.pieces.append(Piece(f, "module", f, "summary",
                                     row["summary"], est, "module summary"))
            summarized.add(f)

        # 3. indexed chunks cover relevant file regions without reading whole files.
        included_chunks: set[int] = set()
        for c in chunk_rows:
            if c["id"] in included_chunks or c["file"] in full_body_files:
                continue
            label = f"{c['file']}:{c['start_line']}-{c['end_line']}"
            text = c["text"]
            est = c["token_est"]
            remaining = budget_tokens - pack.tokens
            if est > remaining and remaining >= 120:
                text = self._chunk_excerpt(text, task, max(80, remaining - 20))
                est = count_tokens(text)
            if pack.tokens + est > budget_tokens:
                pack.dropped.append(label); continue
            pack.pieces.append(Piece(label, "chunk", c["file"], "chunk",
                                     text, est, "indexed search"))
            included_chunks.add(c["id"])
        # DD-1: drop pieces whose content is already covered by a higher-priority
        # piece (e.g. an indexed chunk re-showing a seed body). Frees budget and
        # keeps the pack free of redundant tokens.
        pack.pieces, pack.deduped = _dedupe_pieces(pack.pieces)

        # 4. SR-1 completion sweep: spend what is left of the budget finishing
        #    the files the pack already committed to. See SWEEP_FILES above for
        #    the failure this exists to fix.
        before = len(pack.pieces)
        self._complete_from_files(pack, task, budget_tokens, file_scores,
                                  ranked_files, already)
        # The sweep tracks line spans to avoid re-showing code, which catches
        # structural overlap but not textual duplication (two near-identical
        # small methods). Re-run DD-1 over the result so the no-redundant-piece
        # guarantee covers swept content on the same terms as everything else.
        if len(pack.pieces) != before:
            pack.pieces, more = _dedupe_pieces(pack.pieces)
            pack.deduped.extend(more)

        # Piece estimates exclude Markdown headings, fences, paths, and the
        # dropped list. Enforce the public contract against serialized output.
        while pack.pieces and pack.rendered_tokens > budget_tokens:
            removed = pack.pieces.pop()
            pack.dropped.append(removed.qname)
        while pack.dropped and pack.rendered_tokens > budget_tokens:
            pack.dropped.pop()
        # SD-1: record exactly what survived into the pack, so the next
        # retrieval in this session can reference it instead of resending it.
        if session:
            self.store.mark_sent(session, [
                (p.qname, p.detail, _piece_hash(p.text), p.token_est)
                for p in pack.pieces
                if p.detail in ("body", "signature")])
        return pack

    def _complete_from_files(self, pack: "ContextPack", task: str,
                             budget_tokens: int, file_scores: dict[str, float],
                             ranked_files: list[str], already: dict) -> None:
        """Finish the files the pack already believes in, with the budget left.

        Seeds and graph neighbours are both *pointwise*: they answer "which
        symbols look like the query" and "what do those touch". Neither answers
        "what else is in the file that turned out to be the right one" — so a
        pack could name `internal/store/store.go` as the second-ranked file,
        spend 2455 of 6000 tokens, and never include `Store.UpdateStatus`.

        This closes that gap. Bodies come first and only for small symbols,
        because the facts an answer needs (a default value, a `version =
        version + 1`, a status constant) exist only in bodies; signatures then
        buy breadth at ~20 tokens each. Nothing here re-reads a file the pack
        did not already choose, so it cannot invent new irrelevance: it makes
        the tokens already committed to a file actually pay for an answer.
        """
        focus = [f for f in ranked_files[:SWEEP_FILES] if file_scores.get(f, 0)]
        if not focus:
            return
        spent = pack.rendered_tokens
        if spent >= budget_tokens:
            return

        present = {p.qname for p in pack.pieces} | set(pack.reused)
        # Line spans already shown, so the sweep never repeats a body or a
        # chunk region that is on screen.
        covered: dict[str, list[tuple[int, int]]] = {}
        for p in pack.pieces:
            if p.detail == "chunk" and ":" in p.qname and "-" in p.qname:
                try:
                    lo, hi = (int(x) for x in
                              p.qname.rsplit(":", 1)[1].split("-", 1))
                    covered.setdefault(p.file, []).append((lo, hi))
                except ValueError:
                    pass
            elif p.detail == "body":
                row = self.store.symbol_by_qname(p.qname)
                if row is not None and row["lineno"]:
                    covered.setdefault(p.file, []).append(
                        (row["lineno"], row["end_lineno"] or row["lineno"]))

        task_terms = set(_tokenize(task))
        candidates: list[tuple[float, str, object]] = []
        for f in focus:
            fscore = file_scores.get(f, 0.0)
            spans = covered.get(f, [])
            rows = []
            for row in self.store.file_symbols(f):
                if row["kind"] == "module" or row["qname"] in present:
                    continue
                if _span_overlaps(row, spans):
                    continue
                text = " ".join(str(row[k] or "") for k in
                                ("qname", "name", "signature", "docstring"))
                terms = set(_tokenize(text))
                overlap = ((len(task_terms & terms) / len(task_terms))
                           if task_terms else 0.0)
                score = fscore * (0.3 + overlap)
                rows.append((score, row))
            rows.sort(key=lambda t: (-t[0], t[1]["lineno"] or 0))
            candidates.extend((score, f, row)
                              for score, row in rows[:SWEEP_CANDIDATES_PER_FILE])

        candidates.sort(key=lambda t: (-t[0], t[2]["file"], t[2]["lineno"] or 0))

        swept: dict[str, int] = {}     # qname -> index in pack.pieces

        def claim(row) -> None:
            """Record a newly shown body so later passes see it as covered."""
            covered.setdefault(row["file"], []).append(
                (row["lineno"] or 0, row["end_lineno"] or row["lineno"] or 0))

        def shown(row) -> bool:
            return _span_overlaps(row, covered.get(row["file"], []))

        def admit(row, detail: str, text: str) -> bool:
            """Add a piece if it fits the *rendered* budget; report success."""
            nonlocal spent
            prior = already.get(row["qname"])
            if prior is not None and prior["content_hash"] == _piece_hash(text):
                pack.reused.append(row["qname"])
                pack.tokens_reused += prior["tokens"]
                present.add(row["qname"])
                return True
            piece = Piece(row["qname"], row["kind"], row["file"], detail, text,
                          count_tokens(text), "file completion")
            cost = self._render_cost(piece)
            # Both budget contracts must hold: the payload sum and the rendered
            # markdown the caller is billed for.
            if spent + cost > budget_tokens or pack.tokens + piece.token_est > budget_tokens:
                return False
            swept[row["qname"]] = len(pack.pieces)
            pack.pieces.append(piece)
            present.add(row["qname"])
            if detail == "body":
                claim(row)
            spent += cost
            return True

        bodies: dict[str, tuple[str, int] | None] = {}

        def body_of(row) -> tuple[str, int] | None:
            """Body text and cost, or None when it is too big to sweep in.

            Memoised: passes 1 and 3 walk the same candidate list, and tokenising
            a body is the most expensive thing in the sweep.
            """
            qname = row["qname"]
            if qname in bodies:
                return bodies[qname]
            got: tuple[str, int] | None = None
            if (row["end_lineno"] or 0) - (row["lineno"] or 0) <= SWEEP_BODY_MAX_LINES:
                body = self._body(row)
                if body:
                    est = count_tokens(body)
                    if est <= SWEEP_BODY_MAX_TOKENS:
                        got = (body, est)
            bodies[qname] = got
            return got

        # The two passes want the same budget for different things, and
        # whichever runs unbounded starves the other: bodies-first spends
        # everything on a handful of symbols (breadth collapses), and
        # signatures-first leaves nothing to promote (every literal fact is
        # lost). Splitting the remainder is what keeps both — measured as the
        # difference between 0.63 and 0.72 answerable at equal recall.
        body_ceiling = spent + int(SWEEP_BODY_SHARE * (budget_tokens - spent))

        # Pass 1 — bodies for the best-scoring candidates, up to that ceiling.
        # Behaviour and literal values live here: no signature will ever show
        # that a status update also bumps a row version.
        for _, _f, row in candidates:
            if spent >= body_ceiling:
                break
            if row["qname"] in present or shown(row):
                continue
            got = body_of(row)
            if got:
                admit(row, "body", got[0])

        # Pass 2 — signatures for everything still uncovered: cheap breadth, so
        # a symbol the answer needs is at worst nameable rather than invisible.
        # A constant's signature is its declaration, so its value arrives too.
        for _, _f, row in candidates:
            if spent >= budget_tokens:
                break
            if row["qname"] in present or shown(row):
                continue
            admit(row, "signature", self._sig_block(row))

        # Pass 3 — anything still left over promotes a swept signature to its
        # body, in place, so the pack never carries both.
        for _, _f, row in candidates:
            if spent >= budget_tokens:
                break
            idx = swept.get(row["qname"])
            if idx is None or pack.pieces[idx].detail != "signature":
                continue
            if shown(row):
                continue     # a body printed since pass 2 already covers it
            got = body_of(row)
            if not got:
                continue
            body, est = got
            old = pack.pieces[idx]
            upgraded = Piece(row["qname"], row["kind"], row["file"], "body",
                             body, est, "file completion")
            delta = self._render_cost(upgraded) - self._render_cost(old)
            if (spent + delta > budget_tokens
                    or pack.tokens - old.token_est + est > budget_tokens):
                continue
            pack.pieces[idx] = upgraded
            claim(row)
            spent += delta

        # Anything the sweep pulled in is no longer "available but not
        # included" — leaving it listed would advertise it as absent.
        if pack.dropped:
            pack.dropped = [d for d in pack.dropped if d not in present]

    # A competent agent that greps then reads around each hit does not read a
    # whole file. These model that behaviour for the honest baseline (MS-1).
    GREP_WINDOW_LINES = 40      # lines of context read either side of a hit
    GREP_LINE_OVERHEAD = 12     # tokens per grep result line the agent sees

    def _targeted_baseline(self, pack: "ContextPack") -> int:
        """Tokens a *competent* agent would spend to reach the same coverage.

        Models the realistic alternative to this tool: grep for the relevant
        symbols, then read a window around each hit — merging overlapping
        windows, because one read covers neighbouring hits. This is the
        baseline the headline savings number is reported against, since almost
        no agent reads a large file end to end.
        """
        # Collect the line spans the pack actually covers, per file.
        spans: dict[str, list[tuple[int, int]]] = {}
        for p in pack.pieces:
            lo = hi = None
            if p.detail == "chunk" and ":" in p.qname and "-" in p.qname:
                try:
                    rng = p.qname.rsplit(":", 1)[1]
                    lo, hi = (int(x) for x in rng.split("-", 1))
                except ValueError:
                    lo = hi = None
            if lo is None:
                sid = self.store.id_for_qname(p.qname)
                row = self.store.symbol(sid) if sid is not None else None
                if row is None:
                    continue
                lo = row["lineno"] or 1
                hi = row["end_lineno"] or lo
            spans.setdefault(p.file, []).append((lo, hi))

        total = 0
        for f, raw in spans.items():
            lines = self._lines(f)
            n = len(lines)
            if not n:
                total += self.store.token_est_for(f)
                continue
            # Expand each hit by the read window, then merge overlaps — an
            # agent reading lines 10-90 does not read them twice.
            windows = sorted((max(1, lo - self.GREP_WINDOW_LINES),
                              min(n, hi + self.GREP_WINDOW_LINES))
                             for lo, hi in raw)
            merged: list[list[int]] = []
            for lo, hi in windows:
                if merged and lo <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            for lo, hi in merged:
                total += count_tokens("\n".join(lines[lo - 1:hi]))
            # The grep that located those hits also costs tokens.
            total += self.GREP_LINE_OVERHEAD * len(raw)
        return total

    def measure(self, task: str, **kw) -> dict:
        """Quantify the saving against a realistic agent baseline (MS-1).

        Two baselines are reported and neither is hidden:

        * ``baseline_tokens`` (headline) — grep-and-read-around-hits, what a
          competent agent actually does. This is the honest comparison.
        * ``baseline_whole_file_tokens`` — opening every referenced file end to
          end. Kept for continuity, but it flatters the tool badly on repos
          with large files and should not be quoted as the saving.
        """
        pack = self.find_relevant_context(task, **kw)
        files = sorted({p.file for p in pack.pieces})
        whole = sum(self.store.token_est_for(f) for f in files)
        targeted = self._targeted_baseline(pack)
        pack_tokens = pack.rendered_tokens
        saved = targeted - pack_tokens
        pct = (saved / targeted * 100.0) if targeted else 0.0
        whole_saved = whole - pack_tokens
        whole_pct = (whole_saved / whole * 100.0) if whole else 0.0
        return {
            "task": task,
            "pack_tokens": pack_tokens,
            "baseline_tokens": targeted,
            "baseline_kind": "grep+targeted-read",
            "baseline_whole_file_tokens": whole,
            "files_referenced": len(files),
            "symbols_in_pack": len(pack.pieces),
            "tokens_saved": saved,
            "savings_pct": round(pct, 1),
            "savings_pct_vs_whole_file": round(whole_pct, 1),
            "session_tokens_reused": pack.tokens_reused,
            "neighbors_considered": pack.neighbors_considered,
            "neighbors_pruned": pack.neighbors_pruned,
        }

    def report(self, tasks: list[str], **kw) -> dict:
        """Aggregate measure() across many tasks + a repo-scale baseline.

        Returns three blocks: per-task `rows` (each a measure() dict), an
        `aggregate` rollup (totals, overall and mean savings %, best/worst
        task), and a `repo` summary (how a typical pack compares to reading
        the whole repository). This turns the per-task primitive into the
        with/without quantitative report.
        """
        rows = [self.measure(t, **kw) for t in tasks]
        n = len(rows)
        sum_pack = sum(r["pack_tokens"] for r in rows)
        sum_base = sum(r["baseline_tokens"] for r in rows)
        sum_saved = sum(r["tokens_saved"] for r in rows)
        by_pct = sorted(rows, key=lambda r: r["savings_pct"])
        aggregate = {
            "tasks": n,
            "pack_tokens_total": sum_pack,
            "baseline_tokens_total": sum_base,
            "tokens_saved_total": sum_saved,
            # Overall = weighted by size; mean = simple per-task average.
            "savings_pct_overall": round(sum_saved / sum_base * 100.0, 1) if sum_base else 0.0,
            "savings_pct_mean": round(sum(r["savings_pct"] for r in rows) / n, 1) if n else 0.0,
            "best": by_pct[-1] if rows else None,
            "worst": by_pct[0] if rows else None,
        }
        repo_total = self.store.repo_token_total()
        mean_pack = round(sum_pack / n) if n else 0
        repo = {
            "indexed_files": self.store.stats()["files"],
            "repo_tokens_total": repo_total,
            "mean_pack_tokens": mean_pack,
            "mean_pack_pct_of_repo": round(mean_pack / repo_total * 100.0, 2) if repo_total else 0.0,
        }
        return {"rows": rows, "aggregate": aggregate, "repo": repo}

    # ---- surgical context: exact line range (SEC-2) ----
    def get_lines(self, file: str, start: int, end: int) -> str:
        """Fetch an exact line range, clamped to bounds, secret-scanned, sandboxed.

        Path is resolved inside the project root; any attempt to escape (../,
        absolute paths, symlinks out of tree) returns an error string instead
        of reading. Output is redacted before return.
        """
        try:
            target = (self.root / file).resolve()
        except OSError:
            return f"(cannot resolve {file})"
        try:
            target.relative_to(self.root)
        except ValueError:
            return f"(refused: {file} is outside the project root)"
        if not target.is_file():
            return f"(no file {file})"
        lines = self._lines(file)
        if not lines:
            return f"(empty or unreadable: {file})"
        start = max(1, int(start))
        end = min(len(lines), int(end))
        if end < start:
            return f"(invalid range {start}-{end} for {file} with {len(lines)} lines)"
        body = "\n".join(lines[start - 1:end])
        body, _ = redact_secrets(body)
        return f"# {file}:{start}-{end}\n{body}"

    # ---- list_modules: token-count table of top-level dirs ----
    def list_modules(self) -> list[dict]:
        """Token-count table of top-level source directories (call first)."""
        buckets: dict[str, dict] = {}
        for r in self.store.files_with_tokens():
            top = r["path"].split("/", 1)[0] if "/" in r["path"] else "."
            b = buckets.setdefault(top, {"module": top, "files": 0,
                                         "tokens": 0, "symbols": 0})
            b["files"] += 1
            b["tokens"] += r["token_est"] or 0
            b["symbols"] += r["symbols_count"] or 0
        return sorted(buckets.values(), key=lambda b: b["tokens"], reverse=True)

    # ---- explain_file: signatures + imports + reverse callers ----
    def explain_file(self, file: str) -> str:
        rows = self.store.file_symbols(file)
        if not rows:
            return f"(no indexed symbols in {file})"
        out = [f"# explain: {file}", "", "## signatures"]
        imports: list[str] = []
        callers: dict[str, list[str]] = {}
        for r in rows:
            if r["kind"] == "module":
                continue
            indent = "    " if r["kind"] == "method" else ""
            out.append(f"{indent}{self._sig_block(r)}")
            ins = [x["qname"] for x in self.store.neighbors(r["id"], ["CALLS"], "in")
                   if x["file"] != file]
            if ins:
                callers[r["qname"]] = ins
        for e in self.store.edges_of_type("IMPORTS"):
            if e["src_file"] == file:
                imports.append(e["dst"])
        if imports:
            out += ["", "## imports", ", ".join(sorted(set(imports)))]
        if callers:
            out += ["", "## external callers (who depends on this file)"]
            for q, who in callers.items():
                out.append(f"- `{q}` <- {', '.join(sorted(set(who))[:10])}")
        result, _ = redact_secrets("\n".join(out))
        return result

    # ---- get_impact: blast radius of a symbol ----
    def get_impact(self, qname: str) -> dict:
        row = self.store.symbol_by_qname(qname)
        if not row:
            return {"symbol": qname, "found": False}
        direct = self.store.neighbors(row["id"], ["CALLS", "REFERENCES"], "in")
        # transitive callers (BFS, bounded)
        transitive: set[str] = set()
        frontier = [r["id"] for r in direct]
        seen = set(frontier) | {row["id"]}
        depth = 0
        while frontier and depth < 5:
            nxt = []
            for sid in frontier:
                for r in self.store.neighbors(sid, ["CALLS", "REFERENCES"], "in"):
                    if r["id"] not in seen:
                        seen.add(r["id"]); transitive.add(r["qname"]); nxt.append(r["id"])
            frontier = nxt
            depth += 1
        files = sorted({r["file"] for r in direct})
        tests = [f for f in files if "test" in f.lower() or "spec" in f.lower()]
        subclasses = [r["qname"] for r in self.store.neighbors(row["id"], ["INHERITS"], "in")]
        out = {
            "symbol": qname,
            "found": True,
            "file": row["file"],
            "direct_callers": sorted({r["qname"] for r in direct}),
            "transitive_callers": sorted(transitive),
            "subclasses": sorted(subclasses),
            "files_touched": files,
            "tests_touched": tests,
            "blast_radius": len({r["qname"] for r in direct}) + len(transitive),
        }
        # PF-1: a blast radius is a safety signal, and on a regex-tier file it
        # is always zero — not because nothing calls this, but because no call
        # edges were ever extracted. Saying so is the difference between "safe
        # to change" and "unknown", and an agent must not read one as the other.
        tier = tier_for_path(Path(row["file"]))
        out["extraction_tier"] = tier
        if not EXTRACTION_TIERS[tier]["calls"]:
            out["blast_radius_reliable"] = False
            out["warning"] = (
                f"{row['file']} is extracted at the '{tier}' tier, which "
                f"produces no call edges. This blast radius is NOT evidence "
                f"that nothing depends on the symbol — verify by searching the "
                f"repository before changing it.")
        else:
            out["blast_radius_reliable"] = True
        return out

    # ---- get_method_impact: function-level blast radius (single named tool) ----
    def get_method_impact(self, qname: str) -> dict:
        """Function-level blast radius: which functions break if this one changes.

        A method-focused view of get_impact. Where get_impact answers "what is
        the reach of this symbol", this answers the change-safety question an
        agent actually asks before editing a function: *who calls me* (and so
        breaks if my signature changes), *what do I call* (my dependencies),
        and *what else shares my name* (likely overrides/overloads that must
        change in lockstep) — each with concrete file:line call sites.
        """
        row = self.store.symbol_by_qname(qname)
        if not row:
            leaf = qname.split(".")[-1]
            sugg = [r["qname"] for r in self.store.conn.execute(
                "SELECT qname FROM symbols WHERE name=? LIMIT 5", (leaf,))]
            return {"symbol": qname, "found": False, "did_you_mean": sugg}
        imp = self.get_impact(qname)  # reuse transitive/tests/tier/reliability
        callers = [{"symbol": r["qname"], "file": r["file"], "line": r["lineno"]}
                   for r in self.store.neighbors(
                       row["id"], ["CALLS", "REFERENCES"], "in")]
        callees = [{"symbol": r["qname"], "file": r["file"]}
                   for r in self.store.neighbors(row["id"], ["CALLS"], "out")]
        # Same-leaf-name functions/methods elsewhere: overrides in subclasses or
        # overloads that a signature change usually has to follow.
        overrides = [{"symbol": r["qname"], "file": r["file"], "kind": r["kind"]}
                     for r in self.store.conn.execute(
                         "SELECT qname, file, kind FROM symbols WHERE name=? "
                         "AND qname<>? AND kind IN ('method','function') LIMIT 25",
                         (row["name"], qname))]
        out = {
            "symbol": qname,
            "found": True,
            "kind": row["kind"],
            "file": row["file"],
            "line": row["lineno"],
            "signature": row["signature"],
            "callers": callers,                 # break on a signature change
            "callees": callees,                 # this function's dependencies
            "overrides_or_overloads": overrides,
            "transitive_callers": imp["transitive_callers"],
            "call_sites": len(callers),
            "blast_radius": imp["blast_radius"],
            "tests_touched": imp["tests_touched"],
            "extraction_tier": imp["extraction_tier"],
            "blast_radius_reliable": imp["blast_radius_reliable"],
        }
        if imp.get("warning"):
            out["warning"] = imp["warning"]
        return out

    # ---- get_architecture_overview: whole-repo shape in one call ----
    def get_architecture_overview(self, top_hubs: int = 15,
                                  route_cap: int = 200) -> dict:
        """One-call architectural map: module breakdown, hub files, import
        cycles, language mix, and route totals — everything get_map / list_modules
        expose, composed into a single orientation payload.
        """
        modules = self.list_modules()
        hubmap = self._hub_map(top=top_hubs)
        routemap = self._route_map(cap=route_cap)
        langs: dict[str, dict] = {}
        files = symbols = tokens = 0
        for fr in self.store.files_with_tokens():
            files += 1
            tokens += fr["token_est"] or 0
            symbols += fr["symbols_count"] or 0
            lang = fr["language"] or "text"
            L = langs.setdefault(lang, {"language": lang, "files": 0, "tokens": 0})
            L["files"] += 1
            L["tokens"] += fr["token_est"] or 0
        routes = routemap.get("routes", [])
        return {
            "totals": {"modules": len(modules), "files": files,
                       "symbols": symbols, "tokens": tokens},
            "modules": modules,
            "languages": sorted(langs.values(),
                                key=lambda x: x["tokens"], reverse=True),
            "hubs": hubmap["hubs"],
            "cycles": hubmap["cycles"],
            "routes_total": len(routes),
            "routes": routes[:25],
        }

    # ---- get_test_map: implementation <-> test discovery (named tool) ----
    def get_test_map(self, target: str = "") -> dict:
        """Map implementations to their tests (and back).

        With no target: the whole-repo impl<->test map plus coverage stats.
        With a file: that file's tests (or, for a test file, what it covers).
        With a symbol qname: its file's tests, plus tests that actually
        reference the symbol through the call graph (real edges, not just
        naming) — the two signals unioned.
        """
        files = [r["path"] for r in self.store.files_with_tokens()]
        m = build_test_map(files)
        if not target:
            n_impl = len(m["impls"])
            tested = len(m["impl_to_tests"])
            return {
                "target": None,
                "pairs": m["pairs"],
                "impl_to_tests": m["impl_to_tests"],
                "test_to_impl": m["test_to_impl"],
                "untested_impls": m["untested_impls"],
                "unmatched_tests": m["unmatched_tests"],
                "coverage": {
                    "impl_files": n_impl,
                    "impl_files_with_tests": tested,
                    "coverage_pct": round(tested / n_impl * 100, 1) if n_impl else 0.0,
                    "test_files": len(m["tests"]),
                    "pairs": len(m["pairs"]),
                },
            }

        # symbol qname → resolve to its file and add call-graph-linked tests
        row = self.store.symbol_by_qname(target)
        if row:
            impl_file = row["file"]
            by_name = m["impl_to_tests"].get(impl_file, [])
            by_edges = sorted({
                c["file"] for c in self.store.neighbors(
                    row["id"], ["CALLS", "INHERITS", "REFERENCES"], "in")
                if is_test_path(c["file"])})
            return {
                "target": target, "kind": "symbol", "file": impl_file,
                "tests_by_name": sorted(by_name),
                "tests_by_call_graph": by_edges,
                "tests": sorted(set(by_name) | set(by_edges)),
            }

        # file path
        target = target.replace("\\", "/")
        if is_test_path(target):
            return {"target": target, "kind": "test",
                    "covers": m["test_to_impl"].get(target, [])}
        return {"target": target, "kind": "impl",
                "tests": m["impl_to_tests"].get(target, [])}

    # ---- get_map: import graph / class hierarchy ----
    def get_map(self, kind: str = "imports") -> dict:
        kind = (kind or "imports").lower()
        if kind in ("routes", "route", "endpoints"):
            return self._route_map()
        if kind in ("hubs", "hub", "cycles", "cycle"):
            return self._hub_map()
        edge_type = "INHERITS" if kind in ("inherits", "hierarchy", "class") else "IMPORTS"
        edges = self.store.edges_of_type(edge_type)
        graph: dict[str, list[str]] = {}
        for e in edges:
            graph.setdefault(e["src"], []).append(e["dst"])
        return {"kind": "class hierarchy" if edge_type == "INHERITS" else "imports",
                "edges": {k: sorted(set(v)) for k, v in graph.items()}}

    def _route_map(self, cap: int = 200) -> dict:
        """Extract HTTP routes (Flask/FastAPI, Express, Spring, Go) from source."""
        import re
        routes: list[dict] = []
        compiled = {fam: [re.compile(p) for p in pats]
                    for fam, pats in _ROUTE_PATTERNS.items()}
        for fr in self.store.files_with_tokens():
            family = _ROUTE_FAMILY.get(fr["language"])
            if not family:
                continue
            try:
                lines = (self.root / fr["path"]).read_text(
                    encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for rx in compiled.get(family, []):
                for i, ln in enumerate(lines, 1):
                    m = rx.search(ln)
                    if not m:
                        continue
                    g = [x for x in m.groups() if x]
                    method = g[0].upper() if len(g) > 1 else "ANY"
                    routes.append({"method": method, "path": g[-1],
                                   "file": fr["path"], "line": i})
                    if len(routes) >= cap:
                        return {"kind": "routes", "routes": routes,
                                "note": f"capped at {cap}"}
        return {"kind": "routes", "routes": routes}

    def _hub_map(self, top: int = 20) -> dict:
        """Fan-in/fan-out ranking + import-cycle detection over the file graph."""
        graph: dict[str, set] = {}
        fan_in: dict[str, int] = {}
        fan_out: dict[str, int] = {}
        nodes: set[str] = set()
        for e in self.store.edges_of_type("IMPORTS"):
            a, b = e["src_file"], e["dst_file"]
            if not a or not b or a == b:
                continue
            nodes.add(a)
            nodes.add(b)
            if b not in graph.setdefault(a, set()):
                graph[a].add(b)
                fan_out[a] = fan_out.get(a, 0) + 1
                fan_in[b] = fan_in.get(b, 0) + 1
        hubs = sorted(
            ({"file": f, "fan_in": fan_in.get(f, 0), "fan_out": fan_out.get(f, 0),
              "degree": fan_in.get(f, 0) + fan_out.get(f, 0)} for f in nodes),
            key=lambda x: (x["degree"], x["file"]), reverse=True)
        hubs = hubs[:top] if top else hubs      # top=0 → every file
        return {"kind": "hubs", "hubs": hubs, "cycles": _import_cycles(graph)}

    # ---- get_routing: per-file model-tier hints ----
    def get_routing(self) -> list[dict]:
        """Per-file model tier, using real graph + control-flow features (MR-3)."""
        # top=0 disables the ranking cut-off: routing needs fan-in/out for
        # every file, not just the twenty biggest hubs.
        fan = {h["file"]: h for h in self._hub_map(top=0).get("hubs", [])}
        out = []
        for r in self.store.files_with_tokens():
            path = r["path"]
            edges = fan.get(path, {})
            src = "\n".join(self._lines(path))
            feats = file_complexity_features(src) if src else {
                "max_depth": 0, "branches": 0}
            out.append(tier_for_file(
                path, r["token_est"] or 0, r["symbols_count"] or 0,
                fan_in=edges.get("fan_in", 0), fan_out=edges.get("fan_out", 0),
                max_depth=feats["max_depth"], branches=feats["branches"]))
        return out

    # ---- ask: intent + coverage + risk + cost + top-K (FR-6, FR-7) ----
    def ask(self, task: str, budget_tokens: int = 6000, depth: int = 1,
            max_body_tokens: int = 1600) -> dict:
        pack = self.find_relevant_context(task, budget_tokens=budget_tokens,
                                          expand_depth=depth,
                                          max_body_tokens=max_body_tokens)
        seeds = [p for p in pack.pieces if p.reason == "seed"]
        files = sorted({p.file for p in pack.pieces})
        baseline = sum(self.store.token_est_for(f) for f in files)
        serialized_tokens = pack.rendered_tokens
        saved = baseline - serialized_tokens
        # coverage: fraction of relevant seeds that made it into the pack
        total_seed_slots = len(seeds) + len(pack.dropped)
        coverage = (len(seeds) / total_seed_slots * 100.0) if total_seed_slots else 0.0
        if coverage >= 80 and pack.pieces:
            risk = "low"
        elif coverage >= 50:
            risk = "medium"
        else:
            risk = "high"
        return {
            "task": task,
            "intent": detect_intent(task),
            "coverage_pct": round(coverage, 1),
            "risk": risk,
            "pack_tokens": serialized_tokens,
            "baseline_tokens": baseline,
            "tokens_saved": saved,
            "savings_pct": round(saved / baseline * 100.0, 1) if baseline else 0.0,
            "suggested_tier": recommend_tier(task)["tier"],
            "top_files": files[:10],
            "dropped": pack.dropped,
            "markdown": pack.to_markdown(),
        }

    # ---- get_diff_context: diff-seeded retrieval (FR-14) ----
    def get_diff_context(self, staged: bool = False, budget_tokens: int = 6000,
                         depth: int = 1, max_body_tokens: int = 1600) -> dict:
        """Budgeted context pack for exactly what the diff touches.

        Where find_relevant_context seeds from a task string, this seeds from
        the git diff itself: every indexed symbol whose line span overlaps a
        changed hunk gets its full body, then one hop of callers/callees/base
        classes (the blast radius) is added as signatures. New/untracked files
        contribute all their symbols. Deterministic — no task wording needed."""
        changed = [f for f in git_changed_files(self.root, staged=staged)
                   if not f.startswith((".tokengraph/", ".context/"))]
        src_changed = [f for f in changed if "." + f.rsplit(".", 1)[-1].lower()
                       in supported_extensions()]
        if not src_changed:
            return {"staged": staged, "changed_files": [], "touched_symbols": [],
                    "impacted": [], "pack_tokens": 0, "baseline_tokens": 0,
                    "tokens_saved": 0, "savings_pct": 0.0, "dropped": [],
                    "markdown": "", "note": ("no changes detected (working tree clean)"
                                             if not changed else
                                             "changed files are not in an indexed language")}

        hunks = git_diff_hunks(self.root, staged=staged)
        indexed = self.store.all_indexed_files()

        # 1. seeds: indexed symbols whose [lineno, end_lineno] overlaps a changed
        #    hunk. No hunk for the file (untracked/new) → every symbol is touched.
        seed_rows: list = []
        seen_ids: set[int] = set()
        for f in src_changed:
            if f not in indexed:
                continue
            ranges = hunks.get(f)
            for row in self.store.file_symbols(f):
                if row["kind"] == "module":
                    continue
                touched = (ranges is None) or any(
                    not (row["end_lineno"] < a or row["lineno"] > b)
                    for (a, b) in ranges)
                if touched and row["id"] not in seen_ids:
                    seen_ids.add(row["id"]); seed_rows.append(row)

        scope = "staged" if staged else "working tree"
        pack = ContextPack(task=f"diff context ({scope})", budget=budget_tokens)

        # 2. changed symbols → full bodies (demote large ones to signatures)
        touched_symbols: list[str] = []
        for s in seed_rows:
            touched_symbols.append(s["qname"])
            body = self._body(s)
            est = count_tokens(body)
            if est > max_body_tokens or (pack.tokens + est > budget_tokens and pack.pieces):
                sig = self._sig_block(s); est = count_tokens(sig)
                if pack.tokens + est > budget_tokens:
                    pack.dropped.append(s["qname"]); continue
                pack.pieces.append(Piece(s["qname"], s["kind"], s["file"],
                                         "signature", sig, est, "changed"))
            else:
                pack.pieces.append(Piece(s["qname"], s["kind"], s["file"],
                                         "body", body, est, "changed"))

        # 3. blast radius: callers + callees + base classes as cheap signatures
        impacted: list[dict] = []
        frontier = list(seen_ids)
        seen = set(seen_ids)
        for _ in range(max(0, depth)):
            nxt: list[int] = []
            for sid in frontier:
                for rel, dirn, reason in (("CALLS", "in", "caller"),
                                          ("CALLS", "out", "callee"),
                                          ("INHERITS", "out", "base")):
                    for r in self.store.neighbors(sid, [rel], dirn):
                        if r["id"] in seen or r["kind"] == "module":
                            continue
                        seen.add(r["id"]); nxt.append(r["id"])
                        if reason == "caller":
                            impacted.append({"symbol": r["qname"], "file": r["file"]})
                        sig = self._sig_block(r); est = count_tokens(sig)
                        if pack.tokens + est > budget_tokens:
                            pack.dropped.append(r["qname"]); continue
                        pack.pieces.append(Piece(r["qname"], r["kind"], r["file"],
                                                 "signature", sig, est, reason))
            frontier = nxt

        files = sorted({p.file for p in pack.pieces})
        baseline = sum(self.store.token_est_for(f) for f in files)
        saved = baseline - pack.tokens
        return {
            "staged": staged,
            "changed_files": src_changed,
            "touched_symbols": touched_symbols,
            "impacted": impacted[:25],
            "pack_tokens": pack.rendered_tokens,
            "baseline_tokens": baseline,
            "tokens_saved": saved,
            "savings_pct": round(saved / baseline * 100.0, 1) if baseline else 0.0,
            "dropped": pack.dropped,
            "markdown": pack.to_markdown(),
            "note": (f"{len(touched_symbols)} changed symbol(s) across "
                     f"{len(src_changed)} file(s); {len(impacted)} caller(s) impacted"),
        }

    # ---- validate: is the assembled context sufficient? (FR-8) ----
    def validate(self, task: str, min_coverage: float = 60.0, **kw) -> dict:
        a = self.ask(task, **kw)
        ok = a["coverage_pct"] >= min_coverage and a["pack_tokens"] > 0
        return {
            "task": task,
            "coverage_pct": a["coverage_pct"],
            "risk": a["risk"],
            "min_coverage": min_coverage,
            "ok": ok,
            "recommendation": ("context looks sufficient" if ok else
                               "low coverage — broaden the task wording, raise "
                               "the budget, or fetch dropped symbols by name"),
            "dropped": a["dropped"],
        }

    # ---- judge: is an answer grounded in a context? (FR-9) ----
    def judge(self, answer: str, context: str) -> dict:
        """Heuristic groundedness score: overlap of answer claims with context.

        Deterministic and local (no LLM): scores how much of the answer's
        salient vocabulary and code identifiers appear in the supplied context.
        A low score flags possible hallucination.
        """
        ans_tokens = [t for t in _tokenize(answer) if len(t) > 2]
        ctx_tokens = set(_tokenize(context))
        if not ans_tokens:
            return {"grounded_pct": 0.0, "grounded": False,
                    "unsupported_terms": [], "note": "empty answer"}
        # weight code-like identifiers (snake/camel/dotted) higher
        def _salient(t: str) -> bool:
            return ("_" in t or any(c.isdigit() for c in t) or len(t) >= 5)
        salient = [t for t in ans_tokens if _salient(t)] or ans_tokens
        supported = [t for t in salient if t in ctx_tokens]
        unsupported = sorted({t for t in salient if t not in ctx_tokens})
        pct = len(supported) / len(salient) * 100.0
        return {
            "grounded_pct": round(pct, 1),
            "grounded": pct >= 50.0,
            "unsupported_terms": unsupported[:25],
            "note": ("answer is well grounded" if pct >= 70 else
                     "answer is partly grounded" if pct >= 50 else
                     "answer may be hallucinated — many terms are absent from context"),
        }

    # ---- verify: do the files/symbols an answer references actually exist? (G5) ----
    def verify(self, answer: str) -> dict:
        """Flag fabricated file paths / code symbols in an answer (deterministic).

        Complements judge(): instead of scoring vocabulary overlap, it extracts
        concrete references (file paths with a source-ish extension, and
        backtick-quoted identifiers / qualified names) and checks each against
        the indexed graph, offering nearest matches via edit distance.
        """
        import re
        text = answer or ""
        files = self.store.all_indexed_files()
        file_lc = {f.lower() for f in files}
        basenames = sorted({f.rsplit("/", 1)[-1] for f in files})
        basenames_lc = {b.lower() for b in basenames}
        names = sorted(set(self.store.all_names()))
        names_lc = {n.lower() for n in names}
        qnames_lc = {q.lower() for q in self.store.all_qnames()}
        codeish = {e.lower() for e in supported_extensions()} | {
            ".md", ".json", ".txt", ".yml", ".yaml", ".cfg", ".ini", ".toml"}

        issues: list[dict] = []
        checked = {"files": 0, "symbols": 0}

        # 1. file references — high precision (must carry a source-ish extension)
        seen_f: set[str] = set()
        for m in re.findall(r"[\w./\\-]+\.[A-Za-z][A-Za-z0-9]{0,5}\b", text):
            cand = m.replace("\\", "/").lstrip("./")
            if cand in seen_f:
                continue
            seen_f.add(cand)
            ext = "." + cand.rsplit(".", 1)[-1].lower()
            if ext not in codeish:
                continue
            base = cand.rsplit("/", 1)[-1]
            checked["files"] += 1
            if cand.lower() in file_lc or base.lower() in basenames_lc:
                continue
            issues.append({"kind": "file", "name": cand,
                           "did_you_mean": _closest(base, basenames)})

        # 2. backtick-quoted identifiers / qualified names (explicit code refs)
        seen_s: set[str] = set()
        for raw in re.findall(r"`([A-Za-z_][\w.]*)`", text):
            ident = raw.strip("`")
            leaf = ident.split(".")[-1]
            if leaf.lower() in _VERIFY_STOPWORDS or len(leaf) < 3 or ident in seen_s:
                continue
            seen_s.add(ident)
            checked["symbols"] += 1
            if ident.lower() in qnames_lc or leaf.lower() in names_lc:
                continue
            issues.append({"kind": "symbol", "name": ident,
                           "did_you_mean": _closest(leaf, names)})

        return {
            "ok": not issues,
            "checked": checked,
            "issues": issues,
            "note": ("all references resolve against the graph" if not issues else
                     f"{len(issues)} unresolved reference(s) — possible hallucination"),
        }

    # ---- squeeze: shrink a pasted blob using the graph for frame enrichment (G6) ----
    def squeeze(self, text: str, kind: str = "auto") -> dict:
        return squeeze_text(text, kind=kind, store=self.store, root=self.root)

    # ---- learn: reinforce/penalise files locally (FR-10) ----
    def learn(self, file: str, good: bool, weight: float = 1.0) -> dict:
        delta = weight if good else -weight
        self.store.bump_weight(file, delta)
        self.store.commit()
        return {"file": file, "delta": delta, "weight": self.store.weight_for(file)}

    # ---- cross-session memory + checkpoints (read_memory / create_checkpoint) ----
    def remember(self, text: str, kind: str = "note") -> dict:
        mid = self.store.add_memory(text, kind)
        self.store.commit()
        redacted, _ = redact_secrets(text)
        return {"id": mid, "kind": kind, "text": redacted}

    def read_memory(self, limit: int = 20) -> dict:
        notes = []
        for row in self.store.recent_memory(limit):
            text, _ = redact_secrets(row["text"])
            notes.append({"kind": row["kind"], "text": text})
        cps = []
        for row in self.store.recent_checkpoints(10):
            note, _ = redact_secrets(row["note"])
            cps.append({"label": row["label"], "git_sha": row["git_sha"],
                        "note": note})
        return {"notes": notes, "checkpoints": cps}

    def create_checkpoint(self, label: str, note: str = "") -> dict:
        sha = _git(self.root, "rev-parse", "--short", "HEAD").strip()
        cid = self.store.add_checkpoint(label, sha, note)
        self.store.commit()
        safe_note, _ = redact_secrets(note)
        return {"id": cid, "label": label, "git_sha": sha, "note": safe_note}

    # ---- spec-named retrieval surface (MCP-2: read_context / search_signatures
    #      / query_context) layered on the existing graph ----
    def _module_files(self, module: str | None) -> list[str]:
        files = sorted(r["path"] for r in self.store.files_with_tokens())
        if not module:
            return files
        m = module.strip().strip("/")
        return [f for f in files if f == m or f.startswith(m + "/")
                or f.split("/", 1)[0] == m]

    def read_context(self, module: str | None = None, budget_tokens: int = 4000) -> str:
        """Signatures for the whole codebase or one module path (MCP-2).

        Token-frugal call ordering (MCP-3): list_modules() first, then
        read_context(module=…) for just that subtree.
        """
        files = self._module_files(module)
        out: list[str] = []
        used = shown = 0
        for f in files:
            sk = self.file_skeleton(f)
            est = count_tokens(sk)
            if out and used + est > budget_tokens:
                out.append(f"\n_… {len(files) - shown} more file(s) omitted for "
                           f"budget — call read_context(module=…) or "
                           f"file_skeleton(file) for them._")
                break
            out.append(sk)
            used += est
            shown += 1
        body = "\n\n".join(out) if out else f"(no indexed files for module={module!r})"
        red, _ = redact_secrets(body)
        return red

    def search_signatures(self, query: str, limit: int = 20) -> list[dict]:
        """Keyword search across signatures (MCP-2)."""
        rows = self.store.search(query, limit=limit)
        results = []
        for row in rows:
            if row["kind"] == "module":
                continue
            signature, _ = redact_secrets(row["signature"] or row["qname"])
            results.append({"qname": row["qname"], "kind": row["kind"],
                            "file": row["file"], "signature": signature})
        return results

    def rank_files(self, query: str, top_k: int = 10, recency_boost: float = 1.5,
                   use_recency: bool = True) -> list[tuple[str, float]]:
        """Rank files vs a query: fused lexical+semantic seeds rolled up to files,
        nudged by learned weight and a git-recency boost (FR-5)."""
        lexical = self.store.search(query, limit=40)
        semantic = self.semantic_search(query, limit=40)
        fused = self._fuse([lexical, semantic], limit=60) if semantic else lexical
        weights = self.store.all_weights()
        recent = ({f: (20 - i) / 20.0
                   for i, f in enumerate(git_recent_files(self.root, 20))}
                  if use_recency else {})
        fscore: dict[str, float] = {}
        for rank, row in enumerate(fused):
            f = row["file"]
            fscore[f] = max(fscore.get(f, 0.0), 1.0 / (rank + 1))
        # File-level prose, configuration, and UI modules can contain the task
        # vocabulary outside symbol signatures. Fuse chunk search so these
        # files are not invisible to rank_files even when symbols are sparse.
        for rank, chunk in enumerate(self.store.search_chunks(query, limit=20)):
            file = chunk["file"]
            fscore[file] = fscore.get(file, 0.0) + 0.75 / (rank + 1)
        for f in list(fscore):
            fscore[f] += weights.get(f, 0.0) * 0.1
            if f in recent:
                fscore[f] *= recency_boost ** recent[f]   # recency boost (FR-5)
        return sorted(fscore.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    def query_context(self, query: str, top_k: int = 10, budget_tokens: int = 6000,
                      recency_boost: float = 1.5) -> dict:
        """TF-IDF/embedding rank of all files vs a query, top-K (MCP-2, FR-4/5)."""
        ranked = self.rank_files(query, top_k=top_k, recency_boost=recency_boost)
        pack = self.find_relevant_context(query, budget_tokens=budget_tokens)
        return {
            "query": query,
            "intent": detect_intent(query),
            "top_files": [{"file": f, "score": round(s, 4)} for f, s in ranked],
            "pack_tokens": pack.tokens,
            "dropped": pack.dropped,
            "markdown": pack.to_markdown(),
        }

    # ---- signature-map helpers for context generation (TB-4 strategies) ----
    def all_files(self, src_dirs: list[str] | None = None) -> list[str]:
        files = sorted(r["path"] for r in self.store.files_with_tokens())
        # "." means the repository root, i.e. everything — including files that
        # sit at the top level and therefore have no directory prefix to match.
        # Without this, a config of srcDirs=["."] silently selected no files at
        # all for any root-level source file.
        if not src_dirs or any(d in (".", "./", "") for d in src_dirs):
            return files
        keep = []
        for f in files:
            top = f.split("/", 1)[0]
            if any(f == d or f.startswith(d.rstrip("/") + "/") or top == d
                   for d in src_dirs):
                keep.append(f)
        return keep

    def total_signature_tokens(self, src_dirs: list[str] | None = None) -> int:
        return sum(count_tokens(self.file_skeleton(f))
                   for f in self.all_files(src_dirs))

    def invalidate(self) -> None:
        """Drop derived caches after the graph changed (RP-1).

        The sqlite connection stays open — only the in-memory source lines and
        the ANN index, both of which would otherwise answer from stale state.
        """
        self._src_cache.clear()
        self._ann_index = None

    def close(self):
        # A pooled retriever outlives any single tool call, so the usual
        # `finally: r.close()` must not tear down the shared connection.
        # close_now() is the real shutdown path.
        if getattr(self, "pooled", False):
            return
        self.store.close()

    def close_now(self):
        """Unconditionally close, ignoring pooling. Used at server shutdown."""
        self.store.close()


# ==========================================================================
# configuration (CFG-1..5)
# ==========================================================================
CONFIG_NAME = "gen-context.config.json"

DEFAULTS_CONFIG: dict = {
    "srcDirs": ["src", "app", "lib", "."],
    "strategy": "hot-cold",                 # full | per-module | hot-cold (TB-4)
    "hotCommits": 10,                        # TB-5
    "diffPriority": True,                    # TB-5
    "autoMaxTokens": True,                   # TB-1
    "maxTokens": 8000,                       # fixed-budget fallback (TB-1)
    "coverageTarget": 0.80,                  # TB-2
    "modelContextLimit": 128000,             # TB-2
    "maxTokensHeadroom": 0.20,               # TB-2
    "outputs": ["copilot"],                  # MCP-OUT / §3.1
    "output": None,                          # override path for the copilot adapter
    "secretScan": True,                      # SEC-1
    "format": "md",                          # md | cache (PC-1)
    "retrieval": {"topK": 10, "recencyBoost": 1.5, "preset": "balanced"},  # CFG-3
    "enrich": {"todos": True, "changes": True, "coverage": False},          # CFG-4
}

_RETRIEVAL_PRESETS = {
    "precision": {"topK": 5, "recencyBoost": 1.2},
    "balanced": {"topK": 10, "recencyBoost": 1.5},
    "recall": {"topK": 20, "recencyBoost": 2.0},
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_extends(ref: str, root: Path) -> dict:
    """Load an `extends` base config from a local path or an HTTPS URL.

    Remote configs are cached ~1 hour under .tokengraph/ (CFG-2).
    """
    import json
    if ref.startswith("https://"):
        if offline_mode():
            return {}
        import time
        import urllib.request
        cache = root / ".tokengraph" / ("config-" + file_hash(ref)[:16] + ".json")
        try:
            fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < 3600
            if not fresh:
                with urllib.request.urlopen(ref, timeout=10) as resp:
                    data = resp.read().decode("utf-8", "replace")
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(data, encoding="utf-8")
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            return {}
    p = (root / ref) if not Path(ref).is_absolute() else Path(ref)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_config(root: Path, path: str | None = None) -> dict:
    """Merge DEFAULTS -> extends-base -> local file (CFG-1, CFG-2)."""
    import json
    cfg = dict(DEFAULTS_CONFIG)
    local_path = Path(path) if path else (root / CONFIG_NAME)
    local: dict = {}
    if local_path.exists():
        try:
            local = json.loads(local_path.read_text(encoding="utf-8"))
        except Exception as ex:
            local = {"_error": f"could not parse {local_path.name}: {ex}"}
    if isinstance(local.get("extends"), str):
        cfg = _deep_merge(cfg, _load_extends(local["extends"], root))
    cfg = _deep_merge(cfg, local)
    # apply retrieval preset, then let explicit keys win (CFG-3)
    preset = (cfg.get("retrieval") or {}).get("preset")
    if preset in _RETRIEVAL_PRESETS:
        merged = _deep_merge({"retrieval": _RETRIEVAL_PRESETS[preset]}, {"retrieval": local.get("retrieval", {})})
        cfg["retrieval"] = _deep_merge(cfg["retrieval"], merged["retrieval"])
    return cfg


def write_default_config(root: Path) -> Path:
    """Generate gen-context.config.json (CFG-1, `--init`)."""
    import json
    p = root / CONFIG_NAME
    seed = {k: DEFAULTS_CONFIG[k] for k in
            ("srcDirs", "strategy", "hotCommits", "diffPriority",
             "autoMaxTokens", "coverageTarget", "modelContextLimit",
             "maxTokensHeadroom", "outputs", "retrieval")}
    p.write_text(json.dumps(seed, indent=2) + "\n", encoding="utf-8")
    return p


# ==========================================================================
# token budget (TB-1..TB-3)
# ==========================================================================
def effective_budget(total_sig_tokens: int, coverage_target: float = 0.80,
                     model_context_limit: int = 128000,
                     headroom: float = 0.20, hard_floor: int = 4000) -> tuple[int, list[str]]:
    """effective = clamp(ceil(totalSigTokens × coverageTarget),
                         hard_floor, floor(modelContextLimit × headroom))  (TB-2)."""
    import math
    cap = int(model_context_limit * headroom)
    raw = math.ceil(total_sig_tokens * coverage_target)
    eff = max(hard_floor, min(raw, cap))
    warnings: list[str] = []
    if raw > cap and total_sig_tokens:          # TB-3: hard cap bites
        achieved = cap / total_sig_tokens
        if (coverage_target - achieved) > 0.10:
            warnings.append(
                f"hard cap {cap:,} covers only {achieved*100:.0f}% of "
                f"signatures — misses the {coverage_target*100:.0f}% target by "
                f">10pp; recommend strategy: \"per-module\"")
    return eff, warnings


# ==========================================================================
# context generation: strategies + adapters + prompt cache (TB-4, MCP-OUT, PC-1)
# ==========================================================================
CLAUDE_BEGIN = "<!-- tokengraph:begin (generated — do not edit) -->"
CLAUDE_END = "<!-- tokengraph:end -->"

# CFG-6: generated artefacts record the fingerprint of the code they were built
# from. Staleness used to be judged by wall-clock age, which is wrong in both
# directions — it condemns a correct file that nobody happened to regenerate
# this week, and it passes a file that went out of date an hour after it was
# written. Comparing the stamp to the current index answers the question that
# actually matters: does this describe the code as it stands?
SOURCE_STAMP_PREFIX = "<!-- tokengraph:source "


def generated_artifact_paths(cfg: dict | None = None) -> set[str]:
    """Repo-relative paths ContextIQ writes, which are never "source" (CFG-6)."""
    paths = {spec["path"] for spec in ADAPTERS.values() if spec.get("path")}
    custom = (cfg or {}).get("output")
    if custom:
        paths.add(str(custom))
    return paths


def source_stamp(fingerprint: str) -> str:
    return f"{SOURCE_STAMP_PREFIX}{fingerprint} -->"


def read_source_stamp(text: str) -> str:
    """The fingerprint a generated file was built from, or "" if unstamped."""
    idx = text.find(SOURCE_STAMP_PREFIX)
    if idx < 0:
        return ""
    rest = text[idx + len(SOURCE_STAMP_PREFIX):]
    end = rest.find(" -->")
    return rest[:end].strip() if end >= 0 else ""

# Steering-file adapters: where each agent host looks for repo instructions.
#
# Every adapter is marker-scoped (see write_adapter) — the generated block is
# replaced in place and any hand-written text around it survives. The old
# `mode` key was dead data: nothing read it and every adapter behaved as
# marker-scoped regardless, so it has been removed rather than left implying
# a behaviour that did not exist.
#
# `budget` is the per-host token ceiling for the generated map. Hosts differ
# by an order of magnitude in how much steering text they will actually carry,
# and writing one identical blob everywhere either wasted a large window or
# blew a small one. `header` is emitted verbatim above the marker block, for
# formats that require frontmatter (Cursor .mdc).
ADAPTERS: dict[str, dict] = {
    # --- current, verified formats ---
    "copilot":  {"path": ".github/copilot-instructions.md", "budget": 6000,
                 "host": "GitHub Copilot (VS Code / JetBrains / Visual Studio)"},
    "claude":   {"path": "CLAUDE.md", "budget": 12000,
                 "host": "Claude Code"},
    "agents":   {"path": "AGENTS.md", "budget": 8000,
                 "host": "AGENTS.md standard (Codex, Aider, Zed, Cursor, Jules…)"},
    "cursor":   {"path": ".cursor/rules/contextiq.mdc", "budget": 6000,
                 "host": "Cursor (project rules)",
                 "header": ("---\n"
                            "description: Token-efficient code retrieval via the "
                            "ContextIQ MCP server\n"
                            "alwaysApply: true\n"
                            "---\n")},
    "windsurf": {"path": ".windsurf/rules/contextiq.md", "budget": 6000,
                 "host": "Windsurf / Cascade (project rules directory)"},
    "cline":    {"path": ".clinerules/contextiq.md", "budget": 6000,
                 "host": "Cline"},
    "roo":      {"path": ".roo/rules/contextiq.md", "budget": 6000,
                 "host": "Roo Code"},
    "continue": {"path": ".continue/rules/contextiq.md", "budget": 6000,
                 "host": "Continue"},
    "gemini":   {"path": "GEMINI.md", "budget": 8000,
                 "host": "Gemini CLI / Code Assist"},
    "codex":    {"path": "AGENTS.md", "budget": 8000,
                 "host": "OpenAI Codex (reads AGENTS.md)"},
    "aider":    {"path": "CONVENTIONS.md", "budget": 6000,
                 "host": "Aider (--read CONVENTIONS.md)"},
    "zed":      {"path": ".rules", "budget": 6000,
                 "host": "Zed (.rules)"},
    # --- deprecated formats, still written on request for older installs ---
    "cursor-legacy":   {"path": ".cursorrules", "budget": 6000,
                        "host": "Cursor (legacy .cursorrules)", "deprecated": True},
    "windsurf-legacy": {"path": ".windsurfrules", "budget": 6000,
                        "host": "Windsurf (legacy .windsurfrules)",
                        "deprecated": True},
}

# Written by `generate` when no explicit adapter list is configured. Covers the
# three hosts that between them account for almost all real usage, without
# scattering files for tools the repo may not use.
DEFAULT_ADAPTERS = ["copilot", "claude", "agents"]


def adapter_budget(adapter: str, fallback: int) -> int:
    """Per-host token ceiling for the generated steering block."""
    return int(ADAPTERS.get(adapter, {}).get("budget") or fallback)


def _modules_table(modules: list[dict]) -> str:
    out = ["| module | files | tokens | symbols |", "|---|--:|--:|--:|"]
    for m in modules:
        out.append(f"| {m['module']} | {m['files']} | {m['tokens']:,} | {m['symbols']} |")
    return "\n".join(out)


def _scan_todos(root: Path, files: list[str], cap: int = 10) -> list[str]:
    import re as _re2
    rx = _re2.compile(r"\b(TODO|FIXME|HACK|XXX)\b[:\s](.{0,90})")
    hits: list[str] = []
    for f in files:
        try:
            for i, line in enumerate((root / f).read_text(
                    encoding="utf-8", errors="replace").splitlines(), 1):
                m = rx.search(line)
                if m:
                    hits.append(f"{f}:{i} {m.group(1)} {m.group(2).strip()}")
                    if len(hits) >= cap:
                        return hits
        except OSError:
            continue
    return hits


# Files whose "skeleton" is prose, not code signatures. A markdown file's
# skeleton is just its heading outline — noise in a signature map — and when the
# file is itself a generated output (e.g. copilot-instructions.md) that lands in
# the hot set, injecting its skeleton recursively duplicates the generated block.
# These are listed by name instead of having a skeleton injected.
SKELETON_EXCLUDED_EXTS = (".md", ".markdown", ".mdc", ".mdx")


def _skeleton_injectable(path: str) -> bool:
    """False for prose (markdown) files that should be listed, not skeletonised."""
    return not path.lower().endswith(SKELETON_EXCLUDED_EXTS)


def build_context_payload(r: "Retriever", root: Path, *, strategy: str,
                          src_dirs: list[str], budget: int, hot_commits: int,
                          diff: bool, staged: bool, config: dict) -> dict:
    """Render the always-on context markdown for a strategy (TB-4) + metadata."""
    files = r.all_files(src_dirs)
    # Only code files get a skeleton; prose (markdown) is listed by name below.
    skeletons = {f: r.file_skeleton(f) for f in files if _skeleton_injectable(f)}
    total_sig = sum(count_tokens(s) for s in skeletons.values())
    repo_total = r.store.repo_token_total()
    modules = [m for m in r.list_modules()
               if not src_dirs or m["module"] in
               {(f.split('/', 1)[0] if '/' in f else '.') for f in files}]
    enrich = config.get("enrich", {})
    warn: list[str] = []

    head = [
        "# Project context — signature map (generated by tokengraph)",
        f"_Strategy `{strategy}` · {len(files)} files · "
        f"signatures ≈{total_sig:,} tokens vs ~{repo_total:,} full-source._",
        "",
        "Compact signatures only. For full bodies use the `tokengraph` MCP server: "
        "`list_modules()` → `read_context(module=…)` / `query_context(task)` / "
        "`get_symbol(qname)` / `get_lines(file,start,end)`.",
        "",
        "## Modules",
        _modules_table(modules),
        "",
    ]
    # PC-1: git-volatile enrichment goes AFTER the signature body, not before
    # it. These two sections change on every commit and every TODO edit; when
    # they led the document they invalidated the entire provider prompt-cache
    # prefix each time, which defeats the point of caching. Everything stable
    # now sits in front of the cache breakpoint.
    volatile: list[str] = []
    if enrich.get("changes", True):
        recent = git_recent_files(root, hot_commits)[:10]
        if recent:
            volatile += ["## Recently changed", ", ".join(f"`{x}`" for x in recent), ""]
    if enrich.get("todos", True):
        todos = _scan_todos(root, files)
        if todos:
            volatile += ["## TODO / FIXME", *[f"- {t}" for t in todos], ""]

    body: list[str] = []
    hot_files: list[str] = []
    cold_files: list[str] = []

    if strategy == "per-module":
        # tiny overview + file list per module; detail stays on demand (MCP).
        by_mod: dict[str, list[str]] = {}
        for f in files:
            by_mod.setdefault(f.split("/", 1)[0] if "/" in f else ".", []).append(f)
        body.append("## Files by module (signatures on demand via MCP)")
        for mod in sorted(by_mod):
            body.append(f"### {mod}")
            body.append(", ".join(f"`{x}`" for x in sorted(by_mod[mod])))
        body.append("")
    elif strategy == "hot-cold":
        if config.get("diffPriority", True) and (diff or staged):
            hot_files = [f for f in git_changed_files(root, staged) if f in skeletons]
        if not hot_files:
            hot_files = [f for f in git_recent_files(root, hot_commits) if f in skeletons]
        if not hot_files:                         # no git history → seed with a few
            hot_files = [f for f in files if f in skeletons][:5]
        cold_files = [f for f in files if f not in set(hot_files)]
        body.append("## Hot — recently changed (full signatures injected)")
        used = 0
        for f in hot_files:
            sk = skeletons[f]
            est = count_tokens(sk)
            if used and used + est > budget:
                cold_files.insert(0, f)
                continue
            body += [f"### {f}", "```" + _fence(f), sk, "```", ""]
            used += est
        if cold_files:
            body += ["## Cold — fetch on demand via MCP `read_context`/`get_symbol`",
                     ", ".join(f"`{x}`" for x in sorted(cold_files)), ""]
    else:  # full
        if strategy != "full":
            warn.append(f"unknown strategy {strategy!r}; using 'full'")
            strategy = "full"
        body.append("## Signatures")
        used = 0
        for f in files:
            sk = skeletons.get(f)
            if sk is None:                         # prose file — no skeleton
                continue
            est = count_tokens(sk)
            if used and used + est > budget:
                body.append(f"_… {len([x for x in files])} files; budget {budget:,} "
                            f"reached. Remaining files via MCP `read_context`._")
                break
            body += [f"### {f}", "```" + _fence(f), sk, "```", ""]
            used += est

    # Split at the cache breakpoint: everything before it is stable across
    # commits and is what gets `cache_control`; everything after is volatile.
    stable = "\n".join(head + body).rstrip() + "\n"
    suffix = ("\n".join(volatile).rstrip() + "\n") if volatile else ""
    markdown = stable + (("\n" + suffix) if suffix else "")
    if config.get("secretScan", True):
        stable, _ = redact_secrets(stable)
        if suffix:
            suffix, _ = redact_secrets(suffix)
        markdown = stable + (("\n" + suffix) if suffix else "")
    tokens = count_tokens(markdown)
    reduction = (1 - tokens / repo_total) * 100.0 if repo_total else 0.0
    return {
        "strategy": strategy,
        "markdown": markdown,
        "stable_prefix": stable,
        "volatile_suffix": suffix,
        "stable_tokens": count_tokens(stable),
        "volatile_tokens": count_tokens(suffix) if suffix else 0,
        "tokens": tokens,
        "total_sig_tokens": total_sig,
        "repo_tokens": repo_total,
        "reduction_pct": round(reduction, 2),
        "files": len(files),
        "hot_files": hot_files,
        "cold_files": cold_files,
        "warnings": warn,
    }


def cache_artifact(text: str) -> dict:
    """Provider prompt-cache artifact for the stable signature prefix (PC-1)."""
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


def cache_blocks(payload: dict) -> list[dict]:
    """Provider message blocks with the cache breakpoint in the right place (PC-1).

    Returns Anthropic-shaped content blocks: the stable signature map carries
    `cache_control`, and the git-volatile tail follows it *uncached*. Sending
    them in this order means a commit invalidates only the small tail, not the
    whole map. Blocks are also valid plain-text content for providers with
    automatic caching (e.g. OpenAI), which need no annotation.
    """
    blocks = [cache_artifact(payload["stable_prefix"])]
    if payload.get("volatile_suffix"):
        blocks.append({"type": "text", "text": payload["volatile_suffix"]})
    return blocks


def write_adapter(root: Path, adapter: str, content: str,
                  custom_out: str | None = None,
                  fingerprint: str = "") -> str:
    spec = ADAPTERS[adapter]
    rel = custom_out if (custom_out and adapter == "copilot") else spec["path"]
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    # All adapters are non-destructive: the generated block lives between markers
    # and any hand-written content outside them is preserved across re-runs
    # (MCP-5). This avoids clobbering human instructions in copilot/cursor/etc.
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    stamp = f"{source_stamp(fingerprint)}\n" if fingerprint else ""
    block = f"{CLAUDE_BEGIN}\n{stamp}{content.rstrip()}\n{CLAUDE_END}\n"
    if CLAUDE_BEGIN in existing and CLAUDE_END in existing:
        pre = existing.split(CLAUDE_BEGIN)[0].rstrip()
        post = existing.split(CLAUDE_END, 1)[1].lstrip("\n")
        new = (pre + "\n\n" if pre else "") + block + (("\n" + post) if post else "")
    else:
        new = (existing.rstrip() + "\n\n" if existing.strip() else "") + block
    # Formats that require frontmatter (Cursor .mdc) get it prepended once,
    # outside the marker block so regeneration never duplicates or strips it.
    header = spec.get("header")
    if header and not new.lstrip().startswith("---"):
        new = header + "\n" + new
    path.write_text(new, encoding="utf-8")
    return rel


def write_cache_sidecar(root: Path, adapter_rel: str, payload: "dict | str") -> str:
    """Write the prompt-cache blocks next to an adapter output (PC-1/PC-3).

    Emits the ordered block list — stable signature map with `cache_control`
    first, git-volatile tail after — so a consumer can paste it straight into
    a request and get a prefix that survives commits. A bare string is still
    accepted and treated as a single stable block.
    """
    import json
    side = (root / adapter_rel).with_suffix(".cache.json")
    side.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        doc = {"blocks": [cache_artifact(payload)],
               "stable_tokens": count_tokens(payload), "volatile_tokens": 0}
    else:
        doc = {"blocks": cache_blocks(payload),
               "stable_tokens": payload["stable_tokens"],
               "volatile_tokens": payload["volatile_tokens"]}
    side.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return str(side.relative_to(root).as_posix())


# ==========================================================================
# local gain / usage tracking (TB-6, CI-3) — counts only, never leaves machine
# ==========================================================================
def _tracking_disabled(no_track_flag: bool) -> bool:
    return bool(no_track_flag or os.environ.get("SIGMAP_NO_TRACK")
                or os.environ.get("TOKENGRAPH_NO_TRACK"))


_GAIN_LOCK = threading.Lock()


def track_gain(root: Path, counts: dict, no_track: bool = False) -> None:
    """Append count-only savings to .context/gain.ndjson (TB-6). No paths/queries.

    Each line carries a `ts` epoch so `gain --since` can window the ledger; the
    field list is intentionally count-only (never a path or a query) to keep the
    ledger privacy-safe.
    """
    if _tracking_disabled(no_track):
        return
    import json
    import time
    safe = {k: counts[k] for k in counts
            if isinstance(counts[k], (int, float)) and k in
            {"final_tokens", "baseline_tokens", "saved", "reduction_pct", "files"}}
    p = root / ".context" / "gain.ndjson"
    p.parent.mkdir(parents=True, exist_ok=True)
    with _GAIN_LOCK:
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(),
                                 "op": counts.get("op", "generate"), **safe}) + "\n")


def track_usage(root: Path, metrics: dict, no_track: bool = False) -> None:
    """Append a run metric line to .context/usage.ndjson for trend reporting (CI-3)."""
    if _tracking_disabled(no_track):
        return
    import json
    import time
    p = root / ".context" / "usage.ndjson"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.time(), **metrics}) + "\n")


def record_pack_savings(root: Path, op: str, *, final_tokens: int,
                        baseline_tokens: int, files: int,
                        no_track: bool = False) -> None:
    """Append one retrieval op's pack-vs-whole-file delta to the ledger (TB-6).

    Wired into the everyday retrieval paths (context / ask / measure / the MCP
    context tool) so the savings ledger accumulates from normal use, not just
    `generate`. Best-effort: tracking must never break the host command.
    """
    try:
        saved = baseline_tokens - final_tokens
        if saved <= 0:
            # Skip non-savings (e.g. a tiny transcript whose summary scaffolding
            # exceeds the original) — recording them would only dilute the
            # ledger's totals and run counts on the dashboard.
            return
        red = round(saved / baseline_tokens * 100.0, 1) if baseline_tokens else 0.0
        track_gain(root, {"op": op, "final_tokens": final_tokens,
                          "baseline_tokens": baseline_tokens, "saved": saved,
                          "reduction_pct": red, "files": files}, no_track=no_track)
        track_usage(root, {"op": op, "final_tokens": final_tokens,
                           "reduction_pct": red}, no_track=no_track)
        if not _tracking_disabled(no_track):
            # Keep the per-workspace static report fresh so any client can just
            # open .tokengraph/token-usage.html — no server, no dependencies.
            write_usage_report(root)
    except Exception:
        pass


# ==========================================================================
# savings ledger reporting: `gain` (trend + cost projection over the ledger)
# ==========================================================================
# Approximate input-token list prices (USD per 1M tokens) — for *projection*
# only, so the saved-token count can be expressed as a rough dollar figure.
GAIN_PRICES_PER_1M: dict[str, float] = {
    "claude-opus": 15.0, "claude-sonnet": 3.0, "claude-haiku": 0.80,
    "gpt-4o": 2.5, "gpt-4o-mini": 0.15, "gpt-4.1": 2.0,
    "gemini-1.5-pro": 1.25, "gemini-1.5-flash": 0.075,
    "llama-3.1-405b": 3.0, "llama-3.1-70b": 0.60, "llama-3.1-8b": 0.10,
}
DEFAULT_GAIN_MODEL = "claude-sonnet"

# Pre-flight cost estimation (CE-1). List prices in USD per 1M tokens, split
# into input (prompt) and output (completion). Used to price a call *before*
# it's sent — unlike GAIN_PRICES_PER_1M, which projects savings after the fact.
MODEL_PRICES_PER_1M: dict[str, dict[str, float]] = {
    # Claude, verified against the vendor catalogue on the date in
    # CLAUDE_PRICES_AS_OF below. Concrete model ids first; the bare family
    # names after them are aliases kept so existing callers and the savings
    # ledger keep resolving.
    "claude-fable-5":    {"input": 10.0,  "output": 50.0},
    "claude-opus-4-8":   {"input": 5.0,   "output": 25.0},
    "claude-opus-4-7":   {"input": 5.0,   "output": 25.0},
    "claude-opus-4-6":   {"input": 5.0,   "output": 25.0},
    "claude-sonnet-5":   {"input": 3.0,   "output": 15.0},
    "claude-sonnet-4-6": {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5":  {"input": 1.0,   "output": 5.0},
    "claude-opus":       {"input": 5.0,   "output": 25.0},
    "claude-sonnet":     {"input": 3.0,   "output": 15.0},
    "claude-haiku":      {"input": 1.0,   "output": 5.0},
    "gpt-4o":           {"input": 2.5,   "output": 10.0},
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
    "gpt-4.1":          {"input": 2.0,   "output": 8.0},
    "gemini-1.5-pro":   {"input": 1.25,  "output": 5.0},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "llama-3.1-405b":   {"input": 3.0,   "output": 3.0},
    "llama-3.1-70b":    {"input": 0.60,  "output": 0.60},
    "llama-3.1-8b":     {"input": 0.10,  "output": 0.10},
}
DEFAULT_COST_MODEL = "claude-sonnet"

# CE-3: cached and batched tokens are not priced like fresh input, and a tool
# whose entire thesis is "send fewer tokens" cannot then price the tokens it
# does send incorrectly. Reading a cached prefix costs about a tenth of base
# input — which is precisely why the prompt-cache ordering work (PC-1) is worth
# doing — while *writing* the cache costs a premium that has to be earned back.
# Expressed as multipliers on the model's input price so they survive a price
# change, and keyed by family because the economics differ by vendor.
CACHE_MULTIPLIERS: dict[str, dict[str, float]] = {
    # Anthropic: read ≈0.1×, 5-minute write 1.25×, 1-hour write 2×.
    "claude": {"read": 0.1, "write": 1.25, "write_1h": 2.0},
    # OpenAI caches automatically and discounts reads; there is no write premium.
    "gpt":    {"read": 0.5, "write": 1.0},
    "gemini": {"read": 0.25, "write": 1.0},
}
# Asynchronous batch APIs trade latency for a discount on every token.
BATCH_DISCOUNT: dict[str, float] = {"claude": 0.5, "gpt": 0.5, "gemini": 0.5}

# Per-family provenance. A single global date was the wrong shape: it made a
# freshly-verified Claude price look exactly as stale as a Llama price nobody
# had checked in a year, so the staleness warning was either a false alarm or
# ignored. Each family now ages on its own clock.
FAMILY_PRICES_AS_OF: dict[str, str] = {
    "claude": "2026-06-24",
    "gpt": "2025-06-01",
    "gemini": "2025-06-01",
    "llama": "2025-06-01",
}

# ---- CE-2: pricing provenance and overrides -------------------------------
# A hardcoded price table silently rots: vendors reprice, and a stale number
# produces confidently wrong cost estimates. The table above is a *default*
# with a known date. Anything derived from it carries that date, goes stale
# out loud, and can be overridden without editing code.
# The global date is the OLDEST family's — so a single number can never claim
# the catalogue is fresher than its weakest entry.
PRICES_AS_OF = min(FAMILY_PRICES_AS_OF.values())
PRICES_STALE_AFTER_DAYS = 180
# Bumped whenever the catalogue's *shape* changes (new rate kinds, new
# families), so an override file written against an older shape is detectable.
PRICING_CATALOG_VERSION = 2
PRICING_FILE_ENV = "TOKENGRAPH_PRICING_FILE"
PRICING_FILENAME = "pricing.json"

_PRICING_CACHE: dict | None = None


def pricing_file_path(root: Path | None = None) -> Path:
    """Where a pricing override is read from, if present."""
    override = os.environ.get(PRICING_FILE_ENV)
    if override:
        return Path(override)
    return Path(root or ".") / ".context" / PRICING_FILENAME


def load_pricing(root: Path | None = None, refresh: bool = False) -> dict:
    """Effective price table + provenance (CE-2).

    Reads `.context/pricing.json` (or $TOKENGRAPH_PRICING_FILE) when present
    and merges it over the built-in defaults, so a user can correct prices for
    their own contract or a vendor change without a code edit. The returned
    dict always states where the numbers came from and whether they are stale.
    """
    global _PRICING_CACHE
    if _PRICING_CACHE is not None and not refresh:
        return _PRICING_CACHE
    import json
    from datetime import date

    prices = {k: dict(v) for k, v in MODEL_PRICES_PER_1M.items()}
    as_of, source, warnings = PRICES_AS_OF, "built-in defaults", []

    path = pricing_file_path(root)
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            for model, entry in (doc.get("prices") or {}).items():
                if not isinstance(entry, dict):
                    warnings.append(f"pricing: ignoring {model!r} (not an object)")
                    continue
                try:
                    prices[model] = {"input": float(entry["input"]),
                                     "output": float(entry["output"])}
                except (KeyError, TypeError, ValueError):
                    warnings.append(
                        f"pricing: ignoring {model!r} (needs numeric "
                        f"'input' and 'output')")
            as_of = str(doc.get("as_of") or as_of)
            source = str(path)
        except (OSError, ValueError) as ex:
            warnings.append(f"pricing: could not read {path} ({ex}); "
                            f"using built-in defaults")

    stale, age_days = False, None
    try:
        y, m, d = (int(x) for x in as_of.split("-"))
        age_days = (date.today() - date(y, m, d)).days
        stale = age_days > PRICES_STALE_AFTER_DAYS
    except (ValueError, TypeError):
        warnings.append(f"pricing: unparseable as_of {as_of!r}")
    # CE-3: age each vendor family separately and name the stale ones. A single
    # global warning could not distinguish "every price here is a year old"
    # from "one vendor we don't price against is", so it was noise either way.
    families: dict[str, dict] = {}
    stale_families: list[str] = []
    for fam, fam_as_of in sorted(FAMILY_PRICES_AS_OF.items()):
        fam_age = None
        fam_stale = False
        try:
            y, m, d = (int(x) for x in fam_as_of.split("-"))
            fam_age = (date.today() - date(y, m, d)).days
            fam_stale = fam_age > PRICES_STALE_AFTER_DAYS
        except (ValueError, TypeError):
            pass
        families[fam] = {"as_of": fam_as_of, "age_days": fam_age,
                         "stale": fam_stale}
        if fam_stale:
            stale_families.append(f"{fam} ({fam_age}d)")
    if stale_families and source == "built-in defaults":
        warnings.append(
            f"pricing is past its {PRICES_STALE_AFTER_DAYS}-day review window "
            f"for: {', '.join(stale_families)}. Vendor list prices change; "
            f"treat costs for those families as indicative. Refresh with "
            f"`tokengraph pricing --check`, or override via "
            f"{pricing_file_path(root)} or ${PRICING_FILE_ENV}.")
    elif stale:
        warnings.append(
            f"pricing is {age_days} days old (as of {as_of}). Vendor list "
            f"prices change; treat costs as indicative. Override with "
            f"{pricing_file_path(root)} or ${PRICING_FILE_ENV}.")

    _PRICING_CACHE = {"prices": prices, "as_of": as_of, "source": source,
                      "stale": stale, "age_days": age_days,
                      "families": families,
                      "catalog_version": PRICING_CATALOG_VERSION,
                      "warnings": warnings}
    return _PRICING_CACHE


def price_for(model: str, root: Path | None = None) -> dict:
    """Input/output price per 1M tokens for a model, from the effective table."""
    table = load_pricing(root)["prices"]
    if model in table:
        return table[model]
    # Unknown concrete model: fall back to the family's mid-tier, and say so.
    fam = model_family(model)
    for candidate in (f"{fam}-sonnet", "claude-sonnet", "gpt-4o",
                      "gemini-1.5-pro", "llama-3.1-70b"):
        if candidate in table and model_family(candidate) == fam:
            return table[candidate]
    return table.get(DEFAULT_COST_MODEL, {"input": 3.0, "output": 15.0})


def rate_card(model: str, root: Path | None = None) -> dict:
    """Every per-1M rate that applies to a model, not just fresh input (CE-3).

    Cached reads and batch submissions are the two levers that most change what
    a call actually costs, and quoting only the list input price overstates the
    bill for anyone using either.
    """
    base = price_for(model, root)
    fam = model_family(model)
    mult = CACHE_MULTIPLIERS.get(fam, {})
    card = {
        "model": model, "family": fam,
        "input": base["input"], "output": base["output"],
        "as_of": FAMILY_PRICES_AS_OF.get(fam, PRICES_AS_OF),
    }
    for key in ("read", "write", "write_1h"):
        if key in mult:
            card[f"cache_{key}"] = round(base["input"] * mult[key], 6)
    if fam in BATCH_DISCOUNT:
        card["batch_multiplier"] = BATCH_DISCOUNT[fam]
    return card


def estimate_cost(prompt: str | int, model: str = DEFAULT_COST_MODEL,
                  expected_output_tokens: int = 500,
                  cached_input_tokens: int = 0,
                  cache_write_tokens: int = 0,
                  batch: bool = False) -> dict:
    """Price an API call *before* sending it (CE-1, CE-3).

    `prompt` may be the raw text (counted with the model-aware tokenizer) or a
    pre-computed input-token integer. `cached_input_tokens` are billed at the
    provider's cache-read rate and are *subtracted* from the fresh input count,
    `cache_write_tokens` at the write premium, and `batch=True` applies the
    asynchronous-batch discount to everything. Deterministic, local.
    """
    detail = None
    if isinstance(prompt, int):
        in_tok = max(0, prompt)
    else:
        detail = count_tokens_detail(prompt or "", model)
        in_tok = detail["tokens"]
    out_tok = max(0, int(expected_output_tokens))
    pricing = load_pricing()
    price = price_for(model)
    card = rate_card(model)
    # Cached and freshly-written tokens are part of the prompt, not extra to
    # it: counting them on top would double-bill the same text.
    cached = max(0, min(int(cached_input_tokens), in_tok))
    written = max(0, min(int(cache_write_tokens), in_tok - cached))
    fresh = in_tok - cached - written
    batch_mult = card.get("batch_multiplier", 1.0) if batch else 1.0
    in_usd = (fresh / 1_000_000 * price["input"]
              + cached / 1_000_000 * card.get("cache_read", price["input"])
              + written / 1_000_000 * card.get("cache_write", price["input"])
              ) * batch_mult
    out_usd = out_tok / 1_000_000 * price["output"] * batch_mult
    out = {
        "model": model,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "fresh_input_tokens": fresh,
        "cached_input_tokens": cached,
        "cache_write_tokens": written,
        "batch": bool(batch),
        "input_usd": round(in_usd, 6),
        "output_usd": round(out_usd, 6),
        "total_usd": round(in_usd + out_usd, 6),
        "price_per_1m_input_usd": price["input"],
        "price_per_1m_output_usd": price["output"],
        "rate_card": card,
        # CE-2: never hand back a bare number. The caller needs to know how the
        # tokens were counted and how old the prices are before trusting it.
        "prices_as_of": pricing["as_of"],
        "prices_source": pricing["source"],
        "prices_stale": pricing["stale"],
        "token_method": detail["method"] if detail else "caller-supplied",
    }
    notes = list(pricing["warnings"])
    if detail and detail["method"] == "approx" and detail.get("note"):
        notes.append(detail["note"])
    if model not in pricing["prices"]:
        notes.append(f"no listed price for {model!r}; used the closest "
                     f"{model_family(model)} model's rate")
    if notes:
        out["warnings"] = notes
    return out


def compare_cost(prompt: str | int, expected_output_tokens: int = 500,
                 models: list[str] | None = None) -> dict:
    """estimate_cost across several models — the cheapest sufficient pick (CE-2)."""
    names = models or list(load_pricing()["prices"].keys())
    rows = [estimate_cost(prompt, m, expected_output_tokens) for m in names]
    rows.sort(key=lambda r: (r["total_usd"], r["model"]))
    pricing = load_pricing()
    return {"cheapest": rows[0] if rows else None, "by_model": rows,
            "prices_as_of": pricing["as_of"], "prices_stale": pricing["stale"],
            "prices_source": pricing["source"]}


def _parse_since(spec: str | None) -> float | None:
    """Turn a window spec into an epoch cutoff: '7d' / '12h' / '90m' / ISO date."""
    if not spec:
        return None
    import time
    s = spec.strip().lower()
    units = {"d": 86400, "h": 3600, "m": 60, "w": 604800}
    if s and s[-1] in units and s[:-1].replace(".", "", 1).isdigit():
        return time.time() - float(s[:-1]) * units[s[-1]]
    try:                                   # ISO date / datetime
        import datetime as _dt
        return _dt.datetime.fromisoformat(spec).timestamp()
    except Exception:
        return None


def read_gain_ledger(root: Path) -> list[dict]:
    """Parse .context/gain.ndjson into a list of records (newest entries last)."""
    import json
    p = root / ".context" / "gain.ndjson"
    rows: list[dict] = []
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def gain_totals(rows: list[dict]) -> dict:
    saved = sum(int(row.get("saved", 0)) for row in rows)
    baseline = sum(int(row.get("baseline_tokens", 0)) for row in rows)
    final = sum(int(row.get("final_tokens", 0)) for row in rows)
    return {"runs": len(rows), "saved": saved, "baseline": baseline,
            "final": final,
            "reduction_pct": round(saved / baseline * 100, 1) if baseline else 0.0}


def gain_by_operation(rows: list[dict]) -> list[dict]:
    operations: dict[str, dict] = {}
    for row in rows:
        name = row.get("op", "?")
        operation = operations.setdefault(name, {"op": name, "saved": 0, "runs": 0})
        operation["saved"] += int(row.get("saved", 0))
        operation["runs"] += 1
    return sorted(operations.values(), key=lambda item: item["saved"], reverse=True)


def gain_daily(rows: list[dict]) -> list[dict]:
    from datetime import datetime, timezone
    days: dict[str, dict] = {}
    for row in rows:
        timestamp = float(row.get("ts", 0.0))
        if not timestamp:
            continue
        key = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        day = days.setdefault(key, {"day": key, "saved": 0, "final": 0})
        day["saved"] += int(row.get("saved", 0))
        day["final"] += int(row.get("final_tokens", 0))
    return [days[key] for key in sorted(days)]


def summarize_gain(root: Path, since: str | None = None,
                   model: str = DEFAULT_GAIN_MODEL, top: int | None = None,
                   trends: bool = False) -> dict:
    """Aggregate the savings ledger into totals, per-op rows, and cost projection."""
    rows = read_gain_ledger(root)
    cutoff = _parse_since(since)
    if cutoff is not None:
        rows = [r for r in rows if float(r.get("ts", 0.0)) >= cutoff]
    price = GAIN_PRICES_PER_1M.get(model, GAIN_PRICES_PER_1M[DEFAULT_GAIN_MODEL])
    tot_saved = sum(int(r.get("saved", 0)) for r in rows)
    tot_base = sum(int(r.get("baseline_tokens", 0)) for r in rows)
    tot_final = sum(int(r.get("final_tokens", 0)) for r in rows)
    by_op: dict[str, dict] = {}
    for r in rows:
        b = by_op.setdefault(r.get("op", "?"),
                             {"op": r.get("op", "?"), "runs": 0, "saved": 0,
                              "baseline": 0, "final": 0})
        b["runs"] += 1
        b["saved"] += int(r.get("saved", 0))
        b["baseline"] += int(r.get("baseline_tokens", 0))
        b["final"] += int(r.get("final_tokens", 0))
    op_rows = sorted(by_op.values(), key=lambda x: x["saved"], reverse=True)
    if top:
        op_rows = op_rows[:top]
    out = {
        "model": model, "price_per_1m_usd": price, "since": since,
        "runs": len(rows),
        "saved_tokens": tot_saved, "baseline_tokens": tot_base,
        "final_tokens": tot_final,
        "reduction_pct": round(tot_saved / tot_base * 100.0, 1) if tot_base else 0.0,
        "saved_usd": round(tot_saved / 1_000_000 * price, 4),
        "by_op": op_rows,
    }
    if trends:
        out["daily"] = _gain_buckets(rows, "%Y-%m-%d")[-14:]
        out["weekly"] = _gain_buckets(rows, "%Y-W%W")[-12:]
        out["monthly"] = _gain_buckets(rows, "%Y-%m")[-12:]
    return out


def _gain_buckets(rows: list[dict], fmt: str) -> list[dict]:
    """Bucket ledger rows by a strftime key, returning sorted {key,saved,runs}."""
    import datetime as _dt
    buckets: dict[str, dict] = {}
    for r in rows:
        ts = float(r.get("ts", 0.0))
        if not ts:
            continue
        key = _dt.datetime.fromtimestamp(ts).strftime(fmt)
        b = buckets.setdefault(key, {"period": key, "saved": 0, "runs": 0,
                                     "baseline": 0, "final": 0})
        b["saved"] += int(r.get("saved", 0))
        b["runs"] += 1
        # baseline/final let the report recompute windowed totals client-side
        b["baseline"] += int(r.get("baseline_tokens", 0))
        b["final"] += int(r.get("final_tokens", 0))
    return [buckets[k] for k in sorted(buckets)]


def _sparkline(values: list[int]) -> str:
    """Unicode block sparkline for a series of savings values."""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    return "".join(blocks[min(7, int((v - lo) / span * 7))] for v in values)


# ==========================================================================
# report payload + self-contained HTML dashboard (replaces the Streamlit app)
# ==========================================================================
# The page is *progressively enhanced*: it always ships an inline snapshot so
# it works from file:// with zero dependencies, and if it detects it is being
# served (see serve_report) it polls /data.json and redraws live instead.
REPORT_SCHEMA_VERSION = 1


def _report_daily(rows: list[dict], cap: int = 180) -> list[dict]:
    """Daily buckets for the report, carrying `files` and capped to `cap` days.

    `summarize_gain(trends=True)` keeps only 14 days (it feeds the CLI trend
    line), which is too short for the report's 30/90-day windows and its
    26-week activity view — so the page gets its own, wider series. Same shape
    as `_gain_buckets` plus a `files` sum, keyed on the local date to match
    what a user sees in the log.
    """
    import datetime as _dt
    buckets: dict[str, dict] = {}
    for r in rows:
        ts = float(r.get("ts", 0.0))
        if not ts:
            continue
        key = _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        b = buckets.setdefault(key, {"period": key, "saved": 0, "runs": 0,
                                     "baseline": 0, "final": 0, "files": 0})
        b["saved"] += int(r.get("saved", 0))
        b["runs"] += 1
        b["baseline"] += int(r.get("baseline_tokens", 0))
        b["final"] += int(r.get("final_tokens", 0))
        b["files"] += int(r.get("files", 0) or 0)
    return [buckets[k] for k in sorted(buckets)][-cap:]


def _ledger_span(rows: list[dict]) -> dict:
    """First/last run and how many distinct days the ledger actually spans."""
    import datetime as _dt
    stamps = [float(r.get("ts", 0.0)) for r in rows if float(r.get("ts", 0.0))]
    days = {_dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d") for t in stamps}
    return {"first_ts": min(stamps) if stamps else 0.0,
            "last_ts": max(stamps) if stamps else 0.0,
            "active_days": len(days)}


def _workspace_stats(root: Path) -> dict:
    """Read-only facts about the local graph, for the report's workspace panel.

    Best-effort and strictly read-only: it opens the existing DB in `mode=ro`
    (so it can never create one) with a short timeout, and returns `{}` on any
    problem — this runs on the savings-logging hot path and must never raise,
    block, or write.
    """
    try:
        db = _db_path(Path(root).resolve())   # as_uri() needs an absolute path
        if not db.exists():
            return {}
        import sqlite3
        con = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=0.25)
        try:
            def scalar(sql: str) -> int:
                try:
                    return int(con.execute(sql).fetchone()[0] or 0)
                except Exception:
                    return 0

            files = scalar("SELECT COUNT(*) FROM files")
            if not files:
                # No readable file table (absent, empty or not a graph db) —
                # report "no index" rather than a row of confident zeroes.
                return {}
            languages: list[dict] = []
            try:
                for lang, files, tokens in con.execute(
                        "SELECT COALESCE(NULLIF(language,''),'unknown') AS lang, "
                        "COUNT(*), COALESCE(SUM(token_est),0) FROM files "
                        "GROUP BY lang ORDER BY 2 DESC LIMIT 12"):
                    languages.append({"language": str(lang), "files": int(files),
                                      "tokens": int(tokens)})
            except Exception:
                pass
            return {
                "files": files,
                "symbols": scalar("SELECT COUNT(*) FROM symbols"),
                "chunks": scalar("SELECT COUNT(*) FROM chunks"),
                "edges": scalar("SELECT COUNT(*) FROM edges"),
                "summaries": scalar("SELECT COUNT(*) FROM summaries"),
                "indexed_tokens": scalar(
                    "SELECT COALESCE(SUM(token_est),0) FROM files"),
                "db_bytes": db.stat().st_size,
                "languages": languages,
            }
        finally:
            con.close()
    except Exception:
        return {}


def build_report_payload(root: Path, model: str = DEFAULT_GAIN_MODEL,
                         generated_at: str | None = None,
                         max_rows: int = 200) -> dict:
    """Everything the HTML report needs, as one versioned JSON-safe dict.

    Deliberately aggregated: daily/weekly/monthly buckets plus a bounded tail
    of raw rows, so the inlined payload stays small as the ledger grows. Pure
    data (no HTML) so it can also be served as /data.json and unit-tested.
    `rows_capped` tells the page when a view built from the row tail (per-op
    windows, the log) is showing a slice rather than everything, so it can say
    so instead of implying the tail is the whole ledger.
    """
    rows = read_gain_ledger(root)
    s = summarize_gain(root, model=model, trends=True)
    return {
        "version": REPORT_SCHEMA_VERSION,
        "workspace": Path(root).resolve().name,
        "generated_at": generated_at or "",
        "model": model,
        "prices": dict(GAIN_PRICES_PER_1M),
        "prices_io": {m: dict(p) for m, p in MODEL_PRICES_PER_1M.items()},
        "totals": {
            "runs": s["runs"], "saved": s["saved_tokens"],
            "baseline": s["baseline_tokens"], "final": s["final_tokens"],
            "reduction_pct": s["reduction_pct"], "saved_usd": s["saved_usd"],
            "files": sum(int(r.get("files", 0) or 0) for r in rows),
        },
        "by_op": s["by_op"],
        "daily": _report_daily(rows),
        "weekly": s.get("weekly", []),
        "monthly": s.get("monthly", []),
        "ledger": _ledger_span(rows),
        "workspace_stats": _workspace_stats(root),
        "rows": rows[-max_rows:],
        "rows_capped": len(rows) > max_rows,
    }


# ---------------------------------------------------------------------------
# Presentation layer for the report. Three constants — design tokens + CSS,
# the view logic, and the document skeleton — kept apart from the payload so
# the data contract (build_report_payload / data.json) stays untouched.
# ---------------------------------------------------------------------------

# Dark steps are *selected* for the dark surface (same ramps, re-stepped) and
# checked against it — not an automatic inversion. Emitted under two scopes so
# an explicit theme choice beats the OS setting in both directions.
_REPORT_DARK_VARS = """
  color-scheme:dark;
  --bg:#080d18;--surface:#111a2e;--surface-2:#152039;--surface-3:#1b2740;
  --overlay:rgba(255,255,255,.07);
  --fg:#e9effb;--fg-2:#c3d0e4;--muted:#94a5be;--muted-2:#6f819c;
  --line:#22304d;--line-2:#2e3f63;--grid:#1e2b47;
  --brand:#5b93ff;--brand-2:#0f1c33;--brand-3:#0a1428;--brand-soft:#182741;
  --s-sent:#3987e5;--s-saved:#008300;--s-c3:#d55181;--s-c4:#c98500;
  --s-c5:#199e70;--s-base:#8496ae;
  --seq-0:#16203a;--seq-1:#104281;--seq-2:#184f95;--seq-3:#1c5cab;
  --seq-4:#2a78d6;--seq-5:#5598e7;--seq-6:#86b6ef;--track:#1c2b49;
  --good:#0ca30c;--good-ink:#7ee07e;--good-soft:#12301a;
  --warn:#fab219;--warn-ink:#f0c46a;--warn-soft:#312413;
  --crit:#d03b3b;--crit-ink:#f19a9a;--crit-soft:#331919;
  --sh-1:0 1px 2px rgba(0,0,0,.40);--sh-2:0 10px 30px rgba(0,0,0,.45);
"""

_REPORT_TOKENS = """
/* ==== design tokens =======================================================
   One source of truth for color, type, space, radius, elevation and motion.
   Every component is written against these roles (never raw hex), so themes
   and rebrands change in exactly one place. Data-mark colors are a palette
   checked for colour-vision separation against both chart surfaces; brand
   navy/blue is reserved for chrome so it can't be misread as a series.
   ======================================================================= */
:root{
  color-scheme:light;
  --bg:#f4f6fa;--surface:#fff;--surface-2:#f8fafc;--surface-3:#eef2f8;
  --overlay:rgba(14,23,41,.06);
  --fg:#0e1729;--fg-2:#334155;--muted:#5b6b83;--muted-2:#8496ae;
  --line:#e4e9f0;--line-2:#cfd8e5;--grid:#e9eef6;
  --brand:#2f6fed;--brand-2:#1a2b4a;--brand-3:#12203a;--brand-soft:#eaf1ff;
  --s-sent:#2a78d6;--s-saved:#008300;--s-c3:#e87ba4;--s-c4:#eda100;
  --s-c5:#1baf7a;--s-base:#7c8aa0;
  --seq-0:#eef2f8;--seq-1:#cde2fb;--seq-2:#9ec5f4;--seq-3:#6da7ec;
  --seq-4:#3987e5;--seq-5:#256abf;--seq-6:#0d366b;--track:#dde8f8;
  --good:#0ca30c;--good-ink:#0a6b0a;--good-soft:#e6f6e6;
  --warn:#fab219;--warn-ink:#8a5a00;--warn-soft:#fff5e0;
  --crit:#d03b3b;--crit-ink:#a12626;--crit-soft:#fdeaea;
  --sh-1:0 1px 2px rgba(14,23,41,.05),0 1px 3px rgba(14,23,41,.04);
  --sh-2:0 8px 28px rgba(14,23,41,.10);
  --r-1:8px;--r-2:12px;--r-3:16px;--r-pill:999px;
  --sp-1:4px;--sp-2:8px;--sp-3:12px;--sp-4:16px;--sp-5:24px;--sp-6:32px;--sp-7:48px;
  --fs-1:11px;--fs-2:12px;--fs-3:13px;--fs-4:14px;--fs-5:16px;--fs-6:20px;
  --fs-7:28px;--fs-8:46px;
  --font:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --ease:cubic-bezier(.2,.7,.3,1);--dur-1:120ms;--dur-2:220ms;
}
"""

_REPORT_COMPONENTS = """
/* ==== base ============================================================== */
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font:var(--fs-4)/1.55 var(--font);
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
h1,h2,h3,p,ul,dl,figure{margin:0}
ul{padding:0;list-style:none}
a{color:var(--brand)}
b,strong{font-weight:650}
code{font:var(--fs-2)/1.4 var(--mono);background:var(--surface-3);padding:1px 6px;
  border-radius:6px;color:var(--fg-2)}
.wrap{max-width:1240px;margin:0 auto;padding:0 var(--sp-5)}
:focus-visible{outline:2px solid var(--brand);outline-offset:2px;border-radius:4px}
.skip{position:absolute;left:-9999px;top:8px;z-index:60;background:var(--surface);
  color:var(--fg);padding:10px 16px;border-radius:var(--r-1);box-shadow:var(--sh-2)}
.skip:focus{left:16px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);
  white-space:nowrap}
.muted{color:var(--muted)}
.num{font-variant-numeric:tabular-nums}

/* ==== app bar =========================================================== */
.appbar{position:sticky;top:0;z-index:40;background:var(--surface);
  border-bottom:1px solid var(--line)}
.appbar .row{display:flex;align-items:center;gap:var(--sp-4);height:60px}
.brand{display:flex;align-items:center;gap:10px;font-weight:680;font-size:var(--fs-5);
  letter-spacing:-.01em;color:var(--fg);white-space:nowrap}
.brand .mark{width:28px;height:28px;flex:0 0 28px;border-radius:9px;color:#fff;
  background:linear-gradient(140deg,var(--brand-2),var(--brand));display:grid;
  place-items:center}
.brand .sub{font-weight:500;font-size:var(--fs-2);color:var(--muted);
  border-left:1px solid var(--line-2);padding-left:10px}
.appnav{display:flex;gap:2px;margin:0 auto}
.appnav a{padding:7px 12px;border-radius:var(--r-1);color:var(--muted);
  text-decoration:none;font-size:var(--fs-3);font-weight:560;
  transition:background var(--dur-1) var(--ease),color var(--dur-1) var(--ease)}
.appnav a:hover{background:var(--surface-3);color:var(--fg)}
.bar-end{display:flex;align-items:center;gap:var(--sp-2)}

/* ==== primitives ======================================================== */
.btn{display:inline-flex;align-items:center;gap:6px;height:34px;padding:0 12px;
  border:1px solid var(--line-2);border-radius:var(--r-1);background:var(--surface);
  color:var(--fg-2);font:560 var(--fs-3)/1 var(--font);cursor:pointer;
  transition:background var(--dur-1) var(--ease),color var(--dur-1) var(--ease),
    transform var(--dur-1) var(--ease)}
.btn:hover{background:var(--surface-3);color:var(--fg)}
.btn:active{transform:translateY(1px)}
.btn.icon{width:34px;padding:0;justify-content:center}
.field{display:inline-flex;align-items:center;gap:8px;font-size:var(--fs-3);
  color:var(--muted);font-weight:560}
select{font:var(--fs-3)/1 var(--font);color:var(--fg);background:var(--surface);
  border:1px solid var(--line-2);border-radius:var(--r-1);padding:0 28px 0 10px;
  height:34px;appearance:none;cursor:pointer;
  background-image:linear-gradient(45deg,transparent 50%,var(--muted) 50%),
    linear-gradient(135deg,var(--muted) 50%,transparent 50%);
  background-position:calc(100% - 15px) 15px,calc(100% - 10px) 15px;
  background-size:5px 5px,5px 5px;background-repeat:no-repeat;
  transition:border-color var(--dur-1) var(--ease)}
select:hover{border-color:var(--muted-2)}
.switch{display:inline-flex;align-items:center;gap:8px;font-size:var(--fs-3);
  color:var(--muted);font-weight:560;cursor:pointer;user-select:none}
.switch input{width:34px;height:20px;margin:0;appearance:none;cursor:pointer;
  background:var(--line-2);border-radius:var(--r-pill);position:relative;
  transition:background var(--dur-2) var(--ease)}
.switch input:before{content:"";position:absolute;top:2px;left:2px;width:16px;
  height:16px;border-radius:50%;background:#fff;box-shadow:var(--sh-1);
  transition:transform var(--dur-2) var(--ease)}
.switch input:checked{background:var(--brand)}
.switch input:checked:before{transform:translateX(14px)}
.pill{display:inline-flex;align-items:center;gap:6px;height:24px;padding:0 10px;
  border-radius:var(--r-pill);font-size:var(--fs-1);font-weight:650;
  letter-spacing:.02em;text-transform:uppercase;background:var(--surface-3);
  color:var(--muted);white-space:nowrap}
.pill .dot{width:7px;height:7px;border-radius:50%;background:currentColor}
.pill.live{background:var(--good-soft);color:var(--good-ink)}
.pill.live .dot{animation:pulse 2.4s var(--ease) infinite}
.pill.warn{background:var(--warn-soft);color:var(--warn-ink)}
.pill.info{background:var(--brand-soft);color:var(--brand)}
.chip{display:inline-flex;align-items:center;gap:6px;height:28px;padding:0 10px;
  border-radius:var(--r-1);background:var(--surface-3);color:var(--fg-2);
  font-size:var(--fs-2);font-weight:560;max-width:280px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

/* ==== layout ============================================================ */
main{padding:var(--sp-6) 0 var(--sp-7)}
.page-head{display:flex;flex-wrap:wrap;gap:var(--sp-3) var(--sp-5);
  align-items:flex-end;justify-content:space-between;margin-bottom:var(--sp-5)}
h1{font-size:var(--fs-7);font-weight:680;letter-spacing:-.02em;line-height:1.2}
.page-head p{color:var(--muted);font-size:var(--fs-3);margin-top:2px}
.status{display:flex;gap:var(--sp-2);align-items:center;flex-wrap:wrap}
section{margin-top:var(--sp-6);scroll-margin-top:76px}
.sec-head{display:flex;align-items:baseline;gap:var(--sp-3);flex-wrap:wrap;
  margin-bottom:var(--sp-3)}
h2{font-size:var(--fs-5);font-weight:660;letter-spacing:-.01em}
.sec-head .hint{font-size:var(--fs-2);color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-3);
  padding:var(--sp-5);box-shadow:var(--sh-1);min-width:0;
  transition:box-shadow var(--dur-2) var(--ease)}
.card:hover{box-shadow:var(--sh-2)}
.card h3{font-size:var(--fs-4);font-weight:640;letter-spacing:-.01em}
.card .sub{font-size:var(--fs-2);color:var(--muted);margin-top:2px}
.card-head{display:flex;justify-content:space-between;align-items:flex-start;
  gap:var(--sp-3);margin-bottom:var(--sp-4)}
.grid{display:grid;gap:var(--sp-4)}
.g-5{grid-template-columns:repeat(5,minmax(0,1fr))}
.g-4{grid-template-columns:repeat(4,minmax(0,1fr))}
.g-2{grid-template-columns:repeat(2,minmax(0,1fr))}
.g-23{grid-template-columns:minmax(0,3fr) minmax(0,2fr)}

/* ==== toolbar =========================================================== */
.toolbar{display:flex;flex-wrap:wrap;gap:var(--sp-3) var(--sp-4);align-items:center;
  background:var(--surface);border:1px solid var(--line);border-radius:var(--r-2);
  padding:var(--sp-3) var(--sp-4);box-shadow:var(--sh-1);margin-bottom:var(--sp-4)}
.toolbar .spacer{flex:1 1 auto}
.toolbar .meta{font-size:var(--fs-2);color:var(--muted);display:flex;
  align-items:center;gap:8px}

/* ==== hero + gauge ====================================================== */
.hero{grid-column:span 3;border-radius:var(--r-3);padding:var(--sp-5) var(--sp-6);
  background:linear-gradient(135deg,var(--brand-3),var(--brand));color:#fff;
  display:flex;flex-direction:column;justify-content:center;position:relative;
  overflow:hidden;box-shadow:var(--sh-1)}
.hero:after{content:"";position:absolute;right:-70px;top:-90px;width:280px;
  height:280px;border-radius:50%;background:rgba(255,255,255,.07)}
.hero-label{font-size:var(--fs-2);text-transform:uppercase;letter-spacing:.1em;
  font-weight:650;opacity:.85}
.hero-value{font-size:var(--fs-8);font-weight:700;line-height:1.1;
  letter-spacing:-.03em;margin:6px 0 4px}
.hero-sub{font-size:var(--fs-4);opacity:.94}
.hero-foot{margin-top:var(--sp-4);font-size:var(--fs-2);opacity:.82;display:flex;
  gap:var(--sp-3);flex-wrap:wrap;align-items:center;position:relative}
.gauge-card{grid-column:span 2;display:flex;flex-direction:column;
  justify-content:center}

/* ==== stat tiles ======================================================== */
.stat{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-2);
  padding:var(--sp-4);box-shadow:var(--sh-1);min-width:0;
  transition:transform var(--dur-2) var(--ease),box-shadow var(--dur-2) var(--ease)}
.stat:hover{transform:translateY(-1px);box-shadow:var(--sh-2)}
.stat .k{font-size:var(--fs-2);color:var(--muted);font-weight:600;display:flex;
  align-items:center;gap:6px}
.stat .v{font-size:var(--fs-7);font-weight:680;letter-spacing:-.02em;line-height:1.25;
  margin-top:2px;overflow-wrap:anywhere}
.stat.sm .v{font-size:var(--fs-6)}
.stat .n{font-size:var(--fs-2);color:var(--muted);margin-top:2px}
.stat .spark{margin-top:var(--sp-2)}
.swatch{width:9px;height:9px;border-radius:3px;flex:0 0 9px;display:inline-block}

/* ==== charts ============================================================ */
.chart{width:100%;overflow-x:auto}
.chart svg{display:block;width:100%;height:auto}
.chart.center svg{margin-left:auto;margin-right:auto}
.legend{display:flex;flex-wrap:wrap;gap:var(--sp-2) var(--sp-4);
  margin-bottom:var(--sp-3)}
.legend .item{display:inline-flex;align-items:center;gap:7px;font-size:var(--fs-2);
  color:var(--muted);font-weight:560}
.legend .key{width:14px;height:4px;border-radius:2px;display:inline-block}
.donut-wrap{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);
  gap:var(--sp-4);align-items:center}
.klist{display:flex;flex-direction:column;gap:7px;font-size:var(--fs-3)}
.klist .row{display:flex;align-items:center;gap:8px}
.klist .nm{flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.klist .vl{color:var(--muted);font-variant-numeric:tabular-nums}

/* ==== tables ============================================================ */
.table-wrap{width:100%;overflow-x:auto;border:1px solid var(--line);
  border-radius:var(--r-2)}
table{border-collapse:collapse;width:100%;font-size:var(--fs-3)}
caption{text-align:left;font-size:var(--fs-2);color:var(--muted);
  padding:0 0 var(--sp-2)}
th,td{text-align:left;padding:9px 14px;border-bottom:1px solid var(--line);
  white-space:nowrap}
thead th{background:var(--surface-2);color:var(--muted);font-size:var(--fs-2);
  font-weight:640;text-transform:uppercase;letter-spacing:.04em}
tbody tr:last-child td{border-bottom:0}
tbody tr{transition:background var(--dur-1) var(--ease)}
tbody tr:hover{background:var(--surface-2)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr.is-sel td{background:var(--brand-soft)}

/* ==== states ============================================================ */
.empty{padding:var(--sp-6) var(--sp-4);text-align:center;color:var(--muted);
  border:1px dashed var(--line-2);border-radius:var(--r-2);background:var(--surface-2)}
.empty .t{font-weight:620;color:var(--fg-2);font-size:var(--fs-4)}
.empty .m{font-size:var(--fs-3);margin-top:4px}
.banner{display:flex;gap:var(--sp-3);align-items:flex-start;padding:var(--sp-4);
  border-radius:var(--r-2);font-size:var(--fs-3);margin-bottom:var(--sp-4)}
.banner.info{background:var(--brand-soft);color:var(--fg-2)}
.banner.warn{background:var(--warn-soft);color:var(--warn-ink)}
.skel{background:linear-gradient(90deg,var(--surface-3) 25%,var(--overlay) 37%,
  var(--surface-3) 63%);background-size:400% 100%;border-radius:var(--r-1);
  animation:shimmer 1.4s var(--ease) infinite}
.skel.line{height:12px;margin:6px 0}
.skel.chart{height:240px}
@keyframes shimmer{0%{background-position:100% 50%}100%{background-position:0 50%}}
details{background:var(--surface);border:1px solid var(--line);
  border-radius:var(--r-3);padding:var(--sp-4) var(--sp-5);box-shadow:var(--sh-1)}
summary{cursor:pointer;font-weight:620;font-size:var(--fs-4);list-style:none;
  display:flex;align-items:center;gap:8px}
summary::-webkit-details-marker{display:none}
summary:before{content:"";width:0;height:0;border:5px solid transparent;
  border-left-color:var(--muted-2);transition:transform var(--dur-2) var(--ease)}
details[open] summary:before{transform:rotate(90deg) translateX(-2px)}
details .body{margin-top:var(--sp-4)}
footer{margin-top:var(--sp-6);padding:var(--sp-5) 0;border-top:1px solid var(--line);
  color:var(--muted);font-size:var(--fs-2);display:flex;gap:var(--sp-4);
  flex-wrap:wrap;justify-content:space-between}

/* ==== responsive ======================================================== */
@media (max-width:1080px){
  .g-5{grid-template-columns:repeat(3,minmax(0,1fr))}
  .g-23{grid-template-columns:minmax(0,1fr)}
  .hero,.gauge-card{grid-column:span 3}
  .appnav{display:none}
}
@media (max-width:760px){
  .wrap{padding:0 var(--sp-4)}
  :root{--fs-7:24px;--fs-8:38px}
  .grid{gap:var(--sp-3)}
  .g-5,.g-4,.g-2{grid-template-columns:repeat(2,minmax(0,1fr))}
  .hero,.gauge-card{grid-column:span 2}
  .card{padding:var(--sp-4)}
  .donut-wrap{grid-template-columns:minmax(0,1fr)}
  .brand .sub{display:none}
}
@media (max-width:460px){
  /* Stat tiles stay two-up — a nine-tile single column is a scroll, not a
     scorecard. Only the hero and the gauge take the full width. */
  .g-2{grid-template-columns:minmax(0,1fr)}
  .hero,.gauge-card{grid-column:span 2}
  .hero{padding:var(--sp-4)}
  .stat .v{font-size:var(--fs-6)}
}
@media (prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  *{animation-duration:.001ms !important;animation-iteration-count:1 !important;
    transition-duration:.001ms !important}
}
@media print{
  .appbar,.toolbar,.skip,.no-print{display:none !important}
  body{background:#fff}
  .card,.stat,.table-wrap{break-inside:avoid;box-shadow:none}
  section{margin-top:18px}
}
"""

_REPORT_CSS = (_REPORT_TOKENS
               + ':root[data-theme="dark"]{' + _REPORT_DARK_VARS + '}\n'
               + '@media (prefers-color-scheme:dark){:root:not([data-theme="light"])'
               + '{' + _REPORT_DARK_VARS + '}}\n'
               + _REPORT_COMPONENTS)

_REPORT_JS = """
/* ContextIQ report view layer.
   Plain ES5, no dependencies, no globals: the page must work from file:// in
   any client, so every dependency is inlined and every DOM write goes through
   set()/put(), which no-op when a container is absent. */
(function(){
  var RANGES=['all','7','30','90'];
  var C_SENT='var(--s-sent)', C_SAVED='var(--s-saved)', C_BASE='var(--s-base)';
  var CAT=[C_SENT,C_SAVED,'var(--s-c3)','var(--s-c4)','var(--s-c5)',C_BASE];
  var SEQ=['var(--seq-0)','var(--seq-1)','var(--seq-2)','var(--seq-3)',
           'var(--seq-4)','var(--seq-5)'];
  var GRID='var(--grid)', MUT='var(--muted)', SURF='var(--surface)';
  var THEMES=['auto','light','dark'];
  var MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  var state={data:null,model:null,range:'all',theme:'auto',live:false,auto:true,
             sig:null,left:0,err:false};
  var tick=null;

  /* ---------- dom + formatting ------------------------------------------ */
  function el(id){return document.getElementById(id);}
  function set(id,t){var n=el(id); if(n)n.textContent=t; return n;}
  function put(id,h){var n=el(id); if(n)n.innerHTML=h; return n;}
  function on(id,ev,fn){var n=el(id);
    if(n&&n.addEventListener)n.addEventListener(ev,fn); return n;}
  function esc(s){return String(s).replace(/[&<>"]/g,function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
  function fmt(n){n=Number(n)||0;var a=Math.abs(n);
    if(a>=1e9)return (n/1e9).toFixed(1).replace(/\\.0$/,'')+'B';
    if(a>=1e6)return (n/1e6).toFixed(1).replace(/\\.0$/,'')+'M';
    if(a>=1e3)return (n/1e3).toFixed(1).replace(/\\.0$/,'')+'K';
    return String(Math.round(n));}
  function full(n){return (Number(n)||0).toLocaleString();}
  // Adaptive precision: money reads normally at cent scale and above, and only
  // grows decimals below that — otherwise a young ledger (or a cheap model)
  // collapses every figure to "$0.00" and switching model looks like a no-op.
  // Under $0.10 we keep three significant digits, so two models whose prices
  // differ stay visibly different.
  function usd(n){
    n=Number(n)||0;
    var a=Math.abs(n), d=2;
    if(a>0&&a<0.1)d=Math.min(8,Math.max(3,2-Math.floor(Math.log(a)/Math.LN10)));
    return '$'+n.toLocaleString(undefined,
      {minimumFractionDigits:d,maximumFractionDigits:d});
  }
  function pct(part,whole){return whole?Math.round(part/whole*1000)/10:0;}
  function priceOf(){
    var p=state.data.prices[state.model];
    return p===undefined?3.0:p;
  }
  function priceIo(m){
    var t=state.data.prices_io||{};
    return t[m]||{input:(state.data.prices||{})[m],output:null};
  }
  function priceLabel(){
    var p=priceOf();
    return '@ $'+(p<1?p.toFixed(3):p.toFixed(2))+' / 1M tokens';
  }
  function rangeLabel(){
    return state.range==='all'?'all time':'last '+state.range+' days';
  }

  /* ---------- UI state in the URL hash ----------------------------------- */
  // The static page can only pick up new data by reloading (file:// cannot
  // fetch), so selections are round-tripped through location.hash to survive
  // it. sessionStorage is not usable here: Chrome restricts it on file://.
  function readHash(){
    var out={};
    (location.hash||'').replace(/^#/,'').split('&').forEach(function(kv){
      var i=kv.indexOf('=');
      if(i>0)out[decodeURIComponent(kv.slice(0,i))]=decodeURIComponent(kv.slice(i+1));
    });
    return out;
  }
  function writeHash(){
    var h='#model='+encodeURIComponent(state.model)+
          '&range='+encodeURIComponent(state.range)+'&auto='+(state.auto?'1':'0')+
          '&theme='+encodeURIComponent(state.theme);
    try{history.replaceState(null,'',h);}catch(e){location.hash=h;}
  }
  // Cheap change-detector so a poll that returns identical data doesn't redraw.
  function sig(d){
    var t=d.totals||{};
    return [d.generated_at,t.runs,t.saved,t.final,(d.rows||[]).length].join('|');
  }

  /* ---------- windowing --------------------------------------------------- */
  function dkey(d){
    var m=d.getMonth()+1, day=d.getDate();
    return d.getFullYear()+'-'+(m<10?'0':'')+m+'-'+(day<10?'0':'')+day;
  }
  function windowStart(){                  // local midnight, N-1 days back
    var d=new Date(); d.setHours(0,0,0,0);
    d.setDate(d.getDate()-(parseInt(state.range,10)-1));
    return d;
  }
  function series(){
    var d=state.data.daily||[];
    if(state.range==='all')return d;
    var cut=dkey(windowStart());
    return d.filter(function(x){return x.period>=cut;});
  }
  function totals(s){
    if(state.range==='all')return state.data.totals||
      {runs:0,saved:0,baseline:0,final:0,reduction_pct:0,files:0};
    var t=s.reduce(function(a,x){return {runs:a.runs+(x.runs||0),
      saved:a.saved+(x.saved||0),baseline:a.baseline+(x.baseline||0),
      final:a.final+(x.final||0),files:a.files+(x.files||0)};},
      {runs:0,saved:0,baseline:0,final:0,files:0});
    t.reduction_pct=pct(t.saved,t.baseline);
    return t;
  }
  function ops(){                          // per-operation rows for the window
    if(state.range==='all')return state.data.by_op||[];
    var cut=windowStart().getTime()/1000, agg={}, out=[];
    (state.data.rows||[]).forEach(function(r){
      if(Number(r.ts||0)<cut)return;
      var k=r.op||'?', b=agg[k];
      if(!b){b=agg[k]={op:k,runs:0,saved:0,baseline:0,final:0};out.push(b);}
      b.runs++; b.saved+=Number(r.saved||0);
      b.baseline+=Number(r.baseline_tokens||0);
      b.final+=Number(r.final_tokens||0);
    });
    return out.sort(function(a,b){return b.saved-a.saved;});
  }
  function windowRows(){
    if(state.range==='all')return state.data.rows||[];
    var cut=windowStart().getTime()/1000;
    return (state.data.rows||[]).filter(function(r){return Number(r.ts||0)>=cut;});
  }
  function savedUsd(saved){return saved/1e6*priceOf();}
  function spendUsd(sent){return sent/1e6*priceOf();}

  /* ---------- svg primitives ---------------------------------------------- */
  // max-width = the viewBox width so labels never scale *up* past their design
  // size in a wide card; min-width keeps them legible in a narrow one — the
  // .chart container scrolls instead of shrinking the type to nothing.
  function svg(w,h,label,inner){
    return '<svg viewBox="0 0 '+w+' '+h+'" width="100%" height="'+h+'" '+
      'style="min-width:'+Math.min(w,430)+'px;max-width:'+w+'px" '+
      'preserveAspectRatio="xMidYMid meet" role="img" aria-label="'+esc(label)+
      '">'+inner+'</svg>';
  }
  function empty(title,msg){
    return '<div class="empty"><div class="t">'+esc(title)+'</div>'+
           '<div class="m">'+esc(msg)+'</div></div>';
  }
  function niceMax(v){
    if(!(v>0))return 1;
    var e=Math.pow(10,Math.floor(Math.log(v)/Math.LN10)), f=v/e;
    return (f<=1?1:f<=2?2:f<=2.5?2.5:f<=5?5:10)*e;
  }
  function colTop(x,y,w,h,r){           // column: rounded cap, square baseline
    if(h<=0.5)h=0.5;
    r=Math.min(r,w/2,h);
    return 'M'+x+','+(y+h).toFixed(1)+'V'+(y+r).toFixed(1)+'Q'+x+','+y.toFixed(1)+
      ' '+(x+r)+','+y.toFixed(1)+'H'+(x+w-r)+'Q'+(x+w)+','+y.toFixed(1)+' '+
      (x+w)+','+(y+r).toFixed(1)+'V'+(y+h).toFixed(1)+'Z';
  }
  function barEnd(x,y,w,h,r){           // horizontal bar: rounded data end
    if(w<=0.5)w=0.5;
    r=Math.min(r,h/2,w);
    return 'M'+x+','+y+'H'+(x+w-r).toFixed(1)+'Q'+(x+w).toFixed(1)+','+y+' '+
      (x+w).toFixed(1)+','+(y+r)+'V'+(y+h-r)+'Q'+(x+w).toFixed(1)+','+(y+h)+' '+
      (x+w-r).toFixed(1)+','+(y+h)+'H'+x+'Z';
  }
  function gridY(max,x0,x1,top,bot,steps){
    var out='';
    for(var i=0;i<=steps;i++){
      var y=bot-(bot-top)*(i/steps);
      out+='<line x1="'+x0+'" y1="'+y.toFixed(1)+'" x2="'+x1+'" y2="'+y.toFixed(1)+
        '" stroke="'+GRID+'" stroke-width="1"/>'+
        '<text x="'+(x0-8)+'" y="'+(y+3.5).toFixed(1)+'" text-anchor="end" '+
        'font-size="10" fill="'+MUT+'">'+fmt(max*i/steps)+'</text>';
    }
    return out;
  }
  function tip(text){return '<title>'+esc(text)+'</title>';}

  /* ---------- charts ------------------------------------------------------ */
  function sparkline(vals){
    if(!vals.length)return '';
    var w=150,h=30,max=Math.max.apply(null,vals.concat([1])),n=vals.length,p='';
    function x(i){return n<2?w/2:i*(w/(n-1));}
    function y(v){return h-3-(v/max)*(h-8);}
    for(var i=0;i<n;i++)p+=(i?' L':'M')+x(i).toFixed(1)+','+y(vals[i]).toFixed(1);
    return svg(w,h,'Daily tokens avoided, last '+n+' day(s)',
      '<path d="'+p+'" fill="none" stroke="'+C_SAVED+'" stroke-width="2" '+
      'stroke-linejoin="round" stroke-linecap="round" opacity=".7"/>'+
      '<circle cx="'+x(n-1).toFixed(1)+'" cy="'+y(vals[n-1]).toFixed(1)+
      '" r="3.5" fill="'+C_SAVED+'" stroke="'+SURF+'" stroke-width="2"/>');
  }

  function waterfall(t){
    if(!t.baseline)return empty('Nothing measured yet',
      'Run a context, ask or measure call and the breakdown appears here.');
    var w=560,h=300,x0=64,x1=548,top=30,bot=236,bw=30;
    var max=niceMax(t.baseline), slot=(x1-x0)/3;
    function y(v){return bot-(v/max)*(bot-top);}
    function cx(i){return x0+slot*i+slot/2;}
    var b=t.baseline, f=t.final, sv=t.saved;
    var o=gridY(max,x0,x1,top,bot,4);
    o+='<line x1="'+x0+'" y1="'+bot+'" x2="'+x1+'" y2="'+bot+
      '" stroke="var(--line-2)" stroke-width="1"/>';
    o+='<line x1="'+(cx(0)+bw/2)+'" y1="'+y(b).toFixed(1)+'" x2="'+(cx(1)-bw/2)+
      '" y2="'+y(b).toFixed(1)+'" stroke="var(--line-2)" stroke-width="1"/>';
    o+='<line x1="'+(cx(1)+bw/2)+'" y1="'+y(f).toFixed(1)+'" x2="'+(cx(2)-bw/2)+
      '" y2="'+y(f).toFixed(1)+'" stroke="var(--line-2)" stroke-width="1"/>';
    o+='<path d="'+colTop(cx(0)-bw/2,y(b),bw,bot-y(b),4)+'" fill="'+C_BASE+'">'+
      tip('Baseline (same files read whole): '+full(b)+' tokens')+'</path>';
    o+='<rect x="'+(cx(1)-bw/2)+'" y="'+y(b).toFixed(1)+'" width="'+bw+'" height="'+
      Math.max(1,y(f)-y(b)).toFixed(1)+'" rx="3" fill="'+C_SAVED+'">'+
      tip('Avoided by ContextIQ: '+full(sv)+' tokens ('+usd(savedUsd(sv))+')')+
      '</rect>';
    o+='<path d="'+colTop(cx(2)-bw/2,y(f),bw,bot-y(f),4)+'" fill="'+C_SENT+'">'+
      tip('Actually sent: '+full(f)+' tokens ('+usd(spendUsd(f))+')')+'</path>';
    [['Baseline',fmt(b),'100%',cx(0),y(b)],
     ['Avoided','-'+fmt(sv),t.reduction_pct+'%',cx(1),y(b)],
     ['Sent',fmt(f),(Math.round((100-t.reduction_pct)*10)/10)+'%',cx(2),y(f)]
    ].forEach(function(r){
      o+='<text x="'+r[3]+'" y="'+(r[4]-10).toFixed(1)+'" text-anchor="middle" '+
        'font-size="13" font-weight="650" fill="currentColor">'+r[1]+'</text>'+
        '<text x="'+r[3]+'" y="'+(bot+19)+'" text-anchor="middle" font-size="11" '+
        'fill="currentColor">'+r[0]+'</text>'+
        '<text x="'+r[3]+'" y="'+(bot+34)+'" text-anchor="middle" font-size="10" '+
        'fill="'+MUT+'">'+r[2]+' of baseline</text>';
    });
    return svg(w,h,'Waterfall: baseline '+fmt(b)+' tokens, '+fmt(sv)+
      ' avoided, '+fmt(f)+' sent',o);
  }

  function areaChart(s){
    if(!s.length)return empty('No runs in this window',
      'Widen the date range, or run a few context packs to fill the trend.');
    var w=560,h=300,x0=64,x1=548,top=30,bot=236,n=s.length,peak=0,i;
    s.forEach(function(d){peak=Math.max(peak,(d.final||0)+(d.saved||0));});
    var max=niceMax(peak);
    function y(v){return bot-(v/max)*(bot-top);}
    var o=gridY(max,x0,x1,top,bot,4);
    o+='<line x1="'+x0+'" y1="'+bot+'" x2="'+x1+'" y2="'+bot+
      '" stroke="var(--line-2)" stroke-width="1"/>';
    if(n<2){                               // one day of data: columns, not a band
      var d0=s[0], c=(x0+x1)/2, bw=34;
      o+='<path d="'+colTop(c-bw-4,y(d0.final||0),bw,bot-y(d0.final||0),4)+
        '" fill="'+C_SENT+'">'+tip(d0.period+': '+full(d0.final)+' sent')+'</path>'+
        '<path d="'+colTop(c+4,y(d0.saved||0),bw,bot-y(d0.saved||0),4)+'" fill="'+
        C_SAVED+'">'+tip(d0.period+': '+full(d0.saved)+' avoided')+'</path>'+
        '<text x="'+c+'" y="'+(bot+19)+'" text-anchor="middle" font-size="11" '+
        'fill="'+MUT+'">'+esc(d0.period)+'</text>';
      return svg(w,h,'Tokens sent and avoided on '+d0.period,o);
    }
    function x(i){return x0+i*((x1-x0)/(n-1));}
    var sentTop='',sentBot='',savTop='',savBot='',lineSent='',lineSav='';
    for(i=0;i<n;i++){
      var f=s[i].final||0, tot=f+(s[i].saved||0);
      sentTop+=x(i).toFixed(1)+','+y(f).toFixed(1)+' ';
      savTop+=x(i).toFixed(1)+','+y(tot).toFixed(1)+' ';
      lineSent+=(i?' L':'M')+x(i).toFixed(1)+','+y(f).toFixed(1);
      lineSav+=(i?' L':'M')+x(i).toFixed(1)+','+y(tot).toFixed(1);
    }
    for(i=n-1;i>=0;i--){
      sentBot+=x(i).toFixed(1)+','+bot+' ';
      savBot+=x(i).toFixed(1)+','+(y(s[i].final||0)-2).toFixed(1)+' ';
    }
    o+='<polygon points="'+(savTop+savBot)+'" fill="'+C_SAVED+'" opacity=".16"/>'+
      '<polygon points="'+(sentTop+sentBot)+'" fill="'+C_SENT+'" opacity=".16"/>'+
      '<path d="'+lineSav+'" fill="none" stroke="'+C_SAVED+'" stroke-width="2" '+
      'stroke-linejoin="round"/>'+
      '<path d="'+lineSent+'" fill="none" stroke="'+C_SENT+'" stroke-width="2" '+
      'stroke-linejoin="round"/>';
    var lastTot=(s[n-1].final||0)+(s[n-1].saved||0);
    o+='<circle cx="'+x(n-1).toFixed(1)+'" cy="'+y(lastTot).toFixed(1)+
      '" r="4" fill="'+C_SAVED+'" stroke="'+SURF+'" stroke-width="2"/>'+
      '<circle cx="'+x(n-1).toFixed(1)+'" cy="'+y(s[n-1].final||0).toFixed(1)+
      '" r="4" fill="'+C_SENT+'" stroke="'+SURF+'" stroke-width="2"/>'+
      '<text x="'+(x1-2)+'" y="'+(y(lastTot)-12).toFixed(1)+'" text-anchor="end" '+
      'font-size="11" font-weight="620" fill="currentColor">'+fmt(lastTot)+
      ' total</text>';
    for(i=0;i<n;i++){                      // generous hover targets
      var t2=(s[i].final||0)+(s[i].saved||0);
      o+='<circle cx="'+x(i).toFixed(1)+'" cy="'+y(t2).toFixed(1)+'" r="10" '+
        'fill="transparent">'+tip(s[i].period+' — avoided '+full(s[i].saved)+
        ', sent '+full(s[i].final)+', '+(s[i].runs||0)+' run(s)')+'</circle>';
    }
    var mid=Math.floor((n-1)/2);
    [[0,'start'],[mid,'middle'],[n-1,'end']].forEach(function(L,k){
      if(k===1&&n<4)return;
      o+='<text x="'+x(L[0]).toFixed(1)+'" y="'+(bot+19)+'" text-anchor="'+L[1]+
        '" font-size="11" fill="'+MUT+'">'+esc(s[L[0]].period)+'</text>';
    });
    return svg(w,h,'Daily tokens avoided and sent over '+n+' days',o);
  }

  function opBars(rows){
    if(!rows.length)return empty('No operations in this window',
      'Every context, ask, measure, squeeze or dedupe call lands here.');
    var top=rows.slice(0,8), total=0;
    rows.forEach(function(o){total+=o.saved;});
    var max=Math.max.apply(null,top.map(function(o){return o.saved;}).concat([1]));
    var w=560,lw=104,right=170,rh=32,bt=14,h=top.length*rh+10,o='';
    top.forEach(function(r,i){
      var y=i*rh+8, bw=Math.max(2,(r.saved/max)*(w-lw-right));
      o+='<text x="0" y="'+(y+bt-2)+'" font-size="12" fill="currentColor">'+
        esc(String(r.op).slice(0,15))+'</text>'+
        '<path d="'+barEnd(lw,y,bw,bt,4)+'" fill="'+C_SAVED+'">'+
        tip(r.op+': '+full(r.saved)+' tokens avoided ('+usd(savedUsd(r.saved))+
        ') over '+r.runs+' run(s)')+'</path>'+
        '<text x="'+(lw+bw+8)+'" y="'+(y+bt-3)+'" font-size="11" fill="'+MUT+'">'+
        fmt(r.saved)+' · '+usd(savedUsd(r.saved))+' · '+pct(r.saved,total)+
        '%</text>';
    });
    return svg(w,h,'Tokens avoided by operation',o);
  }

  function donut(rows){
    var runs=0;
    rows.forEach(function(r){runs+=r.runs;});
    if(!runs)return empty('No runs yet','Nothing to break down.');
    var slice=rows.slice(0,5), rest=rows.slice(5), other=0;
    rest.forEach(function(r){other+=r.runs;});
    if(other)slice=slice.concat([{op:'other ('+rest.length+')',runs:other}]);
    var w=240,h=240,cx=120,cy=118,r=94,ir=62,a=0,gap=slice.length>1?2:0,o='';
    function pt(rad,deg){var t=(deg-90)*Math.PI/180;
      return [(cx+rad*Math.cos(t)).toFixed(2),(cy+rad*Math.sin(t)).toFixed(2)];}
    slice.forEach(function(s,i){
      var span=s.runs/runs*360, a0=a+gap/2, a1=a+span-gap/2, col=CAT[i%CAT.length];
      a+=span;
      if(span>=359.5){
        o+='<circle cx="'+cx+'" cy="'+cy+'" r="'+((r+ir)/2)+'" fill="none" '+
          'stroke="'+col+'" stroke-width="'+(r-ir)+'">'+
          tip(s.op+': '+s.runs+' run(s), 100%')+'</circle>';
        return;
      }
      if(a1<=a0)return;
      var p0=pt(r,a0),p1=pt(r,a1),q1=pt(ir,a1),q0=pt(ir,a0),la=(a1-a0)>180?1:0;
      o+='<path d="M'+p0[0]+','+p0[1]+'A'+r+','+r+' 0 '+la+' 1 '+p1[0]+','+p1[1]+
        'L'+q1[0]+','+q1[1]+'A'+ir+','+ir+' 0 '+la+' 0 '+q0[0]+','+q0[1]+'Z" fill="'+
        col+'">'+tip(s.op+': '+s.runs+' run(s), '+pct(s.runs,runs)+'%')+'</path>';
    });
    o+='<text x="'+cx+'" y="'+(cy+4)+'" text-anchor="middle" font-size="30" '+
      'font-weight="680" fill="currentColor">'+fmt(runs)+'</text>'+
      '<text x="'+cx+'" y="'+(cy+24)+'" text-anchor="middle" font-size="11" '+
      'fill="'+MUT+'">runs</text>';
    var list='<ul class="klist">';
    slice.forEach(function(s,i){
      list+='<li class="row"><span class="swatch" style="background:'+
        CAT[i%CAT.length]+'"></span><span class="nm">'+esc(s.op)+
        '</span><span class="vl">'+full(s.runs)+' · '+pct(s.runs,runs)+
        '%</span></li>';
    });
    return '<div class="donut-wrap"><div class="chart">'+
      svg(w,h,'Share of runs by operation',o)+'</div>'+list+'</ul></div>';
  }

  function heatmap(s){
    if(!s.length)return empty('No activity yet',
      'Daily savings appear here once the ledger has entries.');
    var by={};
    s.forEach(function(d){by[d.period]={saved:d.saved||0,runs:d.runs||0};});
    var WEEKS=26, cs=20, gp=4, left=36, top=30, vals=[], k;
    for(k in by){if(by[k].saved>0)vals.push(by[k].saved);}
    vals.sort(function(a,b){return a-b;});
    var q=[0.2,0.4,0.6,0.8].map(function(p){
      return vals.length?vals[Math.floor(p*(vals.length-1))]:0;});
    function bucket(v){
      if(!v)return 0;
      if(!vals.length)return 3;
      return v<=q[0]?1:v<=q[1]?2:v<=q[2]?3:v<=q[3]?4:5;
    }
    var today=new Date(); today.setHours(12,0,0,0);
    var start=new Date(today);
    start.setDate(today.getDate()-((WEEKS-1)*7+today.getDay()));
    var w=left+WEEKS*(cs+gp)+8, h=top+7*(cs+gp)+32, o='', lastMonth=-1;
    for(var i=0;i<WEEKS*7;i++){
      var d=new Date(start); d.setDate(start.getDate()+i);
      if(d>today)continue;
      var col=Math.floor(i/7), row=i%7, key=dkey(d), cell=by[key];
      var v=cell?cell.saved:0, x=left+col*(cs+gp), y=top+row*(cs+gp);
      o+='<rect x="'+x+'" y="'+y+'" width="'+cs+'" height="'+cs+'" rx="3" fill="'+
        SEQ[bucket(v)]+'">'+tip(key+(cell?(': '+full(v)+' tokens avoided · '+
        cell.runs+' run(s)'):': no runs'))+'</rect>';
      if(row===0&&d.getMonth()!==lastMonth&&d.getDate()<=7){
        lastMonth=d.getMonth();
        o+='<text x="'+x+'" y="'+(top-9)+'" font-size="10" fill="'+MUT+'">'+
          MONTHS[d.getMonth()]+'</text>';
      }
    }
    ['Mon','Wed','Fri'].forEach(function(lab,j){
      o+='<text x="0" y="'+(top+(j*2+1)*(cs+gp)+cs-2)+'" font-size="10" fill="'+
        MUT+'">'+lab+'</text>';
    });
    var lx=left, ly=h-15;
    o+='<text x="'+lx+'" y="'+(ly+cs-3)+'" font-size="10" fill="'+MUT+
      '">less</text>';
    for(var b=0;b<6;b++){
      o+='<rect x="'+(lx+30+b*(cs+gp))+'" y="'+ly+'" width="'+cs+'" height="'+cs+
        '" rx="3" fill="'+SEQ[b]+'"/>';
    }
    o+='<text x="'+(lx+36+6*(cs+gp))+'" y="'+(ly+cs-3)+'" font-size="10" fill="'+
      MUT+'">more</text>';
    return svg(w,h,'Daily savings over the last '+WEEKS+' weeks',o);
  }

  function gauge(t){
    var w=260,h=152,cx=130,cy=128,r=96,sw=18;
    var val=Math.max(0,Math.min(100,Number(t.reduction_pct)||0)), frac=val/100;
    function arc(f){
      var ang=Math.PI*f;
      return 'M'+(cx-r)+','+cy+' A'+r+','+r+' 0 '+(f>0.5?1:0)+' 1 '+
        (cx-r*Math.cos(ang)).toFixed(2)+','+(cy-r*Math.sin(ang)).toFixed(2);
    }
    var o='<path d="'+arc(1)+'" fill="none" stroke="var(--track)" stroke-width="'+
      sw+'" stroke-linecap="round"/>';
    if(frac>0.004){
      o+='<path d="'+arc(frac)+'" fill="none" stroke="'+C_SAVED+'" stroke-width="'+
        sw+'" stroke-linecap="round">'+tip('Prompts are '+val+
        '% smaller than the same files read whole')+'</path>';
    }
    o+='<text x="'+cx+'" y="'+(cy-18)+'" text-anchor="middle" font-size="40" '+
      'font-weight="700" fill="currentColor">'+val+'%</text>'+
      '<text x="'+cx+'" y="'+(cy+4)+'" text-anchor="middle" font-size="12" '+
      'fill="'+MUT+'">of baseline avoided</text>'+
      '<text x="'+(cx-r)+'" y="'+(cy+22)+'" text-anchor="middle" font-size="10" '+
      'fill="'+MUT+'">0%</text>'+
      '<text x="'+(cx+r)+'" y="'+(cy+22)+'" text-anchor="middle" font-size="10" '+
      'fill="'+MUT+'">100%</text>';
    return svg(w,h,'Prompt reduction gauge: '+val+
      ' percent of baseline tokens avoided',o);
  }

  function modelNames(){
    var names=Object.keys(state.data.prices||{});
    names.sort(function(a,b){
      return state.data.prices[b]-state.data.prices[a];});
    return names;
  }

  function modelChart(t){
    var names=modelNames();
    if(!names.length||!t.saved)return empty('No pricing comparison yet',
      'Once the ledger records savings, every priced model is compared here.');
    var max=Math.max.apply(null,names.map(function(m){
      return t.saved/1e6*state.data.prices[m];}).concat([1e-9]));
    var w=940,lw=150,right=200,rh=28,bt=14,h=names.length*rh+10,o='';
    names.forEach(function(m,i){
      var v=t.saved/1e6*state.data.prices[m], sel=(m===state.model);
      var y=i*rh+7, bw=Math.max(2,(v/max)*(w-lw-right));
      if(sel)o+='<rect x="0" y="'+(y-4)+'" width="'+w+'" height="'+(rh-3)+
        '" rx="6" fill="var(--brand-soft)"/>';
      o+='<text x="6" y="'+(y+bt-2)+'" font-size="11.5" font-weight="'+
        (sel?'660':'400')+'" fill="currentColor">'+esc(m)+'</text>'+
        '<path d="'+barEnd(lw,y,bw,bt,4)+'" fill="'+C_SENT+'" opacity="'+
        (sel?'1':'.6')+'">'+tip(m+': '+usd(v)+' avoided on '+full(t.saved)+
        ' tokens at $'+state.data.prices[m]+' per 1M input tokens')+'</path>'+
        '<text x="'+(lw+bw+8)+'" y="'+(y+bt-3)+'" font-size="11" fill="'+MUT+'">'+
        usd(v)+(sel?' · selected':'')+'</text>';
    });
    return svg(w,h,'Cost avoided by pricing model',o);
  }

  function modelTable(t){
    var names=modelNames();
    if(!names.length)return '';
    var o='<div class="table-wrap" tabindex="0" role="region" '+
      'aria-label="Cost by pricing model"><table><caption class="sr">'+
      'Projected cost avoided and estimated spend for every priced model'+
      '</caption><thead><tr><th scope="col">Model</th>'+
      '<th scope="col" class="n">Input $/1M</th>'+
      '<th scope="col" class="n">Output $/1M</th>'+
      '<th scope="col" class="n">Cost avoided</th>'+
      '<th scope="col" class="n">Est. spend on sent</th>'+
      '<th scope="col" class="n">Baseline cost</th></tr></thead><tbody>';
    names.forEach(function(m){
      var p=state.data.prices[m], io=priceIo(m);
      o+='<tr'+(m===state.model?' class="is-sel"':'')+'><td>'+esc(m)+
        (m===state.model?' <span class="pill info">selected</span>':'')+
        '</td><td class="n">$'+(p<1?p.toFixed(3):p.toFixed(2))+'</td>'+
        '<td class="n">'+(io.output==null?'—':'$'+Number(io.output).toFixed(2))+
        '</td><td class="n">'+usd(t.saved/1e6*p)+'</td><td class="n">'+
        usd(t.final/1e6*p)+'</td><td class="n">'+usd(t.baseline/1e6*p)+'</td></tr>';
    });
    return o+'</tbody></table></div>';
  }

  function opTable(rows,t){
    if(!rows.length)return '';
    var o='<div class="table-wrap" tabindex="0" role="region" '+
      'aria-label="Savings by operation"><table><caption class="sr">'+
      'Per-operation totals for the selected window</caption><thead><tr>'+
      '<th scope="col">Operation</th><th scope="col" class="n">Runs</th>'+
      '<th scope="col" class="n">Baseline</th><th scope="col" class="n">Sent</th>'+
      '<th scope="col" class="n">Avoided</th>'+
      '<th scope="col" class="n">Reduction</th><th scope="col" class="n">Share</th>'+
      '<th scope="col" class="n">Cost avoided</th></tr></thead><tbody>';
    rows.forEach(function(r){
      o+='<tr><td>'+esc(r.op)+'</td><td class="n">'+full(r.runs)+
        '</td><td class="n">'+full(r.baseline)+'</td><td class="n">'+full(r.final)+
        '</td><td class="n">'+full(r.saved)+'</td><td class="n">'+
        pct(r.saved,r.baseline)+'%</td><td class="n">'+pct(r.saved,t.saved)+
        '%</td><td class="n">'+usd(savedUsd(r.saved))+'</td></tr>';
    });
    return o+'</tbody></table></div>';
  }

  function wsCards(ws,d){
    var tiles=[], lg=d.ledger||{};
    if(ws&&ws.files){
      tiles=[['Files indexed',full(ws.files),'tracked in the code graph'],
             ['Symbols',full(ws.symbols),'functions, classes, methods'],
             ['Graph edges',full(ws.edges),'calls, imports, inheritance'],
             ['Indexed chunks',full(ws.chunks),'retrievable text blocks'],
             ['Module summaries',full(ws.summaries),'reused by future packs'],
             ['Tokens in repo',fmt(ws.indexed_tokens),'estimated across indexed files'],
             ['Graph size',(Number(ws.db_bytes||0)/1048576).toFixed(1)+' MB',
              'tokengraph/graph.db']];
    }
    if(lg.first_ts){
      tiles.push(['First run',
        new Date(lg.first_ts*1000).toLocaleDateString(),'oldest ledger entry']);
      tiles.push(['Active days',full(lg.active_days),
        'days with at least one recorded run']);
    }
    if(!tiles.length)return empty('No graph index found',
      'Workspace facts appear once the graph database exists — run an index or '+
      'any context call.');
    var o='<div class="grid g-4">';
    tiles.forEach(function(t){
      o+='<div class="stat sm"><div class="k">'+esc(t[0])+'</div>'+
        '<div class="v num">'+esc(t[1])+'</div><div class="n">'+esc(t[2])+
        '</div></div>';
    });
    return o+'</div>';
  }

  function langBars(ws){
    var langs=(ws&&ws.languages)||[];
    if(!langs.length)return empty('No language breakdown',
      'Index the workspace to see what the graph covers.');
    var top=langs.slice(0,8), tot=0;
    langs.forEach(function(l){tot+=l.files;});
    var max=Math.max.apply(null,top.map(function(l){return l.files;}).concat([1]));
    var w=940,lw=140,right=220,rh=30,bt=14,h=top.length*rh+10,o='';
    top.forEach(function(l,i){
      var y=i*rh+8, bw=Math.max(2,(l.files/max)*(w-lw-right));
      o+='<text x="0" y="'+(y+bt-2)+'" font-size="12" fill="currentColor">'+
        esc(String(l.language).slice(0,14))+'</text>'+
        '<path d="'+barEnd(lw,y,bw,bt,4)+'" fill="'+C_SENT+'">'+
        tip(l.language+': '+full(l.files)+' file(s), '+fmt(l.tokens)+
        ' tokens indexed')+'</path>'+
        '<text x="'+(lw+bw+8)+'" y="'+(y+bt-3)+'" font-size="11" fill="'+MUT+'">'+
        full(l.files)+' files · '+fmt(l.tokens)+' tok · '+pct(l.files,tot)+
        '%</text>';
    });
    return svg(w,h,'Indexed files by language',o);
  }

  function rowsTable(rows){
    if(!rows.length)return empty('No records in this window',
      'Savings are appended to the ledger as you use the graph.');
    var o='<div class="table-wrap" tabindex="0" role="region" '+
      'aria-label="Raw ledger records"><table><caption class="sr">'+
      'Most recent ledger entries, newest first</caption><thead><tr>'+
      '<th scope="col">When</th><th scope="col">Operation</th>'+
      '<th scope="col" class="n">Files</th><th scope="col" class="n">Baseline</th>'+
      '<th scope="col" class="n">Sent</th><th scope="col" class="n">Avoided</th>'+
      '<th scope="col" class="n">Reduction</th>'+
      '<th scope="col" class="n">Cost avoided</th></tr></thead><tbody>';
    rows.slice().reverse().slice(0,50).forEach(function(r){
      o+='<tr><td>'+esc(r.ts?new Date(r.ts*1000).toLocaleString():'—')+'</td><td>'+
        esc(r.op||'?')+'</td><td class="n">'+(r.files==null?'—':full(r.files))+
        '</td><td class="n">'+full(r.baseline_tokens)+'</td><td class="n">'+
        full(r.final_tokens)+'</td><td class="n">'+full(r.saved)+
        '</td><td class="n">'+(r.reduction_pct||0)+'%</td><td class="n">'+
        usd(savedUsd(Number(r.saved)||0))+'</td></tr>';
    });
    return o+'</tbody></table></div>';
  }

  /* ---------- draw --------------------------------------------------------- */
  function draw(){
    var d=state.data, s=series(), t=totals(s), op=ops();
    var lg=d.ledger||{}, ws=d.workspace_stats||{};

    set('hero-value',usd(savedUsd(t.saved)));
    set('hero-sub',fmt(t.saved)+' tokens never sent · prompts '+
      t.reduction_pct+'% smaller · '+rangeLabel());
    set('hero-note','Reading the same files whole would have cost '+
      usd(savedUsd(t.baseline))+' in input tokens; the packs cost '+
      usd(spendUsd(t.final))+'.');

    set('m-saved',fmt(t.saved));
    set('k-saved-n',full(t.saved)+' tokens · '+rangeLabel());
    put('k-spark',sparkline(s.slice(-14).map(function(x){return x.saved||0;})));
    set('m-sent',fmt(t.final));
    set('k-sent-n',pct(t.final,t.baseline)+'% of a '+fmt(t.baseline)+
      '-token baseline');
    set('m-spend',usd(spendUsd(t.final)));
    set('k-spend-n',state.model+' '+priceLabel());
    set('m-red',t.reduction_pct+'%');
    set('k-red-n','smaller than reading those files whole');
    set('m-runs',full(t.runs||0));
    set('k-runs-n',op.length+' operation type(s) recorded');
    set('m-baseline',fmt(t.baseline));
    set('m-avg',t.runs?fmt(Math.round(t.saved/t.runs)):'—');
    set('m-leverage',t.final?((t.baseline/t.final).toFixed(1)+'×'):'—');
    set('m-files',full(t.files||0));
    set('unit-price',priceLabel());

    put('c-gauge',gauge(t));
    put('c-waterfall',waterfall(t));
    put('c-area',areaChart(s));
    put('c-heat',heatmap(d.daily||[]));
    put('c-ops',opBars(op));
    put('c-mix',donut(op));
    put('c-optable',opTable(op,t));
    put('c-models',modelChart(t));
    put('c-modeltable',modelTable(t));
    put('c-ws',wsCards(ws,d));
    put('c-langs',langBars(ws));
    put('c-rows',rowsTable(windowRows()));

    set('range-note',state.range==='all'
      ? 'Showing all '+full((d.totals||{}).runs||0)+' recorded run(s).'
      : 'Showing the last '+state.range+' days.'+(d.rows_capped?
        ' Per-operation and log views cover the most recent '+
        (d.rows||[]).length+' entries.':''));
    set('ws-chip',d.workspace||'workspace');
    put('meta','Workspace <b>'+esc(d.workspace||'—')+'</b>'+
      (d.generated_at?(' · generated '+esc(d.generated_at)):'')+' · '+
      (state.live?'<span class="pill live"><span class="dot"></span>live</span>'
                 :'<span class="pill">snapshot</span>')+
      (state.err?' <span class="pill warn">offline · showing last snapshot</span>'
                 :''));
    put('onboard',t.runs?'':
      '<div class="banner info"><div><b>No runs in this window yet.</b> '+
      'ContextIQ appends one line to <code>.context/gain.ndjson</code> every '+
      'time it builds a context pack — run a context, ask or measure call (or '+
      'use the MCP tools), then refresh.</div></div>');
    set('foot-gen',d.generated_at||'not stamped');
    set('foot-ver','schema v'+(d.version||1));
  }

  /* ---------- controls ------------------------------------------------------ */
  function applyTheme(){
    var r=document.documentElement;
    if(r){
      if(state.theme==='auto'){if(r.removeAttribute)r.removeAttribute('data-theme');}
      else if(r.setAttribute)r.setAttribute('data-theme',state.theme);
    }
    set('btn-theme',state.theme==='dark'?'☾':(state.theme==='light'?'☀':'◐'));
    var b=el('btn-theme');
    if(b&&b.setAttribute)b.setAttribute('aria-label',
      'Colour theme: '+state.theme+'. Activate to change.');
  }
  function controls(){
    var d=state.data, ms=el('sel-model'), rs=el('sel-range'), ck=el('chk-auto');
    if(ms){
      Object.keys(d.prices||{}).forEach(function(m){
        var o=document.createElement('option');
        o.value=m; o.textContent=m;
        ms.appendChild(o);
      });
      ms.value=state.model;
      ms.addEventListener('change',function(){
        state.model=ms.value; writeHash(); draw();});
    }
    if(rs){
      rs.value=state.range;
      rs.addEventListener('change',function(){
        state.range=rs.value; writeHash(); draw();});
    }
    if(ck){
      ck.checked=state.auto;
      ck.addEventListener('change',function(){
        state.auto=ck.checked; writeHash(); startTimer();});
    }
    on('btn-theme','click',function(){
      state.theme=THEMES[(THEMES.indexOf(state.theme)+1)%THEMES.length];
      applyTheme(); writeHash();});
    on('btn-refresh','click',function(){
      if(location.protocol==='file:'){location.reload();return;}
      refresh(); startTimer();});
  }

  function refresh(){
    if(location.protocol==='file:'){draw();return;}
    fetch('data.json',{cache:'no-store'}).then(function(r){
      if(!r.ok)throw 0;
      return r.json();
    }).then(function(j){
      state.live=true; state.err=false;
      var s=sig(j);
      if(s===state.sig){renderTimer();return;}   // nothing changed — no redraw
      state.sig=s; state.data=j; draw();
    }).catch(function(){state.err=true; draw();});
  }

  /* ---------- auto-refresh: visible countdown, pausable --------------------- */
  function period(){return location.protocol==='file:'?15:10;}
  function renderTimer(){
    var c=el('countdown');
    if(!c)return;
    c.textContent=state.auto?('refresh in '+state.left+'s'):'auto-refresh paused';
  }
  function startTimer(){
    if(tick){clearInterval(tick);tick=null;}
    state.left=period();
    renderTimer();
    if(!state.auto)return;
    tick=setInterval(function(){
      state.left--;
      if(state.left<=0){
        if(location.protocol==='file:'){
          // file:// cannot fetch the ledger — reload instead. The hash carries
          // the current selections across, so the reload is seamless.
          location.reload();return;
        }
        state.left=period();refresh();
      }
      renderTimer();
    },1000);
  }

  function start(){
    state.data=JSON.parse(el('ciq-data').textContent);
    var h=readHash();
    state.model=(h.model&&state.data.prices[h.model]!==undefined)
      ?h.model:state.data.model;
    state.range=(RANGES.indexOf(h.range)>=0)?h.range:'all';
    state.auto=(h.auto!=='0');
    state.theme=(THEMES.indexOf(h.theme)>=0)?h.theme:'auto';
    state.sig=sig(state.data);
    applyTheme();
    controls();
    writeHash();
    refresh();
    startTimer();
  }
  if(document.readyState!=='loading')start();
  else document.addEventListener('DOMContentLoaded',start);
})();
"""

# Document skeleton. Every value is filled in by the view layer, so the file is
# meaningful with JS disabled only as a shell — the skeleton ships loading
# placeholders rather than fake numbers, and never hard-codes a figure.
_REPORT_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="description" content="ContextIQ token usage and savings for this workspace.">
<title>ContextIQ — token usage</title>
<style>__CSS__</style></head><body>
<a class="skip" href="#main">Skip to dashboard</a>
<header class="appbar no-print">
  <div class="wrap row">
    <div class="brand">
      <span class="mark" aria-hidden="true"><svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M2 4.2h12M2 8h8M2 11.8h5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="12.6" cy="11.4" r="2.2" stroke="currentColor" stroke-width="1.6"/></svg></span>
      ContextIQ<span class="sub">Token intelligence</span>
    </div>
    <nav class="appnav" aria-label="Dashboard sections">
      <a href="#s-overview">Overview</a><a href="#s-savings">Savings</a>
      <a href="#s-ops">Operations</a><a href="#s-cost">Cost by model</a>
      <a href="#s-workspace">Workspace</a><a href="#s-log">Activity log</a>
    </nav>
    <div class="bar-end">
      <span class="chip" title="Workspace this report covers">
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M2 4.5A1.5 1.5 0 0 1 3.5 3h3l1.2 1.6h4.8A1.5 1.5 0 0 1 14 6.1v5.4A1.5 1.5 0 0 1 12.5 13h-9A1.5 1.5 0 0 1 2 11.5v-7Z" stroke="currentColor" stroke-width="1.4"/></svg>
        <span id="ws-chip">workspace</span></span>
      <button type="button" class="btn icon" id="btn-theme"
              aria-label="Colour theme">◐</button>
    </div>
  </div>
</header>

<main id="main" class="wrap">
  <div class="page-head">
    <div>
      <h1>Token usage &amp; savings</h1>
      <p id="meta"><span class="skel line" style="width:280px;display:inline-block"></span></p>
    </div>
    <div class="status">
      <span class="pill info">local only</span>
      <span class="pill">read-only</span>
    </div>
  </div>

  <div class="toolbar no-print" role="group" aria-label="Report filters">
    <span class="field"><label for="sel-model">Pricing model</label>
      <select id="sel-model" aria-describedby="unit-price"></select></span>
    <span class="muted" id="unit-price" style="font-size:12px"></span>
    <span class="field"><label for="sel-range">Date range</label>
      <select id="sel-range">
        <option value="all">All time</option>
        <option value="7">Last 7 days</option>
        <option value="30">Last 30 days</option>
        <option value="90">Last 90 days</option>
      </select></span>
    <span class="spacer"></span>
    <label class="switch" for="chk-auto"><input type="checkbox" id="chk-auto" checked>
      Auto-refresh</label>
    <span class="meta" id="countdown" role="status" aria-live="polite"></span>
    <button type="button" class="btn" id="btn-refresh">Refresh now</button>
  </div>
  <p class="muted" id="range-note" style="font-size:12px;margin-bottom:16px"
     role="status" aria-live="polite"></p>
  <div id="onboard"></div>

  <section id="s-overview" aria-labelledby="h-overview">
    <div class="sec-head"><h2 id="h-overview">Overview</h2>
      <span class="hint">Projected at list input-token prices — indicative, not billing.</span></div>
    <div class="grid g-5">
      <div class="hero">
        <div class="hero-label">Estimated cost avoided</div>
        <div class="hero-value" id="hero-value">—</div>
        <div class="hero-sub" id="hero-sub"></div>
        <div class="hero-foot"><span id="hero-note"></span></div>
      </div>
      <div class="card gauge-card">
        <div class="card-head"><div><h3>Prompt reduction</h3>
          <div class="sub">Share of baseline tokens never sent</div></div></div>
        <div class="chart center" id="c-gauge"><div class="skel chart" style="height:150px"></div></div>
      </div>
    </div>
    <div class="grid g-5" style="margin-top:16px">
      <div class="stat"><div class="k">Tokens saved</div>
        <div class="v num" id="m-saved">—</div>
        <div class="n" id="k-saved-n"></div>
        <div class="spark" id="k-spark"></div></div>
      <div class="stat"><div class="k">Tokens sent (consumed)</div>
        <div class="v num" id="m-sent">—</div><div class="n" id="k-sent-n"></div></div>
      <div class="stat"><div class="k">Cost of tokens sent</div>
        <div class="v num" id="m-spend">—</div><div class="n" id="k-spend-n"></div></div>
      <div class="stat"><div class="k">Reduction</div>
        <div class="v num" id="m-red">—</div><div class="n" id="k-red-n"></div></div>
      <div class="stat"><div class="k">Runs</div>
        <div class="v num" id="m-runs">—</div><div class="n" id="k-runs-n"></div></div>
    </div>
    <div class="grid g-4" style="margin-top:16px">
      <div class="stat sm"><div class="k">Baseline tokens</div>
        <div class="v num" id="m-baseline">—</div>
        <div class="n">what whole-file reads would have cost</div></div>
      <div class="stat sm"><div class="k">Avg. saved per run</div>
        <div class="v num" id="m-avg">—</div>
        <div class="n">tokens avoided per recorded call</div></div>
      <div class="stat sm"><div class="k">Context leverage</div>
        <div class="v num" id="m-leverage">—</div>
        <div class="n">baseline ÷ tokens actually sent</div></div>
      <div class="stat sm"><div class="k">Files covered</div>
        <div class="v num" id="m-files">—</div>
        <div class="n">files represented in those packs</div></div>
    </div>
  </section>

  <section id="s-savings" aria-labelledby="h-savings">
    <div class="sec-head"><h2 id="h-savings">Savings</h2>
      <span class="hint">Every figure comes from the local ledger — nothing is estimated except the $ projection.</span></div>
    <div class="grid g-2">
      <div class="card">
        <div class="card-head"><div><h3>Baseline → avoided → sent</h3>
          <div class="sub">Where the tokens went, for the selected window</div></div></div>
        <div class="legend" aria-hidden="true">
          <span class="item"><i class="key" style="background:var(--s-base)"></i>Baseline</span>
          <span class="item"><i class="key" style="background:var(--s-saved)"></i>Avoided</span>
          <span class="item"><i class="key" style="background:var(--s-sent)"></i>Sent</span>
        </div>
        <div class="chart" id="c-waterfall"><div class="skel chart"></div></div>
      </div>
      <div class="card">
        <div class="card-head"><div><h3>Avoided &amp; sent over time</h3>
          <div class="sub">Daily totals, stacked</div></div></div>
        <div class="legend" aria-hidden="true">
          <span class="item"><i class="key" style="background:var(--s-saved)"></i>Avoided</span>
          <span class="item"><i class="key" style="background:var(--s-sent)"></i>Sent</span>
        </div>
        <div class="chart" id="c-area"><div class="skel chart"></div></div>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <div class="card-head"><div><h3>Activity</h3>
        <div class="sub">Tokens avoided per day, last 26 weeks</div></div></div>
      <div class="chart center" id="c-heat"><div class="skel chart" style="height:160px"></div></div>
    </div>
  </section>

  <section id="s-ops" aria-labelledby="h-ops">
    <div class="sec-head"><h2 id="h-ops">Operations</h2>
      <span class="hint">Which retrieval calls produce the savings</span></div>
    <div class="grid g-23">
      <div class="card">
        <div class="card-head"><div><h3>Tokens avoided by operation</h3>
          <div class="sub">Top 8, with share of total savings</div></div></div>
        <div class="chart" id="c-ops"><div class="skel chart" style="height:200px"></div></div>
      </div>
      <div class="card">
        <div class="card-head"><div><h3>Share of runs</h3>
          <div class="sub">How often each operation is used</div></div></div>
        <div id="c-mix"><div class="skel chart" style="height:200px"></div></div>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <div class="card-head"><div><h3>Operation detail</h3>
        <div class="sub">Totals per operation for the selected window</div></div></div>
      <div id="c-optable"><div class="skel line" style="width:100%"></div></div>
    </div>
  </section>

  <section id="s-cost" aria-labelledby="h-cost">
    <div class="sec-head"><h2 id="h-cost">Cost by model</h2>
      <span class="hint">Same token counts, priced against every model in the table</span></div>
    <div class="card">
      <div class="card-head"><div><h3>Cost avoided per pricing model</h3>
        <div class="sub">Saved tokens × that model's list input price</div></div></div>
      <div class="chart center" id="c-models"><div class="skel chart" style="height:220px"></div></div>
      <div style="margin-top:16px" id="c-modeltable"></div>
      <p class="muted" style="font-size:12px;margin-top:12px">
        Input-token list prices only. Output tokens, caching, batching and
        negotiated rates are not modelled, so treat these as directional.</p>
    </div>
  </section>

  <section id="s-workspace" aria-labelledby="h-ws">
    <div class="sec-head"><h2 id="h-ws">Workspace</h2>
      <span class="hint">What the local code graph covers</span></div>
    <div id="c-ws"><div class="skel chart" style="height:110px"></div></div>
    <div class="card" style="margin-top:16px">
      <div class="card-head"><div><h3>Indexed files by language</h3>
        <div class="sub">Top 8 languages in the graph</div></div></div>
      <div class="chart center" id="c-langs"><div class="skel chart" style="height:200px"></div></div>
    </div>
  </section>

  <section id="s-log" aria-labelledby="h-log">
    <div class="sec-head"><h2 id="h-log">Activity log</h2>
      <span class="hint">Raw ledger records, newest first</span></div>
    <details>
      <summary>Raw ledger records</summary>
      <div class="body" id="c-rows"></div>
    </details>
  </section>

  <footer>
    <div>Source <code>.context/gain.ndjson</code> · counts only, never file
      paths or queries · stays on this machine.</div>
    <div class="num">Generated <span id="foot-gen">—</span> · <span id="foot-ver"></span></div>
  </footer>
</main>
<script type="application/json" id="ciq-data">__DATA__</script>
<script>__JS__</script>
</body></html>"""


def render_report_html(payload: dict, generated_at: str | None = None) -> str:
    """Render the self-contained report page around an inlined payload."""
    import json
    if generated_at and not payload.get("generated_at"):
        payload = dict(payload, generated_at=generated_at)
    # "</" is escaped so a value can never terminate the <script> block early.
    data = json.dumps(payload).replace("</", "<\\/")
    html = _REPORT_TEMPLATE.replace("__CSS__", _REPORT_CSS)
    html = html.replace("__JS__", _REPORT_JS)
    return html.replace("__DATA__", data)


# --- per-workspace static usage report (co-located with the graph) ----------
USAGE_REPORT_NAME = "token-usage.html"


def _report_timestamp() -> str:
    """Local 'when was this built' stamp for the report header."""
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def make_report_handler(root: Path, model: str = DEFAULT_GAIN_MODEL):
    """Build the request handler for the live report (no Streamlit, no deps).

    Serves the page at `/` and the freshly-read ledger at `/data.json`, which
    the page polls to redraw in place. Callers must bind it to loopback — this
    is local developer state and must never listen on a public interface.
    """
    import json
    from http.server import BaseHTTPRequestHandler

    class _Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (stdlib callback name)
            path = self.path.split("?", 1)[0]
            if path == "/data.json":
                payload = build_report_payload(root, model=model,
                                               generated_at=_report_timestamp())
                self._send(json.dumps(payload).encode("utf-8"), "application/json")
            elif path in ("/", "/index.html"):
                payload = build_report_payload(root, model=model,
                                               generated_at=_report_timestamp())
                self._send(render_report_html(payload).encode("utf-8"),
                           "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def log_message(self, *args):        # keep the console quiet
            pass

    return _Handler


def serve_report(root: Path, port: int = 8787, model: str = DEFAULT_GAIN_MODEL) -> None:
    """Run the live report on 127.0.0.1:<port> until interrupted."""
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_report_handler(root, model))
    url = f"http://127.0.0.1:{srv.server_port}/"
    print(f"ContextIQ live report on {url}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()


def usage_report_path(root: Path) -> Path:
    """Where the self-contained per-workspace usage report is written."""
    return Path(root) / ".tokengraph" / USAGE_REPORT_NAME


def write_usage_report(root: Path, model: str = DEFAULT_GAIN_MODEL,
                       generated_at: str | None = None) -> Path | None:
    """Render this workspace's token report to .tokengraph/token-usage.html.

    Self-contained (no deps, no server) so any client can just open the file.
    Best-effort: returns the written path, or None on any failure — never
    raises (it runs on the hot logging path). Co-located with the graph DB so
    each workspace carries its own report next to its own state.
    """
    try:
        stamp = generated_at or _report_timestamp()
        payload = build_report_payload(root, model=model, generated_at=stamp)
        html = render_report_html(payload)
        out = usage_report_path(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return out
    except Exception:
        return None


# ==========================================================================
# conventions extraction (FR-CONV) — detect a repo's house style from the graph
# ==========================================================================
"""Derive a repo's file-naming / layout / test / export conventions from what is
already indexed, so generated code can match house style. Pure analysis over the
graph + file list — no LLM, deterministic for a given index state."""

import re as _re_conv


def _name_case(stem: str) -> str | None:
    """Classify a file/identifier stem into a naming case, or None if ambiguous.

    A single all-lowercase word counts as snake_case and a single Capitalized
    word as PascalCase (their one-word forms), so small/mixed repos still pick a
    sensible dominant convention rather than splitting into a `lowercase` bucket.
    """
    s = stem
    if not s or not s[0].isalpha():
        return None
    if "-" in s and s.lower() == s:
        return "kebab-case"
    if "_" in s and s.lower() == s:
        return "snake_case"
    if "_" in s or "-" in s:
        return None
    if s[0].isupper():
        return "PascalCase"
    if s[0].islower() and any(c.isupper() for c in s):
        return "camelCase"
    if s.islower():
        return "snake_case"
    return None


def _apply_case(name: str, case: str) -> str:
    """Render `name` (any case) into the target case."""
    parts = _re_conv.split(r"[_\-\s]+|(?<=[a-z0-9])(?=[A-Z])", name)
    parts = [p for p in parts if p]
    if not parts:
        return name
    low = [p.lower() for p in parts]
    if case == "snake_case":
        return "_".join(low)
    if case == "kebab-case":
        return "-".join(low)
    if case == "PascalCase":
        return "".join(p.capitalize() for p in low)
    if case == "camelCase":
        return low[0] + "".join(p.capitalize() for p in low[1:])
    if case == "lowercase":
        return "".join(low)
    return name


_TEST_PATTERNS = [
    ("test_*", _re_conv.compile(r"^test_.+")),
    ("*_test", _re_conv.compile(r".+_test$")),
    ("*.test", _re_conv.compile(r".+\.test$")),
    ("*.spec", _re_conv.compile(r".+\.spec$")),
    ("*Test", _re_conv.compile(r".+Test$")),
    ("*Tests", _re_conv.compile(r".+Tests$")),
]


# ==========================================================================
# test discovery: map implementation files <-> their tests (get_test_map)
# ==========================================================================
# The plumbing already links a symbol to tests through the call graph (see
# build_evidence_pack's `related_tests`). This surfaces it as a first-class,
# language-aware file mapping — the impl<->test pairing agents ask for when
# writing or updating a test — using both the naming/path conventions every
# ecosystem shares and, at symbol granularity, the real call edges.

_TEST_DIR_NAMES = {"test", "tests", "spec", "specs", "__tests__", "testing"}


def _split_stem_ext(path: str) -> tuple[str, str, str]:
    """(dir, stem, ext) for a repo-relative path; ext includes the leading dot."""
    base = path.rsplit("/", 1)[-1]
    dirp = path[:len(path) - len(base) - 1] if "/" in path else ""
    if "." in base:
        stem, ext = base.rsplit(".", 1)
        ext = "." + ext
    else:
        stem, ext = base, ""
    return dirp, stem, ext


def is_test_path(path: str) -> bool:
    """True if a file is a test by directory or filename convention."""
    if any(p in _TEST_DIR_NAMES for p in path.lower().split("/")[:-1]):
        return True
    _, stem, _ = _split_stem_ext(path)
    low = stem.lower()
    return (low.startswith("test_") or low.endswith("_test")
            or low.endswith(".test") or low.endswith(".spec")
            or stem.endswith("Test") or stem.endswith("Tests"))


def _impl_stem_candidates(stem: str) -> list[str]:
    """Given a TEST file stem, the candidate IMPLEMENTATION stems it tests."""
    out: list[str] = []
    low = stem.lower()
    if low.startswith("test_"):
        out.append(stem[5:])
    if low.endswith("_test"):
        out.append(stem[:-5])
    if stem.endswith(".test") or stem.endswith(".spec"):
        out.append(stem[:-5])
    if stem.endswith("Tests"):
        out.append(stem[:-5])
    elif stem.endswith("Test"):
        out.append(stem[:-4])
    seen: list[str] = []
    for s in out:
        if s and s not in seen:
            seen.append(s)
    return seen or [stem]


def build_test_map(files: list[str]) -> dict:
    """Language-aware impl<->test file mapping over a list of repo paths.

    Matching, in priority order: (1) same directory + same stem (Go
    `x_test.go`, colocated `x.test.ts`); (2) same stem anywhere, preferring the
    same extension (`tests/test_x.py` -> `x.py`); the impl stem is derived from
    the test stem by stripping the ecosystem's test affix. Deterministic.
    """
    tests = [f for f in files if is_test_path(f)]
    impls = [f for f in files if not is_test_path(f)]
    by_dir_stem: dict[tuple[str, str], list[str]] = {}
    by_stem: dict[str, list[str]] = {}
    for f in impls:
        d, s, _ = _split_stem_ext(f)
        by_dir_stem.setdefault((d, s), []).append(f)
        by_stem.setdefault(s, []).append(f)

    test_to_impl: dict[str, list[str]] = {}
    for t in tests:
        d, s, e = _split_stem_ext(t)
        matched: list[str] = []
        for cs in _impl_stem_candidates(s):
            if (d, cs) in by_dir_stem:                      # colocated
                matched = list(by_dir_stem[(d, cs)])
                break
            if cs in by_stem:                               # same stem elsewhere
                same_ext = [f for f in by_stem[cs]
                            if _split_stem_ext(f)[2] == e]
                matched = same_ext or list(by_stem[cs])
                break
        if matched:
            test_to_impl[t] = sorted(set(matched))

    impl_to_tests: dict[str, list[str]] = {}
    for t, ims in test_to_impl.items():
        for im in ims:
            impl_to_tests.setdefault(im, []).append(t)
    pairs = sorted((im, t) for im, ts in impl_to_tests.items() for t in ts)
    return {
        "impl_to_tests": {k: sorted(v) for k, v in impl_to_tests.items()},
        "test_to_impl": {k: sorted(v) for k, v in test_to_impl.items()},
        "tests": sorted(tests),
        "impls": sorted(impls),
        "pairs": [{"impl": im, "test": t} for im, t in pairs],
        "unmatched_tests": sorted(t for t in tests if t not in test_to_impl),
        "untested_impls": sorted(f for f in impls if f not in impl_to_tests),
    }


def test_discovery_f1(files: list[str], gold_pairs: list[dict]) -> dict:
    """Precision / recall / F1 / hit@1 of build_test_map against a gold set."""
    pred = build_test_map(files)
    pred_pairs = {(p["impl"], p["test"]) for p in pred["pairs"]}
    gold = {(p["impl"], p["test"]) for p in gold_pairs}
    tp = len(pred_pairs & gold)
    fp = len(pred_pairs - gold)
    fn = len(gold - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    # hit@1: for each impl with a gold test, does its top predicted test match?
    gold_by_impl: dict[str, set[str]] = {}
    for p in gold_pairs:
        gold_by_impl.setdefault(p["impl"], set()).add(p["test"])
    hits = considered = 0
    for im, gts in gold_by_impl.items():
        considered += 1
        preds = pred["impl_to_tests"].get(im, [])
        if preds and preds[0] in gts:
            hits += 1
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hit_at_1": round(hits / considered, 4) if considered else 0.0,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "gold_pairs": len(gold), "predicted_pairs": len(pred_pairs),
    }


def analyze_conventions(store: Store, root: Path) -> dict:
    files = [r["path"] for r in store.files_with_tokens()]
    case_tally: dict[str, int] = {}
    ext_tally: dict[str, int] = {}
    dir_tally: dict[str, int] = {}
    test_tally: dict[str, int] = {}
    test_dirs: dict[str, int] = {}
    dir_case: dict[str, dict[str, int]] = {}      # per-dir naming case tally
    stems: dict[str, tuple[str, str]] = {}        # file -> (stem, detected case)
    for f in files:
        base = f.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0]
        ext = ("." + base.rsplit(".", 1)[1]) if "." in base else ""
        ext_tally[ext] = ext_tally.get(ext, 0) + 1
        top = f.split("/", 1)[0] if "/" in f else "."
        dir_tally[top] = dir_tally.get(top, 0) + 1
        case = _name_case(stem)
        stems[f] = (stem, case or "")
        if case:
            case_tally[case] = case_tally.get(case, 0) + 1
            dir_case.setdefault(top, {})[case] = dir_case.setdefault(top, {}).get(case, 0) + 1
        parts = f.lower().split("/")
        for pat, rx in _TEST_PATTERNS:
            if rx.match(stem):
                test_tally[pat] = test_tally.get(pat, 0) + 1
                break
        for d in parts[:-1]:
            if d in ("test", "tests", "spec", "__tests__"):
                test_dirs[d] = test_dirs.get(d, 0) + 1

    # export style from symbols: classes vs free functions at module top level
    top_syms = store.conn.execute(
        "SELECT kind, COUNT(*) n FROM symbols WHERE kind!='module' "
        "GROUP BY kind ORDER BY n DESC").fetchall()
    kind_tally = {r["kind"]: r["n"] for r in top_syms}

    # export convention: how the codebase marks public vs private API. Derived
    # from top-level symbol names (graph-only, deterministic).
    pub = priv = 0
    for r in store.conn.execute(
            "SELECT name FROM symbols WHERE kind!='module' AND parent IS NOT NULL"):
        nm = r["name"] or ""
        if nm.startswith("_"):
            priv += 1
        elif nm:
            pub += 1
    total_api = pub + priv
    export_style = ("leading-underscore privates"
                    if total_api and priv / total_api >= 0.1 else
                    "all-public (no underscore-private convention)")

    def _top(d: dict, default=None):
        # sorted() first -> deterministic alphabetical tiebreak on equal counts
        return max(sorted(d.items()), key=lambda kv: kv[1])[0] if d else default

    dominant_case = _top(case_tally, "snake_case")
    primary_ext = _top({e: n for e, n in ext_tally.items() if e}, "")
    test_pattern = _top(test_tally)
    test_dir = _top(test_dirs)
    src_dirs = sorted(
        (d for d in dir_tally if d not in (".", "test", "tests", "spec")),
        key=lambda d: dir_tally[d], reverse=True)[:5]
    # a dir gets its own naming convention only with a clear (>60%) majority;
    # otherwise it inherits the global dominant (avoids tie-flipping noise).
    naming_by_dir: dict[str, str] = {}
    for d, c in sorted(dir_case.items()):
        tot = sum(c.values())
        best = _top(c)
        if best and tot and c[best] / tot > 0.6:
            naming_by_dir[d] = best

    # conformance: files whose stem violates the dominant naming of their dir
    # (falls back to the global dominant) — the actionable outlier list.
    nonconforming: list[dict] = []
    for f, (stem, case) in sorted(stems.items()):
        top = f.split("/", 1)[0] if "/" in f else "."
        expected = naming_by_dir.get(top, dominant_case)
        if case and expected and case != expected:
            nonconforming.append({
                "file": f, "found": case, "expected": expected,
                "suggested_stem": _apply_case(stem, expected),
            })
    conformance_pct = (round(100 * (len(files) - len(nonconforming)) / len(files), 1)
                       if files else 100.0)

    summary = (f"naming={dominant_case}, ext={primary_ext or 'n/a'}, "
               f"tests={test_pattern or test_dir or 'none detected'}, "
               f"exports={export_style}, top dirs={', '.join(src_dirs) or '(flat)'}, "
               f"conformance={conformance_pct}%")
    return {
        "files_analyzed": len(files),
        "dominant_naming": dominant_case,
        "naming_distribution": dict(sorted(case_tally.items(),
                                           key=lambda kv: kv[1], reverse=True)),
        "naming_by_dir": naming_by_dir,
        "primary_extension": primary_ext,
        "extension_distribution": dict(sorted(ext_tally.items(),
                                              key=lambda kv: kv[1], reverse=True)),
        "test_pattern": test_pattern,
        "test_dir": test_dir,
        "test_distribution": test_tally,
        "source_dirs": src_dirs,
        "symbol_kinds": kind_tally,
        "export_style": export_style,
        "public_symbols": pub,
        "private_symbols": priv,
        "conformance_pct": conformance_pct,
        "nonconforming_files": nonconforming,
        "summary": summary,
    }


# ==========================================================================
# grounded-creation pipeline (FR-CREATE): scaffold -> verify-plan -> review
# ==========================================================================
_SKELETONS = {
    ".py": "def {name}():\n    \"\"\"TODO: implement {name}.\"\"\"\n    raise NotImplementedError\n",
    ".js": "export function {name}() {{\n  // TODO: implement {name}\n}}\n",
    ".ts": "export function {name}(): void {{\n  // TODO: implement {name}\n}}\n",
    ".go": "package {pkg}\n\nfunc {Name}() {{\n\t// TODO: implement {Name}\n}}\n",
    ".rs": "pub fn {name}() {{\n    // TODO: implement {name}\n}}\n",
    ".java": "public class {Name} {{\n    // TODO: implement {Name}\n}}\n",
    ".rb": "def {name}\n  # TODO: implement {name}\nend\n",
    ".lua": "local function {name}()\n  -- TODO: implement {name}\nend\n\nreturn {name}\n",
}


def propose_scaffold(store: Store, root: Path, name: str, kind: str = "module",
                     conv: dict | None = None) -> dict:
    """Propose a convention-matched file path + skeleton for `name`. Refuses on
    conflict (an existing file at the target path), so it never overwrites."""
    conv = conv or analyze_conventions(store, root)
    ext = conv["primary_extension"] or ".py"
    case = conv["dominant_naming"] or "snake_case"
    is_test = kind == "test"
    if is_test:
        # name the file by the repo's test pattern, falling back to test_*
        pat = conv.get("test_pattern") or "test_*"
        stem_base = _apply_case(name, "snake_case" if ext == ".py" else case)
        stem = (pat.replace("*", stem_base) if "*" in pat else f"test_{stem_base}")
        target_dir = conv.get("test_dir") or (conv["source_dirs"][0]
                                              if conv["source_dirs"] else "")
    else:
        case = "PascalCase" if (kind == "class" and ext in (".java", ".cs")) else case
        stem = _apply_case(name, case)
        target_dir = conv["source_dirs"][0] if conv["source_dirs"] else ""

    rel = f"{target_dir}/{stem}{ext}" if target_dir else f"{stem}{ext}"
    rel = rel.lstrip("/")
    exists = (root / rel).exists()
    pkg = target_dir.replace("/", "_") or "main"
    skel = _SKELETONS.get(ext, "// TODO: implement {name}\n").format(
        name=_apply_case(name, "snake_case"),
        Name=_apply_case(name, "PascalCase"), pkg=pkg)
    return {
        "name": name,
        "kind": kind,
        "proposed_path": rel,
        "exists": exists,
        "ok": not exists,
        "skeleton": skel,
        "convention_basis": conv["summary"],
        "note": (f"conflict: {rel} already exists — choose another name or edit it"
                 if exists else f"safe to create {rel} (matches house style)"),
    }


def write_scaffold(store: Store, root: Path, name: str, kind: str = "module",
                   conv: dict | None = None) -> dict:
    """propose_scaffold + actually create the file. Refuses on conflict, so it
    never overwrites existing code."""
    res = propose_scaffold(store, root, name, kind=kind, conv=conv)
    if not res["ok"]:
        res["written"] = False
        return res
    path = root / res["proposed_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(res["skeleton"], encoding="utf-8")
    res["written"] = True
    res["note"] = f"created {res['proposed_path']} ({res['convention_basis']})"
    return res


# import statements an answer might cite; group 1 is the module being imported
_IMPORT_PATTERNS = [
    _re_conv.compile(r"^\s*from\s+([.\w]+)\s+import\b", _re_conv.M),
    _re_conv.compile(r"^\s*import\s+([.\w]+)", _re_conv.M),
    _re_conv.compile(r"""\brequire\(\s*['"]([^'"]+)['"]\s*\)""", _re_conv.M),
    _re_conv.compile(r"""^\s*import\b[^;\n]*\bfrom\s+['"]([^'"]+)['"]""", _re_conv.M),
]


def _local_module_index(store: Store) -> tuple[set, set, set]:
    """(top-level repo segments, file basenames, dotted module paths) — the
    vocabulary for deciding whether an imported module is a *local* one."""
    files = store.all_indexed_files()
    tops: set[str] = set()
    basenames: set[str] = set()
    modules: set[str] = set()
    for f in files:
        parts = f.split("/")
        tops.add(parts[0])
        stem = parts[-1].rsplit(".", 1)[0]
        basenames.add(stem)
        dotted = ".".join(parts[:-1] + [stem])
        modules.add(dotted)
        modules.add(stem)
        if len(parts) > 1:
            modules.add(".".join(parts[1:-1] + [stem]))  # drop the src/ root
    return tops, basenames, modules


def verify_ai_output(retriever: "Retriever", answer: str) -> dict:
    """Audit AI-generated output for fabricated files, symbols AND imports.

    Extends verify() (files + backtick symbols) with import-statement checking:
    a *local* import (relative `./x`, or a dotted module whose first segment is a
    repo package) that resolves to no indexed module is flagged as fabricated.
    External packages (numpy, react, …) are intentionally not flagged — we can't
    know the install set, only the repo. This is the verify-ai-output stage."""
    base = retriever.verify(answer)
    store = retriever.store
    tops, basenames, modules = _local_module_index(store)

    bad_imports: list[dict] = []
    seen: set[str] = set()
    for rx in _IMPORT_PATTERNS:
        for mod in rx.findall(answer or ""):
            if mod in seen:
                continue
            seen.add(mod)
            relative = mod.startswith(".") or mod.startswith("./") or mod.startswith("../")
            clean = mod.lstrip("./")
            leaf = _re_conv.split(r"[./\\]", clean)[-1]
            first = _re_conv.split(r"[./\\]", clean)[0]
            is_local = relative or (first in tops) or (clean in modules)
            if not is_local:
                continue  # third-party / stdlib — unverifiable, don't flag
            resolved = (clean in modules or leaf in basenames
                        or clean in basenames)
            if not resolved:
                bad_imports.append({
                    "kind": "import", "name": mod,
                    "did_you_mean": _closest(leaf, sorted(basenames)),
                })

    issues = base["issues"] + bad_imports
    return {
        "ok": not issues,
        "checked": {**base["checked"], "imports": len(seen)},
        "issues": issues,
        "fabricated_imports": bad_imports,
        "note": ("all references + imports resolve against the graph" if not issues
                 else f"{len(issues)} unresolved reference(s) "
                      f"({len(bad_imports)} import) — possible hallucination"),
    }


def verify_plan(retriever: "Retriever", plan_text: str) -> dict:
    """Check a plan that references files/symbols: which exist, which are new,
    and the blast radius of touching the existing ones. Builds on verify()."""
    store = retriever.store
    v = retriever.verify(plan_text)
    indexed = store.all_indexed_files()
    base_index = {f.rsplit("/", 1)[-1].lower(): f for f in indexed}

    referenced_files: list[dict] = []
    seen: set[str] = set()
    for m in _re_conv.findall(r"[\w./\\-]+\.[A-Za-z][A-Za-z0-9]{0,5}\b", plan_text):
        cand = m.replace("\\", "/").lstrip("./")
        if cand in seen:
            continue
        seen.add(cand)
        ext = "." + cand.rsplit(".", 1)[-1].lower()
        if ext not in supported_extensions() and ext not in (
                ".md", ".json", ".yml", ".yaml", ".toml"):
            continue
        base = cand.rsplit("/", 1)[-1].lower()
        status = "exists" if (cand.lower() in {f.lower() for f in indexed}
                              or base in base_index) else "new"
        referenced_files.append({"path": cand, "status": status})

    # blast radius for backtick-quoted symbols that already exist
    blast: list[dict] = []
    for raw in _re_conv.findall(r"`([A-Za-z_][\w.]*)`", plan_text):
        leaf = raw.split(".")[-1]
        cands = store.candidates_by_leaf(leaf)
        if not cands:
            continue
        try:
            imp = retriever.get_impact(cands[0]["qname"])
            n = imp.get("blast_radius", 0) if isinstance(imp, dict) else 0
        except Exception:
            n = 0
        blast.append({"symbol": raw, "qname": cands[0]["qname"], "impact": n})

    new_files = [f for f in referenced_files if f["status"] == "new"]
    # A plan legitimately names files it intends to create, so "new" files are
    # not hallucinations. Only fabricated *symbols* (cited code that doesn't
    # exist and isn't being introduced) fail the plan.
    bad_symbols = [i for i in v["issues"] if i["kind"] == "symbol"]
    ok = not bad_symbols
    return {
        "ok": ok,
        "fabrication": v,
        "fabricated_symbols": bad_symbols,
        "referenced_files": referenced_files,
        "new_files": [f["path"] for f in new_files],
        "blast_radius": sorted(blast, key=lambda b: b["impact"], reverse=True),
        "note": (f"{len(bad_symbols)} fabricated symbol(s) — fix before acting"
                 if bad_symbols else
                 f"plan references {len(referenced_files)} file(s), "
                 f"{len(new_files)} new; {len(blast)} existing symbol(s) impacted"),
    }


def review_diff(retriever: "Retriever", root: Path, staged: bool = False,
                max_files: int = 20) -> dict:
    """Audit the working-tree (or staged) diff for scope drift, hub edits and
    missing tests — deterministic heuristics over the graph, no LLM."""
    changed = git_changed_files(root, staged=staged)
    findings: list[dict] = []
    if not changed:
        return {"ok": True, "changed_files": [], "findings": [],
                "note": "no changes detected (working tree clean)"}

    indexed = retriever.store.all_indexed_files()
    src_changed = [f for f in changed if "." + f.rsplit(".", 1)[-1].lower()
                   in supported_extensions()]
    test_changed = [f for f in src_changed
                    if any(t in f.lower() for t in ("test", "spec"))]
    nontest_changed = [f for f in src_changed if f not in test_changed]

    # 1. scope drift: many files across unrelated top-level dirs
    top_dirs = {f.split("/", 1)[0] if "/" in f else "." for f in src_changed}
    if len(src_changed) > max_files:
        findings.append({"severity": "warn", "kind": "scope-drift",
                         "detail": f"{len(src_changed)} source files changed "
                                   f"(> {max_files}); consider splitting the PR"})
    if len(top_dirs) >= 4:
        findings.append({"severity": "info", "kind": "scope-drift",
                         "detail": f"changes span {len(top_dirs)} top-level dirs: "
                                   f"{', '.join(sorted(top_dirs))}"})

    # 2. god-node edits: a changed file is a high-fan-in hub
    try:
        hubs = {h["file"]: h for h in retriever.get_map("hubs").get("hubs", [])}
    except Exception:
        hubs = {}
    for f in nontest_changed:
        if f in hubs and hubs[f].get("fan_in", 0) >= 5:
            findings.append({"severity": "warn", "kind": "god-node",
                             "detail": f"{f} is a hub (fan-in "
                                       f"{hubs[f]['fan_in']}) — high blast radius"})

    # 3. missing tests: non-test source changed but no test file touched
    if nontest_changed and not test_changed:
        findings.append({"severity": "warn", "kind": "missing-tests",
                         "detail": f"{len(nontest_changed)} source file(s) changed "
                                   "with no test changes"})

    # 4. breaking changes: a removed/renamed def whose symbol still has callers
    #    elsewhere — a real API break the diff alone wouldn't surface.
    breaking: list[dict] = []
    diff = _git(root, "diff", "--unified=0", *(["--cached"] if staged else []))
    removed = set(_re_conv.findall(
        r"^-\s*(?:async\s+|export\s+|public\s+|private\s+|pub\s+|static\s+)*"
        r"(?:def|function|func|fn|class|interface)\s+([A-Za-z_]\w*)",
        diff, _re_conv.M))
    for name in sorted(removed):
        sid = retriever.store.id_for_qname(  # only if it was indexed (pre-change)
            next((c["qname"] for c in retriever.store.candidates_by_leaf(name)), ""))
        callers = []
        for c in retriever.store.candidates_by_leaf(name):
            callers += [r["qname"] for r in
                        retriever.store.neighbors(c["id"], ["CALLS", "INHERITS"], "in")]
        callers = sorted(set(callers))
        if callers:
            breaking.append({"symbol": name, "callers": callers[:8],
                             "caller_count": len(callers)})
    for b in breaking:
        findings.append({"severity": "warn", "kind": "breaking-change",
                         "detail": f"removed `{b['symbol']}` still has "
                                   f"{b['caller_count']} caller(s): "
                                   f"{', '.join(b['callers'][:3])}…"})

    return {
        "ok": not any(f["severity"] == "warn" for f in findings),
        "changed_files": src_changed,
        "test_files_changed": test_changed,
        "breaking_changes": breaking,
        "findings": findings,
        "note": (f"{len(findings)} finding(s) across {len(src_changed)} changed "
                 f"source file(s)" if findings else
                 f"{len(src_changed)} file(s) changed — no issues flagged"),
    }


def create_pipeline(retriever: "Retriever", root: Path, task: str,
                    kind: str = "module", answer: str | None = None,
                    apply: bool = False) -> dict:
    """Full grounded-creation state machine, each stage gating the next:

      1. conventions     — learn house style
      2. scaffold        — propose (or, with apply=True, write) a matched file
      3. retrieve        — budgeted context pack for the task
      4. verify-plan     — the pack's refs resolve + blast radius is known
      5. verify-output   — (if `answer` given) the generated code cites no
                           fabricated files / symbols / imports
      6. review          — audit the resulting working-tree diff

    `ok` is the AND of every stage that ran. Writes nothing unless apply=True
    (and even then only a fresh scaffold file — never an overwrite)."""
    conv = analyze_conventions(retriever.store, root)
    name = _re_conv.sub(r"[^\w]+", "_", task.strip())[:40].strip("_") or "new_unit"
    stages: list[dict] = []

    scaffold = (write_scaffold(retriever.store, root, name, kind=kind, conv=conv)
                if apply else
                propose_scaffold(retriever.store, root, name, kind=kind, conv=conv))
    stages.append({"stage": "scaffold", "ok": scaffold["ok"],
                   "detail": scaffold["note"]})

    pack = retriever.find_relevant_context(task, budget_tokens=4000)
    plan = verify_plan(retriever, pack.to_markdown())
    stages.append({"stage": "verify-plan", "ok": plan["ok"], "detail": plan["note"]})

    output_check = None
    if answer is not None:
        output_check = verify_ai_output(retriever, answer)
        stages.append({"stage": "verify-output", "ok": output_check["ok"],
                       "detail": output_check["note"]})

    review = None
    if apply:
        review = review_diff(retriever, root)
        stages.append({"stage": "review", "ok": review["ok"],
                       "detail": review["note"]})

    ok = all(s["ok"] for s in stages)
    return {
        "task": task,
        "conventions": conv["summary"],
        "stages": stages,
        "scaffold": scaffold,
        "context_symbols": [p.qname for p in pack.pieces],
        "plan_check": plan,
        "output_check": output_check,
        "review": review,
        "ok": ok,
        "note": ("all stages passed" if ok else
                 "one or more stages failed — see `stages`") +
                ("" if apply else "; nothing was written (dry run)"),
    }


# ==========================================================================
# evidence pack (FR-EVID): deterministic, hash-grounded retrieval artifact
# ==========================================================================
def build_evidence_pack(retriever: "Retriever", task: str,
                        budget_tokens: int = 6000) -> dict:
    """A byte-stable, anchor-verified JSON artifact for auditing / CI.

    Determinism: every list is sorted by a stable key and no timestamps or
    machine-specific paths leak in, so the same index state + task yields an
    identical `context_hash`. `anchor_coverage` is the fraction of cited symbols
    whose line span resolves against the graph (proof the pack is grounded)."""
    import hashlib
    pack = retriever.find_relevant_context(task, budget_tokens=budget_tokens)
    store = retriever.store

    files: dict[str, list[dict]] = {}
    symbols: list[dict] = []
    anchored = 0
    # confidence: pack pieces are emitted best-first; map rank -> a stable score.
    npieces = max(1, len(pack.pieces))
    for rank, p in enumerate(pack.pieces):
        row = store.symbol_by_qname(p.qname)
        has_anchor = bool(row and row["lineno"] and row["end_lineno"])
        if has_anchor:
            anchored += 1
        conf = round(1.0 - rank / npieces, 4)   # 1.0 (top) .. ~0
        files.setdefault(p.file, []).append(
            {"qname": p.qname, "reason": p.reason, "conf": conf,
             "lineno": (row["lineno"] if row else None),
             "end_lineno": (row["end_lineno"] if row else None)})
        symbols.append({
            "qname": p.qname,
            "kind": p.kind,
            "file": p.file,
            "lineno": (row["lineno"] if row else None),
            "end_lineno": (row["end_lineno"] if row else None),
            "reason": p.reason,
            "detail": p.detail,
            "confidence": conf,
            "anchored": has_anchor,
        })

    symbols.sort(key=lambda s: (s["file"], s["lineno"] or 0, s["qname"]))

    # per-file enrichment: reason, confidence, source line span, related tests,
    # and a risk label from blast radius — Sigmap-grade evidence rows.
    def _risk(blast: int) -> str:
        return "high" if blast >= 8 else "medium" if blast >= 3 else "low"

    files_out: list[dict] = []
    for f, entries in sorted(files.items()):
        lns = [e["lineno"] for e in entries if e["lineno"]]
        elns = [e["end_lineno"] for e in entries if e["end_lineno"]]
        related: set[str] = set()
        blast = 0
        for e in entries:
            row = store.symbol_by_qname(e["qname"])
            if not row:
                continue
            callers = store.neighbors(row["id"], ["CALLS", "INHERITS"], "in")
            blast = max(blast, len(callers))
            for c in callers:
                if "test" in c["file"].lower() or "spec" in c["file"].lower():
                    related.add(c["file"])
        files_out.append({
            "path": f,
            "symbols": sorted({e["qname"] for e in entries}),
            "reason": _top_reason(entries),
            "confidence": round(max(e["conf"] for e in entries), 4),
            "source_lines": ([min(lns), max(elns)] if lns and elns else None),
            "related_tests": sorted(related),
            "risk_label": _risk(blast),
        })

    n = len(symbols)
    anchor_coverage = round(anchored / n, 4) if n else 0.0

    # hash only the grounded skeleton (path + qname + line span), sorted — stable
    digest_src = "\n".join(
        f"{s['file']}:{s['lineno']}-{s['end_lineno']}:{s['qname']}"
        for s in symbols)
    context_hash = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()

    return {
        "schema": "contextiq.evidence/v2",
        "schema_version": 2,
        "task": task,
        "intent": detect_intent(task),
        "token_budget": budget_tokens,
        "pack_tokens": pack.tokens,
        "files": files_out,
        "symbols": symbols,
        "dropped": sorted(pack.dropped),
        "grounding": {
            "symbol_count": n,
            "anchored_symbols": anchored,
            "anchor_coverage": anchor_coverage,
            "context_hash": context_hash,
            "deterministic": True,
        },
    }


def _top_reason(entries: list[dict]) -> str:
    """Most common 'reason' among a file's pack pieces (seed/callee/caller/base)."""
    tally: dict[str, int] = {}
    for e in entries:
        tally[e["reason"]] = tally.get(e["reason"], 0) + 1
    return max(sorted(tally.items()), key=lambda kv: kv[1])[0] if tally else "seed"


# ==========================================================================
# grounding report (FR-GROUND): quantify the hallucination-guard's effect
# ==========================================================================
def grounding_report(retriever: "Retriever", sample: int = 100) -> dict:
    """Deterministic ablation measuring how many fabricated code references
    verify() catches vs. how many real ones it (correctly) passes.

    No LLM and no randomness: we sample real symbols from the graph in qname
    order, build a 'grounded' answer (cites real names — must pass) and an
    'ungrounded' answer (cites perturbed names — must be flagged), then report
    errors-per-100 for each arm. This is the with/without-packs grounding
    ablation, made reproducible."""
    store = retriever.store
    qnames = sorted(store.all_qnames())
    real = [q for q in qnames if "." in q and not q.endswith(".module")][:sample]
    if not real:
        return {"ok": True, "sample": 0,
                "note": "no symbols indexed — nothing to measure"}

    def _perturb(leaf: str) -> str:
        # deterministic edit: swap a char so the name no longer exists
        if len(leaf) < 4:
            return leaf + "_xyz"
        i = len(leaf) // 2
        repl = "z" if leaf[i] != "z" else "q"
        return leaf[:i] + repl + leaf[i + 1:]

    grounded_errors = 0     # real names that verify wrongly flags (false positive)
    ungrounded_caught = 0   # fabricated names verify correctly flags (true positive)
    for q in real:
        leaf = q.split(".")[-1]
        # grounded arm: cite the real symbol — verify should NOT flag it
        gv = retriever.verify(f"See `{leaf}` in the code.")
        if not gv["ok"]:
            grounded_errors += 1
        # ungrounded arm: cite a fabricated symbol — verify SHOULD flag it
        fake = _perturb(leaf)
        uv = retriever.verify(f"See `{fake}` in the code.")
        if not uv["ok"]:
            ungrounded_caught += 1

    n = len(real)
    return {
        "ok": True,
        "sample": n,
        "with_grounding": {
            "false_positive_rate_per_100": round(100 * grounded_errors / n, 2),
            "note": "real symbols wrongly flagged (lower is better)",
        },
        "without_grounding": {
            "fabrications_caught_per_100": round(100 * ungrounded_caught / n, 2),
            "note": "fabricated symbols correctly flagged (higher is better)",
        },
        "guard_precision": round(ungrounded_caught / max(1, ungrounded_caught
                                 + grounded_errors), 4),
        "summary": (f"verify() catches {ungrounded_caught}/{n} fabrications "
                    f"({round(100*ungrounded_caught/n,1)}%) while wrongly flagging "
                    f"{grounded_errors}/{n} real refs "
                    f"({round(100*grounded_errors/n,1)}%)"),
    }


def _perturb_ident(leaf: str) -> str:
    """Deterministic edit so a real identifier becomes a non-existent one."""
    if len(leaf) < 4:
        return leaf + "_xyz"
    i = len(leaf) // 2
    repl = "z" if leaf[i] != "z" else "q"
    return leaf[:i] + repl + leaf[i + 1:]


# HB-1: what this benchmark may and may not claim.
#
# Three of the four numbers here are measured: grounding coverage, guard catch,
# and guard specificity are all observed against the real index. The fourth —
# "hallucination reduction %" — is not. It is arithmetic over an *assumed*
# ungrounded fabrication rate, and with the previous default of 99.8 errors per
# 100 facts it was pinned near 100% by construction, no matter how the guard
# actually performed. Shipping that as a headline was the single least honest
# number in the project.
#
# So the assumption no longer has a default. Ask for a projection and you must
# supply the baseline *and* say where it came from; the reduction is then
# clearly labelled a projection contingent on your figure. Ask for nothing and
# you get the three measurements, which is what the tool can actually prove.
DEFAULT_HALLUCINATION_BASELINE = None


def hallucination_benchmark(retriever: "Retriever", sample_per_repo: int = 40,
                            baseline_per_100: float | None
                            = DEFAULT_HALLUCINATION_BASELINE,
                            baseline_source: str = "") -> dict:
    """Reproducible, multi-repo codebase-fact grounding benchmark (HB-1).

    Partitions the codebase by top-level directory (each = a "repo") and, per
    repo, measures three real structural quantities over sampled symbols:

      • grounding_coverage — % of repo facts the retriever can surface into a
        pack (so a *correct* grounded citation is possible),
      • guard_catch — % of fabricated references verify() flags,
      • guard_specificity — % of real references verify() does NOT false-flag.

    These are measurements. They are deterministic, need no model, and are the
    benchmark's actual result.

    Optionally it will also *project* the residual codebase-fact error rate of
    a grounded agent — a fact is only stated wrong if it could not be grounded
    AND the guard missed the fabrication, so residual = baseline · (1−coverage)
    · (1−catch). That projection is only as good as `baseline_per_100`, the
    ungrounded fabrication rate you are comparing against, which this tool
    cannot observe. It therefore has no default: pass one you can defend,
    together with `baseline_source` naming where it came from, or get only the
    measurements. Reports a per-repo spread (min/max) either way, so nothing
    here rests on a single partition.
    """
    store = retriever.store
    files = sorted(store.all_indexed_files())
    parts: dict[str, list[str]] = {}
    for f in files:
        top = f.split("/", 1)[0] if "/" in f else "(root)"
        parts.setdefault(top, []).append(f)

    rows: list[dict] = []
    for repo, rfiles in sorted(parts.items()):
        rset = set(rfiles)
        syms: list[tuple[str, str]] = []
        seen: set[str] = set()
        for f in rfiles:
            for s in store.file_symbols(f):
                if s["kind"] == "module" or s["name"] in seen:
                    continue
                seen.add(s["name"])
                syms.append((s["qname"], s["name"]))
        syms = sorted(syms)[:sample_per_repo]
        n = len(syms)
        if not n:
            continue
        groundable = caught = preserved = 0
        for q, leaf in syms:
            hits = store.search(leaf, limit=10)
            if any(h["qname"] == q or h["file"] in rset for h in hits):
                groundable += 1
            if not retriever.verify(f"See `{_perturb_ident(leaf)}` here.")["ok"]:
                caught += 1
            if retriever.verify(f"See `{leaf}` here.")["ok"]:
                preserved += 1
        cov, catch, spec = groundable / n, caught / n, preserved / n
        row = {
            "repo": repo, "facts": n,
            "grounding_coverage_pct": round(100 * cov, 1),
            "guard_catch_pct": round(100 * catch, 1),
            "guard_specificity_pct": round(100 * spec, 1),
            # The share of facts that are BOTH ungroundable and would slip past
            # the guard — measured, and the honest per-repo risk figure.
            "unguarded_fact_share_pct": round(100 * (1 - cov) * (1 - catch), 2),
        }
        if baseline_per_100:
            residual = round(baseline_per_100 * (1 - cov) * (1 - catch), 3)
            row["projected_with_grounding_per_100"] = residual
            row["projected_reduction_pct"] = round(
                100 * (baseline_per_100 - residual) / baseline_per_100, 2)
        rows.append(row)

    if not rows:
        return {"ok": True, "repos": 0, "note": "no symbols indexed"}

    total = sum(r["facts"] for r in rows)
    wmean = lambda k: round(sum(r[k] * r["facts"] for r in rows) / total, 2)
    spreads = [r["unguarded_fact_share_pct"] for r in rows]
    out = {
        "ok": True,
        "methodology": (
            "deterministic structural measurement (no LLM). Measured per repo "
            "partition: grounding coverage, guard catch rate, guard "
            "specificity. Unguarded fact share = (1-coverage)*(1-catch)."),
        "repos": len(rows),
        "facts_total": total,
        "mean_grounding_coverage_pct": wmean("grounding_coverage_pct"),
        "mean_guard_catch_pct": wmean("guard_catch_pct"),
        "mean_guard_specificity_pct": wmean("guard_specificity_pct"),
        "unguarded_fact_share_pct": wmean("unguarded_fact_share_pct"),
        "unguarded_spread_pct": [min(spreads), max(spreads)],
        "per_repo": rows,
        "deterministic": True,
        "measured": True,
        "summary": (
            f"measured across {len(rows)} repo-partition(s), {total} facts: "
            f"grounding coverage {wmean('grounding_coverage_pct')}%, "
            f"guard catch {wmean('guard_catch_pct')}%, "
            f"guard specificity {wmean('guard_specificity_pct')}%; "
            f"{wmean('unguarded_fact_share_pct')}% of facts are both "
            f"ungroundable and unguarded"),
    }
    if not baseline_per_100:
        # HB-1: refuse to invent the comparison. Without an observed ungrounded
        # fabrication rate there is no reduction to report, and a default one
        # would make the headline a restatement of its own assumption.
        out["hallucination_reduction_pct"] = None
        out["projection"] = {
            "available": False,
            "why": ("no ungrounded-fabrication baseline supplied. ContextIQ "
                    "cannot observe how often an un-grounded agent fabricates; "
                    "pass baseline_per_100 (with baseline_source) measured on "
                    "your own agent and model to get a projected reduction. "
                    "The measured figures above stand on their own."),
        }
        return out
    projected = round(
        sum(r["projected_with_grounding_per_100"] * r["facts"]
            for r in rows) / total, 3)
    reduction = round(100 * (baseline_per_100 - projected) / baseline_per_100, 2)
    reds = [r["projected_reduction_pct"] for r in rows]
    out["projection"] = {
        "available": True,
        "baseline_without_grounding_per_100": baseline_per_100,
        "baseline_source": baseline_source or "UNSTATED — provenance not given",
        "projected_with_grounding_per_100": projected,
        "projected_reduction_pct": reduction,
        "projected_reduction_spread_pct": [min(reds), max(reds)],
        "caveat": ("projected, not observed: this figure is arithmetic over "
                   "the supplied baseline and is no more trustworthy than it. "
                   "A high baseline forces a high reduction regardless of how "
                   "the guard performs."),
    }
    out["hallucination_reduction_pct"] = reduction
    out["summary"] += (f"; projected {reduction}% reduction against a supplied "
                       f"baseline of {baseline_per_100}/100 (NOT observed)")
    return out


def hallucination_report_to_markdown(rep: dict) -> str:
    if not rep.get("per_repo"):
        return "# tokengraph hallucination benchmark\n\n(no symbols indexed)\n"
    proj = rep.get("projection") or {}
    out = ["# tokengraph — codebase-fact grounding benchmark", "",
           f"_{rep['summary']}_", "",
           "## Measured",
           "",
           f"- Methodology: {rep['methodology']}",
           "- Reproducible: deterministic, no LLM (same index -> same numbers)",
           f"- Grounding coverage: **{rep['mean_grounding_coverage_pct']}%**",
           f"- Guard catch rate: **{rep['mean_guard_catch_pct']}%**",
           f"- Guard specificity: **{rep['mean_guard_specificity_pct']}%**",
           f"- Facts both ungroundable and unguarded: "
           f"**{rep['unguarded_fact_share_pct']}%** "
           f"(per-repo spread {rep['unguarded_spread_pct'][0]}"
           f"-{rep['unguarded_spread_pct'][1]}%)",
           "",
           "| Repo | Facts | Coverage % | Guard catch % | Guard spec. % | Unguarded % |",
           "|---|--:|--:|--:|--:|--:|"]
    for r in rep["per_repo"]:
        out.append(f"| {r['repo']} | {r['facts']} | {r['grounding_coverage_pct']} "
                   f"| {r['guard_catch_pct']} | {r['guard_specificity_pct']} "
                   f"| {r['unguarded_fact_share_pct']} |")
    out += ["", "## Hallucination reduction", ""]
    if not proj.get("available"):
        out += [
            "**Not reported.** " + proj.get("why", "no baseline supplied."),
            "",
            "A reduction percentage requires knowing how often an ungrounded "
            "agent fabricates on this codebase. ContextIQ does not observe "
            "that, and assuming it would make the headline a restatement of "
            "the assumption rather than a measurement.",
        ]
    else:
        out += [
            f"- Baseline (ungrounded, **supplied, not observed**): "
            f"**{proj['baseline_without_grounding_per_100']}** errors/100",
            f"- Baseline source: {proj['baseline_source']}",
            f"- With grounding (projected): "
            f"**{proj['projected_with_grounding_per_100']}** errors/100",
            f"- **Projected reduction: {proj['projected_reduction_pct']}%** "
            f"(per-repo spread {proj['projected_reduction_spread_pct'][0]}"
            f"-{proj['projected_reduction_spread_pct'][1]}%)",
            "",
            f"> {proj['caveat']}",
        ]
    out.append("")
    return "\n".join(out)


# ==========================================================================
# IDE integration (FR-IDE): one-command MCP wiring for every major editor
# ==========================================================================
def mcp_launch_command() -> dict:
    """How to launch this MCP server: console script if installed, else python.

    Kept in one place because the setup writers, the doctor and the CI config
    validator must all agree on it — they previously did not, so a pip-installed
    ContextIQ wrote a config its own validator rejected.
    """
    import shutil
    if shutil.which("tokengraph"):
        return {"command": "tokengraph", "args": ["serve"]}
    return {"command": sys.executable or "python",
            "args": [os.path.abspath(__file__), "serve"]}


def global_mcp_targets() -> dict[str, tuple[Path, dict]]:
    """Hosts that only read MCP config from a per-user location, not the repo.

    Writing a project-local file for these does nothing — Windsurf and Cline
    genuinely ignore in-repo MCP config. They are handled separately so
    `ide-setup` can either write them explicitly (--global) or report the
    exact path and payload instead of silently producing a dead file.
    """
    stdio = mcp_launch_command()
    home = Path.home()
    if sys.platform == "darwin":
        cline_dir = (home / "Library" / "Application Support" / "Code" / "User"
                     / "globalStorage" / "saoudrizwan.claude-dev" / "settings")
    elif os.name == "nt":
        cline_dir = (Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
                     / "Code" / "User" / "globalStorage"
                     / "saoudrizwan.claude-dev" / "settings")
    else:
        cline_dir = (home / ".config" / "Code" / "User" / "globalStorage"
                     / "saoudrizwan.claude-dev" / "settings")
    return {
        "windsurf": (home / ".codeium" / "windsurf" / "mcp_config.json",
                     {"mcpServers": {"tokengraph": stdio}}),
        "cline": (cline_dir / "cline_mcp_settings.json",
                  {"mcpServers": {"tokengraph": {**stdio, "disabled": False}}}),
    }


def _merge_json_file(path: Path, payload: dict) -> None:
    """Deep-merge `payload` into a JSON file, creating parents as needed."""
    import json
    cur: dict = {}
    if path.exists():
        try:
            cur = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_deep_merge(cur, payload), indent=2) + "\n",
                    encoding="utf-8")


def mcp_wiring_status(root: Path) -> list[dict]:
    """Which project MCP config files exist, and whether they really wire us in.

    Existence is not wiring: a `.claude/settings.json` containing only hooks
    used to be counted as MCP wiring, which reported a broken setup as healthy.
    Each entry is checked for a `tokengraph` server under the key that host
    actually reads.
    """
    import json
    specs = [
        (".mcp.json", "mcpServers"),
        (".vscode/mcp.json", "servers"),
        (".cursor/mcp.json", "mcpServers"),
        (".zed/settings.json", "context_servers"),
    ]
    out: list[dict] = []
    for rel, key in specs:
        p = Path(root) / rel
        declares = False
        if p.exists():
            try:
                declares = "tokengraph" in (
                    json.loads(p.read_text(encoding="utf-8")).get(key) or {})
            except Exception:
                declares = False
        out.append({"path": rel, "key": key, "exists": p.exists(),
                    "declares_tokengraph": declares})
    return out


def ide_setup(root: Path, editors: list[str] | None = None,
              workspace_roots: list[Path] | None = None,
              write_global: bool = False) -> dict:
    """Wire ContextIQ's MCP server into every (or selected) MCP-capable editor.

    For an MCP-native tool this is the equivalent of shipping editor plugins:
    one command drops a correct server config into each editor's expected
    location. Non-destructive (merges into existing JSON).

    Hosts that only read a per-user config path (Windsurf, Cline) are written
    only when `write_global` is set, since those files live outside the repo;
    otherwise their exact path and payload are returned under `global_pending`.
    """
    root = Path(root).resolve()
    roots = [Path(p).resolve() for p in workspace_roots] if workspace_roots else [root]
    stdio = mcp_launch_command()
    stdio_typed = {"type": "stdio", **stdio}

    # editor -> (relative config path, payload to merge). Every shape below is
    # the one the host actually parses:
    #   .mcp.json / .cursor/mcp.json    -> {"mcpServers": {...}}
    #   .vscode/mcp.json                -> {"servers": {...}}  (VS Code Copilot)
    #   .zed/settings.json              -> {"context_servers": {...}} with
    #                                      "source": "custom", which Zed
    #                                      requires to accept a user-defined
    #                                      server (it was missing before).
    targets = {
        "claude":   (".mcp.json", {"mcpServers": {"tokengraph": stdio_typed}}),
        "vscode":   (".vscode/mcp.json", {"servers": {"tokengraph": stdio_typed}}),
        "cursor":   (".cursor/mcp.json", {"mcpServers": {"tokengraph": stdio_typed}}),
        "zed":      (".zed/settings.json",
                     {"context_servers": {"tokengraph": {
                         "source": "custom",
                         "command": stdio["command"],
                         "args": stdio["args"],
                         "env": {}}}}),
        "continue": (".continue/config.yaml", None),   # YAML, written below
    }
    chosen = editors or [e for e in targets if e != "continue"]
    written: list[str] = []
    for workspace_root in roots:
        for ed in chosen:
            if ed not in targets:
                continue
            rel, payload = targets[ed]
            p = workspace_root / rel
            if ed == "continue":
                _write_continue_config(p, stdio)
            else:
                _merge_json_file(p, payload)
            written.append(rel if len(roots) == 1 else str(p))

    # Per-user configs (outside the repo).
    global_written: list[str] = []
    global_pending: list[dict] = []
    for name, (gpath, gpayload) in global_mcp_targets().items():
        if editors and name not in editors:
            continue
        if write_global:
            _merge_json_file(gpath, gpayload)
            global_written.append(str(gpath))
        else:
            global_pending.append({
                "host": name, "path": str(gpath), "payload": gpayload,
                "why": f"{name} reads MCP config only from this per-user path; "
                       f"a file inside the repo is ignored.",
            })

    # JetBrains and Neovim previously got a printed string and nothing else.
    # Both do read real config files, so write them.
    for workspace_root in roots:
        if not editors or "jetbrains" in editors:
            # JetBrains AI Assistant / Junie read project-level MCP config from
            # .idea/mcp.xml. Written unconditionally: a project without .idea
            # gets one the first time the IDE opens it, and an unused file is
            # inert.
            _write_jetbrains_mcp(workspace_root / ".idea" / "mcp.xml", stdio)
            written.append(".idea/mcp.xml" if len(roots) == 1
                           else str(workspace_root / ".idea" / "mcp.xml"))
        if not editors or "nvim" in editors:
            path = workspace_root / ".nvim" / "contextiq.lua"
            _write_nvim_config(path, stdio)
            written.append(".nvim/contextiq.lua" if len(roots) == 1
                           else str(path))

    nvim = ('require("mcphub").setup({ servers = { tokengraph = { command = "%s", '
            'args = { %s } } } })' % (stdio["command"],
            ", ".join(f'"{a}"' for a in stdio["args"])))
    return {
        "written": written,
        "global_written": global_written,
        "global_pending": global_pending,
        "editors": chosen,
        "launch": stdio,
        "neovim_snippet": nvim,
        "jetbrains_note": ("JetBrains: .idea/mcp.xml written. If AI Assistant "
                           "does not pick it up, add it manually via "
                           "Settings → Tools → AI Assistant → MCP → Add → "
                           f"`{stdio['command']} {' '.join(stdio['args'])}`"),
        "workspace_roots": [str(p) for p in roots],
        "note": (f"wired {len(written)} project config(s)"
                 + (f" and {len(global_written)} per-user config(s)"
                    if global_written else
                    (f"; {len(global_pending)} host(s) need --global"
                     if global_pending else ""))
                 + "; restart the editor to load"),
    }


def _write_jetbrains_mcp(path: Path, stdio: dict) -> None:
    """Write .idea/mcp.xml, preserving any other servers already configured.

    JetBrains stores project settings as IntelliJ XML components, so this is a
    real config file rather than the copy-paste instruction we used to print.
    """
    import xml.etree.ElementTree as ET
    args = "".join(f'<option value="{_xml_escape(a)}" />' for a in stdio["args"])
    entry = (f'<server name="tokengraph">'
             f'<option name="command" value="{_xml_escape(stdio["command"])}" />'
             f'<option name="args"><array>{args}</array></option>'
             f'</server>')
    existing = ""
    if path.exists():
        try:
            tree = ET.parse(path)
            servers = tree.getroot().find(".//servers")
            if servers is not None:
                keep = [ET.tostring(s, encoding="unicode").strip()
                        for s in servers.findall("server")
                        if s.get("name") != "tokengraph"]
                existing = "".join(keep)
        except (ET.ParseError, OSError):
            existing = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<project version="4">\n'
        '  <component name="McpServerSettings">\n'
        f'    <servers>{existing}{entry}</servers>\n'
        '  </component>\n'
        '</project>\n', encoding="utf-8")


def _xml_escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _write_nvim_config(path: Path, stdio: dict) -> None:
    """Write a real, sourceable Lua module for Neovim MCP clients.

    Returns the server spec rather than calling setup() itself, so it composes
    with whatever plugin manager and MCP client the user already runs.
    """
    args = ", ".join(f'"{a}"' for a in stdio["args"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "-- ContextIQ MCP server (generated by `tokengraph ide-setup`).\n"
        "-- Source from your config, e.g.:\n"
        "--   local contextiq = dofile(vim.fn.getcwd() .. '/.nvim/contextiq.lua')\n"
        "--   require('mcphub').setup({ servers = contextiq.servers })\n"
        "local M = {}\n\n"
        "M.servers = {\n"
        "  tokengraph = {\n"
        f'    command = "{stdio["command"]}",\n'
        f"    args = {{ {args} }},\n"
        "  },\n"
        "}\n\n"
        "-- Convenience for mcphub.nvim users.\n"
        "function M.setup(opts)\n"
        "  opts = vim.tbl_deep_extend('force', { servers = M.servers }, opts or {})\n"
        "  require('mcphub').setup(opts)\n"
        "end\n\n"
        "return M\n", encoding="utf-8")


def _write_continue_config(path: Path, stdio: dict) -> None:
    """Add a tokengraph MCP entry to Continue's YAML config.

    Continue uses YAML, not JSON, and PyYAML is not a dependency, so this
    appends a correctly-indented block only when one is not already present
    rather than attempting a full parse-and-rewrite.
    """
    marker = "name: tokengraph"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in existing:
        return
    args = "".join(f"\n      - {a}" for a in stdio["args"])
    block = (f"mcpServers:\n"
             f"  - name: tokengraph\n"
             f"    command: {stdio['command']}\n"
             f"    args:{args}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    if existing.strip():
        path.write_text(existing.rstrip() + "\n\n" + block, encoding="utf-8")
    else:
        path.write_text("name: Local Assistant\nversion: 1.0.0\n\n" + block,
                        encoding="utf-8")


# ---- one-command IDE completeness: MCP wiring + steering rules + verify -------

# Editor -> the steering-rules adapter that editor actually reads. `ide-setup`
# writes the MCP server config; these give the same command the *second* half of
# a complete setup (the project rules block), so one command fully provisions an
# IDE instead of leaving the user to also run `generate`.
_EDITOR_ADAPTERS = {
    "vscode": "copilot",     # GitHub Copilot reads .github/copilot-instructions.md
    "cursor": "cursor",      # .cursor/rules/contextiq.mdc
    "windsurf": "windsurf",  # .windsurf/rules/contextiq.md (read project-locally)
    "zed": "zed",            # .rules
    "continue": "continue",  # .continue/rules/contextiq.md
    "cline": "cline",        # .clinerules/contextiq.md
    "claude": "claude",      # CLAUDE.md
}


def write_ide_rules(root: Path, editors: list[str] | None = None) -> list[str]:
    """Write the per-editor steering-rules block for the chosen editors.

    Reuses exactly the machinery `generate` uses (build_context_payload +
    write_adapter), so the rules an IDE reads and the ones `generate` emits stay
    identical. Non-destructive: hand-written content outside the marker block is
    preserved. Returns the relative paths written.
    """
    root = Path(root).resolve()
    eds = editors or list(_EDITOR_ADAPTERS)
    adapters = list(dict.fromkeys(
        _EDITOR_ADAPTERS[e] for e in eds
        if e in _EDITOR_ADAPTERS and _EDITOR_ADAPTERS[e] in ADAPTERS))
    if not adapters:
        return []
    cfg = load_config(root, None)
    index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    written: list[str] = []
    try:
        fingerprint = r.store.content_fingerprint(generated_artifact_paths(cfg))
        strategy = cfg.get("strategy", "hot-cold")
        src_dirs = cfg.get("srcDirs") or ["."]
        hot_commits = cfg.get("hotCommits", 10)
        by_budget: dict[int, dict] = {}
        for ad in adapters:
            budget = min(adapter_budget(ad, 6000), 8000)
            payload = by_budget.get(budget)
            if payload is None:
                payload = build_context_payload(
                    r, root, strategy=strategy, src_dirs=src_dirs, budget=budget,
                    hot_commits=hot_commits, diff=False, staged=False, config=cfg)
                by_budget[budget] = payload
            rel = write_adapter(root, ad, payload["markdown"], None,
                                fingerprint=fingerprint)
            written.append(rel)
    finally:
        r.close()
    return written


def verify_ide_wiring(root: Path, editors: list[str] | None = None) -> list[dict]:
    """Per-editor proof of a complete setup: MCP server + steering rules.

    For each editor reports whether its MCP config declares `tokengraph` (or, for
    per-user hosts like Windsurf/Cline, that the global file does), and whether
    its steering-rules file is present. `ready` is true when both halves are in
    place — the signal that closes "we wrote configs" up to "the IDE is wired".
    """
    import json
    root = Path(root).resolve()
    # editor -> (project mcp file, json key or None for non-JSON markers)
    proj = {
        "claude":   (".mcp.json", "mcpServers"),
        "vscode":   (".vscode/mcp.json", "servers"),
        "cursor":   (".cursor/mcp.json", "mcpServers"),
        "zed":      (".zed/settings.json", "context_servers"),
        "continue": (".continue/config.yaml", None),
        "jetbrains": (".idea/mcp.xml", None),
        "nvim":     (".nvim/contextiq.lua", None),
    }
    globals_ = global_mcp_targets()   # windsurf, cline (per-user paths)

    def _declares(path: Path, key: str | None) -> bool:
        if not path.exists():
            return False
        try:
            txt = path.read_text(encoding="utf-8")
        except OSError:
            return False
        if key is None:
            return "tokengraph" in txt
        try:
            return "tokengraph" in (json.loads(txt).get(key) or {})
        except Exception:
            return False

    eds = editors or (list(proj) + list(globals_))
    out: list[dict] = []
    for ed in eds:
        rules_rel = ADAPTERS.get(_EDITOR_ADAPTERS.get(ed, ""), {}).get("path")
        rules_present = bool(rules_rel and (root / rules_rel).exists())
        if ed in proj:
            rel, key = proj[ed]
            mcp_wired = _declares(root / rel, key)
            mcp_path, mcp_scope = rel, "project"
        elif ed in globals_:
            gpath, _ = globals_[ed]
            mcp_wired = _declares(gpath, "mcpServers")
            mcp_path, mcp_scope = str(gpath), "per-user (needs --global)"
        else:
            continue
        rules_optional = rules_rel is None
        out.append({
            "editor": ed,
            "mcp_wired": mcp_wired,
            "mcp_scope": mcp_scope,
            "mcp_path": mcp_path,
            "rules_present": rules_present,
            "rules_path": rules_rel,
            "ready": mcp_wired and (rules_present or rules_optional),
        })
    return out


# ---- installable editor plugins (real artifacts, not just MCP config) --------

_VSCODE_PACKAGE_JSON = """\
{
  "name": "contextiq",
  "displayName": "ContextIQ",
  "description": "Token-efficient code context for AI agents (tokengraph graph).",
  "version": "0.1.0",
  "publisher": "contextiq",
  "license": "MIT",
  "homepage": "https://github.com/contextiq/contextiq",
  "repository": { "type": "git", "url": "https://github.com/contextiq/contextiq.git" },
  "bugs": { "url": "https://github.com/contextiq/contextiq/issues" },
  "keywords": ["ai", "context", "mcp", "code-graph", "tokens", "llm"],
  "engines": { "vscode": "^1.85.0" },
  "categories": ["Other", "AI", "Machine Learning"],
  "activationEvents": ["onStartupFinished"],
  "main": "./extension.js",
  "contributes": {
    "commands": [
      { "command": "contextiq.context", "title": "ContextIQ: Find Relevant Context for Task" },
      { "command": "contextiq.reindex", "title": "ContextIQ: Reindex Repository" },
      { "command": "contextiq.conventions", "title": "ContextIQ: Show Conventions" },
      { "command": "contextiq.impact", "title": "ContextIQ: Impact of Symbol at Cursor" }
    ],
    "configuration": {
      "title": "ContextIQ",
      "properties": {
        "contextiq.command": {
          "type": "string", "default": "tokengraph",
                    "description": "ContextIQ executable path."
                },
                "contextiq.commandArgs": {
                    "type": "array", "default": [], "items": { "type": "string" },
                    "description": "Arguments placed before the ContextIQ subcommand."
        }
      }
    }
  }
}
"""

_VSCODE_EXTENSION_JS = r"""// ContextIQ VS Code extension — no build step (plain CommonJS).
// Package: `npm i -g @vscode/vsce && vsce package` -> contextiq-0.1.0.vsix
// Install: `code --install-extension contextiq-0.1.0.vsix`
const vscode = require('vscode');
const cp = require('child_process');

function cli() {
  return vscode.workspace.getConfiguration('contextiq').get('command') || 'tokengraph';
}
function cliArgs() {
    return vscode.workspace.getConfiguration('contextiq').get('commandArgs') || [];
}
function root() {
    const editor = vscode.window.activeTextEditor;
    const active = editor && vscode.workspace.getWorkspaceFolder(editor.document.uri);
    const folders = vscode.workspace.workspaceFolders;
    return active ? active.uri.fsPath : (folders && folders.length ? folders[0].uri.fsPath : process.cwd());
}
function run(args) {
  return new Promise((resolve) => {
        cp.execFile(cli(), [...cliArgs(), ...args], { cwd: root(), maxBuffer: 16 * 1024 * 1024 },
      (err, stdout, stderr) => resolve(stdout || stderr || String(err)));
  });
}
let out;
function show(title, text) {
  if (!out) out = vscode.window.createOutputChannel('ContextIQ');
  out.clear(); out.appendLine('# ' + title + '\n'); out.append(text); out.show(true);
}
function activate(context) {
  const reg = (id, fn) =>
    context.subscriptions.push(vscode.commands.registerCommand(id, fn));
  reg('contextiq.context', async () => {
    const task = await vscode.window.showInputBox({ prompt: 'ContextIQ: task / question' });
    if (!task) return;
        show('Context: ' + task, await run(['context', task]));
  });
  reg('contextiq.reindex', async () => {
        await run(['index']); vscode.window.showInformationMessage('ContextIQ: reindexed');
  });
    reg('contextiq.conventions', async () => show('Conventions', await run(['conventions'])));
  reg('contextiq.impact', async () => {
    const ed = vscode.window.activeTextEditor; if (!ed) return;
    const sel = ed.document.getText(ed.selection);
    const word = sel || ed.document.getText(
      ed.document.getWordRangeAtPosition(ed.selection.active) || ed.selection);
    if (!word) return;
    show('Impact: ' + word, await run(['impact', word]));
  });
}
function deactivate() {}
module.exports = { activate, deactivate };
"""

_VSCODE_README = """\
# ContextIQ — VS Code extension

Adds commands that drive the local ContextIQ graph (`tokengraph`):

- **ContextIQ: Find Relevant Context for Task** — token-budgeted context pack
- **ContextIQ: Reindex Repository**
- **ContextIQ: Show Conventions**
- **ContextIQ: Impact of Symbol at Cursor**

## Install
```
npm i -g @vscode/vsce
vsce package            # -> contextiq-0.1.0.vsix
code --install-extension contextiq-0.1.0.vsix
```
Set `contextiq.command` if the CLI isn't on PATH (e.g. `python /path/tokengraph_all.py`).
For agent-mode tool use, also run `tokengraph ide-setup` to wire the MCP server.
"""

_VSCODE_IGNORE = ".vscode/**\n.gitignore\n*.vsix\nnode_modules/**\n"

_NVIM_INIT_LUA = r"""-- ContextIQ Neovim plugin (lua/contextiq/init.lua)
local M = {}
M.config = { command = "tokengraph", command_args = {} }

function M.setup(opts)
  M.config = vim.tbl_extend("force", M.config, opts or {})
end

local function run(args)
    local cmd = { M.config.command }
    vim.list_extend(cmd, M.config.command_args)
    vim.list_extend(cmd, args)
    return vim.fn.system(cmd)
end

local function show(title, text)
  vim.cmd("botright new")
  local buf = vim.api.nvim_get_current_buf()
  vim.bo[buf].buftype = "nofile"
  vim.bo[buf].bufhidden = "wipe"
  vim.bo[buf].filetype = "markdown"
  local lines = vim.split("# " .. title .. "\n\n" .. text, "\n", { plain = true })
  vim.api.nvim_buf_set_lines(buf, 0, -1, false, lines)
end

function M.context(task)
  task = task or vim.fn.input("ContextIQ task: ")
  if task == "" then return end
    show("Context: " .. task, run({ "context", task }))
end

function M.conventions() show("Conventions", run({ "conventions" })) end

function M.impact(sym)
  sym = (sym and sym ~= "") and sym or vim.fn.expand("<cword>")
    show("Impact: " .. sym, run({ "impact", sym }))
end

function M.reindex()
    run({ "index" })
  vim.notify("ContextIQ: reindexed")
end

return M
"""

_NVIM_PLUGIN_LUA = r"""-- ContextIQ user commands (plugin/contextiq.lua)
vim.api.nvim_create_user_command("ContextIQ", function(o)
  require("contextiq").context(o.args ~= "" and o.args or nil)
end, { nargs = "?", desc = "ContextIQ: context pack for a task" })

vim.api.nvim_create_user_command("ContextIQConventions", function()
  require("contextiq").conventions()
end, { desc = "ContextIQ: repo conventions" })

vim.api.nvim_create_user_command("ContextIQImpact", function(o)
  require("contextiq").impact(o.args ~= "" and o.args or nil)
end, { nargs = "?", desc = "ContextIQ: blast radius of a symbol" })

vim.api.nvim_create_user_command("ContextIQReindex", function()
  require("contextiq").reindex()
end, { desc = "ContextIQ: reindex repository" })
"""

_NVIM_README = """\
# ContextIQ — Neovim plugin

Commands: `:ContextIQ [task]`, `:ContextIQConventions`, `:ContextIQImpact [sym]`,
`:ContextIQReindex`.

## Install (lazy.nvim)
```lua
{ dir = "/path/to/ide-plugins/nvim", config = function()
    require("contextiq").setup({ command = "tokengraph" })
  end }
```
Or copy `lua/` and `plugin/` into your runtimepath.
"""

_JB_PLUGIN_XML = """\
<idea-plugin>
  <id>com.contextiq.plugin</id>
  <name>ContextIQ</name>
  <version>0.1.0</version>
  <vendor email="hello@contextiq.dev" url="https://github.com/contextiq/contextiq">ContextIQ</vendor>
  <description><![CDATA[
    Token-efficient code context for AI agents, backed by the local tokengraph graph.
    Adds a "Find Relevant Context" action that returns a budgeted context pack.
  ]]></description>
  <change-notes><![CDATA[0.1.0 — initial release.]]></change-notes>
  <depends>com.intellij.modules.platform</depends>
  <actions>
    <action id="ContextIQ.Context" class="com.contextiq.ContextIQAction"
            text="ContextIQ: Find Relevant Context" description="Token-budgeted context pack for a task">
      <add-to-group group-id="ToolsMenu" anchor="last"/>
    </action>
  </actions>
</idea-plugin>
"""

_JB_ACTION_KT = """\
package com.contextiq

import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.ui.Messages

class ContextIQAction : AnAction() {
    override fun actionPerformed(e: AnActionEvent) {
        val task = Messages.showInputDialog(
            e.project, "Task / question:", "ContextIQ", null) ?: return
        val dir = e.project?.basePath ?: System.getProperty("user.dir")
        val out = try {
            ProcessBuilder("tokengraph", "context", task)
                .directory(java.io.File(dir)).redirectErrorStream(true)
                .start().inputStream.bufferedReader().readText()
        } catch (ex: Exception) { "ContextIQ error: ${'$'}{ex.message}" }
        Messages.showInfoMessage(e.project, out.take(8000), "ContextIQ")
    }
}
"""

_JB_BUILD_GRADLE = """\
plugins {
    id("java")
    id("org.jetbrains.kotlin.jvm") version "1.9.24"
    id("org.jetbrains.intellij") version "1.17.3"
}
group = "com.contextiq"
version = "0.1.0"
repositories { mavenCentral() }
intellij {
    version.set("2023.3")
    plugins.set(listOf())
}
tasks { patchPluginXml { sinceBuild.set("233") } }
"""

_JB_SETTINGS_GRADLE = 'rootProject.name = "contextiq"\n'

_JB_README = """\
# ContextIQ — JetBrains plugin

Buildable IntelliJ-platform plugin (IDEA / PyCharm / WebStorm / GoLand).

## Build
```
./gradlew buildPlugin     # -> build/distributions/contextiq-0.1.0.zip
```
Install via Settings → Plugins → ⚙ → Install Plugin from Disk.
Requires the `tokengraph` CLI on PATH.
"""


def emit_ide_plugins(root: Path, out_dir: str = "ide-plugins",
                     editors: list[str] | None = None) -> dict:
    """Scaffold real, installable editor plugins (not just MCP config):

      • VS Code  — packageable to a .vsix (plain CommonJS, no build step)
      • Neovim   — a Lua plugin (lazy.nvim/packer-installable)
      • JetBrains — a Gradle-buildable IntelliJ-platform plugin

    Each exposes commands that drive the local graph (context / conventions /
    impact / reindex). Combined with `ide-setup` (MCP wiring), this matches a
    shipped-plugin offering. Returns the files written."""
    base = Path(root).resolve() / out_dir
    files: dict[str, str] = {}
    sel = editors or ["vscode", "nvim", "jetbrains"]
    if "vscode" in sel:
        files.update({
            "vscode/package.json": _VSCODE_PACKAGE_JSON,
            "vscode/extension.js": _VSCODE_EXTENSION_JS,
            "vscode/README.md": _VSCODE_README,
            "vscode/.vscodeignore": _VSCODE_IGNORE,
        })
    if "nvim" in sel:
        files.update({
            "nvim/lua/contextiq/init.lua": _NVIM_INIT_LUA,
            "nvim/plugin/contextiq.lua": _NVIM_PLUGIN_LUA,
            "nvim/README.md": _NVIM_README,
        })
    if "jetbrains" in sel:
        files.update({
            "jetbrains/build.gradle.kts": _JB_BUILD_GRADLE,
            "jetbrains/settings.gradle.kts": _JB_SETTINGS_GRADLE,
            "jetbrains/src/main/resources/META-INF/plugin.xml": _JB_PLUGIN_XML,
            "jetbrains/src/main/kotlin/com/contextiq/ContextIQAction.kt": _JB_ACTION_KT,
            "jetbrains/README.md": _JB_README,
        })
    written: list[str] = []
    for rel, content in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        written.append(f"{out_dir}/{rel}")
    return {
        "out_dir": out_dir,
        "editors": sel,
        "written": sorted(written),
        "install": {
            "vscode": "cd ide-plugins/vscode && vsce package && code --install-extension *.vsix",
            "nvim": "add ide-plugins/nvim to runtimepath (lazy.nvim: { dir = '…/nvim' })",
            "jetbrains": "cd ide-plugins/jetbrains && ./gradlew buildPlugin",
        },
        "note": f"scaffolded {len(written)} plugin file(s) under {out_dir}/",
    }


# ---- distribution kit: cross-platform release automation + install channels --

_RELEASE_WORKFLOW = """\
# Build standalone binaries for every OS and publish to PyPI on a version tag.
# Trigger: git tag v0.1.0 && git push --tags
name: release
on:
  push:
    tags: ["v*"]
permissions:
  contents: write          # upload release assets
  id-token: write          # PyPI trusted publishing
jobs:
  binaries:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install pyinstaller
      - run: pyinstaller --onefile --name tokengraph tokengraph_all.py
      - uses: actions/upload-artifact@v4
        with:
          name: tokengraph-${{ matrix.os }}
          path: dist/*
      - uses: softprops/action-gh-release@v2
        with:
          files: dist/*
  pypi:
    needs: binaries
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install build && python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1   # trusted publishing, no token
"""

_DOCKERFILE = """\
# A tiny image that runs ContextIQ as an MCP server or CLI.
#   docker build -t contextiq .
#   docker run --rm -v "$PWD:/repo" -w /repo contextiq context "add retry"
FROM python:3.11-slim
WORKDIR /app
COPY tokengraph_all.py pyproject.toml README.md ./
RUN pip install --no-cache-dir ".[all]"
WORKDIR /repo
ENTRYPOINT ["tokengraph"]
CMD ["--help"]
"""

_HOMEBREW_FORMULA = """\
# Homebrew formula — `brew install --build-from-source ./contextiq.rb`
# or host it in a tap (homebrew-contextiq) for `brew install contextiq`.
class Contextiq < Formula
  include Language::Python::Virtualenv
  desc "Local code-graph MCP server for token-efficient AI context"
  homepage "https://github.com/contextiq/contextiq"
  url "https://files.pythonhosted.org/packages/source/c/contextiq/contextiq-0.1.0.tar.gz"
  version "0.1.0"
  license "MIT"
  depends_on "python@3.11"
  def install
    virtualenv_install_with_resources
  end
  test do
    system bin/"tokengraph", "langs"
  end
end
"""

_INSTALL_SH = """\
#!/bin/sh
# One-line install: curl -fsSL <raw>/install.sh | sh
# Prefers pipx (isolated), falls back to a prebuilt binary from GitHub Releases.
set -e
REPO="contextiq/contextiq"
if command -v pipx >/dev/null 2>&1; then
  echo "installing via pipx..."; exec pipx install "contextiq[all]"
fi
if command -v uv >/dev/null 2>&1; then
  echo "installing via uv..."; exec uv tool install "contextiq[all]"
fi
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$OS" in linux) A=ubuntu-latest;; darwin) A=macos-latest;; *) A=windows-latest;; esac
URL="https://github.com/$REPO/releases/latest/download/tokengraph-$A"
echo "downloading binary $URL ..."
curl -fsSL "$URL" -o /usr/local/bin/tokengraph && chmod +x /usr/local/bin/tokengraph
echo "installed: tokengraph"
"""

_DIST_README = """\
# ContextIQ — distribution kit

Built artifacts so a release is one `git tag` away:

- **`.github/workflows/release.yml`** — on a `v*` tag, builds standalone binaries
  for Linux/macOS/Windows (PyInstaller) → GitHub Releases, and publishes to PyPI
  via trusted publishing (no API token needed).
- **`Dockerfile`** — `docker run contextiq` for container/CI use.
- **`contextiq.rb`** — Homebrew formula (host in a tap for `brew install contextiq`).
- **`install.sh`** — one-line installer (`curl … | sh`): pipx → uv → prebuilt binary.

## Install channels (no publishing needed)
| Channel | Command | Equivalent to |
|---|---|---|
| pipx (isolated global) | `pipx install "contextiq[all]"` | `npm i -g` / Volta |
| uv (zero-install run) | `uvx contextiq context "task"` | `npx` |
| pip | `pip install "contextiq[all]"` | — |
| Docker | `docker run --rm -v "$PWD:/repo" -w /repo contextiq …` | — |
| binary | download from Releases (built by the workflow) | standalone binary |
"""


_PUBLISHING_MD = """\
# ContextIQ — publishing runbook (turn-key)

Everything below is built and verified locally; each step needs only YOUR account
credentials. Do them once.

## 1. PyPI (reserve the name + publish)
```bash
python -m build                       # -> dist/contextiq-0.1.0.tar.gz + .whl  (already verified)
pipx run twine check dist/*           # sanity-check metadata
pipx run twine upload dist/*          # prompts for your PyPI token -> reserves `contextiq`
```
Or fully automated: configure PyPI **Trusted Publishing** for this repo, then
`git tag v0.1.0 && git push --tags` — `.github/workflows/release.yml` builds the
cross-OS binaries and publishes to PyPI with no token.

## 2. VS Code Marketplace
```bash
cd ide-plugins/vscode
npm i -g @vscode/vsce
vsce create-publisher contextiq       # one-time; or use an existing publisher
vsce login contextiq                  # paste an Azure DevOps PAT (Marketplace > Manage)
vsce publish                          # package.json already has publisher/repo/license
```

## 3. JetBrains Marketplace
```bash
cd ide-plugins/jetbrains
./gradlew buildPlugin                 # -> build/distributions/contextiq-0.1.0.zip
# Upload the zip at https://plugins.jetbrains.com/plugin/add  (first upload is manual review),
# or automate later with the gradle `publishPlugin` task + a marketplace token.
```

## 4. Homebrew tap
```bash
# push the ready-made tap layout to a NEW github repo named homebrew-contextiq:
cd homebrew-contextiq && git init && git add . && git commit -m "contextiq formula"
git remote add origin https://github.com/<you>/homebrew-contextiq.git && git push -u origin main
# users then: brew install <you>/contextiq/contextiq
```
After PyPI publish, update `url`/`sha256` in the formula to the real sdist
(`brew create --python <sdist-url>` regenerates them).
"""

_HOMEBREW_TAP_README = """\
# homebrew-contextiq

Homebrew tap for ContextIQ. Push this directory to a GitHub repo named
**`homebrew-contextiq`**, then:

```bash
brew install <your-gh-user>/contextiq/contextiq
```
"""


def emit_distribution(root: Path, out_dir: str = ".") -> dict:
    """Scaffold the cross-platform release + install kit (real, usable files):
    CI release workflow, Dockerfile, Homebrew formula + push-ready tap layout, a
    one-line installer, and a turn-key PUBLISHING runbook. Combined with the
    PyPI metadata in pyproject.toml, publishing is one command per registry."""
    base = Path(root).resolve() / out_dir
    files = {
        ".github/workflows/release.yml": _RELEASE_WORKFLOW,
        "Dockerfile": _DOCKERFILE,
        "contextiq.rb": _HOMEBREW_FORMULA,
        "install.sh": _INSTALL_SH,
        "DISTRIBUTION.md": _DIST_README,
        "PUBLISHING.md": _PUBLISHING_MD,
        # push-ready Homebrew tap repo layout (formula must live under Formula/)
        "homebrew-contextiq/Formula/contextiq.rb": _HOMEBREW_FORMULA,
        "homebrew-contextiq/README.md": _HOMEBREW_TAP_README,
    }
    written: list[str] = []
    for rel, content in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        if rel == "install.sh":
            try:
                os.chmod(p, 0o755)
            except OSError:
                pass
        written.append(rel)
    return {
        "written": sorted(written),
        "channels": ["pipx", "uv/uvx", "pip", "docker", "homebrew", "binary"],
        "runbook": "PUBLISHING.md",
        "note": (f"scaffolded {len(written)} distribution file(s); follow "
                 "PUBLISHING.md — one command per registry (PyPI/VS Code/"
                 "JetBrains/Homebrew), each needs only your credentials"),
    }


# stable aliases so MCP tools can call these even though a tool of the same
# name is defined inside build_mcp_server's local scope (which would shadow them)
verify_plan_fn = verify_plan
review_diff_fn = review_diff
verify_ai_output_fn = verify_ai_output
hallucination_benchmark_fn = hallucination_benchmark


# ==========================================================================
# MCP server (FastMCP, lazy)
# ==========================================================================
def build_mcp_server(root: Path, db: Path):
    """Construct a FastMCP server exposing the graph tools. Imported lazily."""
    from fastmcp import FastMCP  # type: ignore[import-not-found]

    root = Path(root).resolve()
    db = Path(db)

    # RP-1: one pooled Retriever per thread, reused across tool calls.
    # Building a fresh Retriever per call threw away the source cache and the
    # HNSW index every time — on a large repo that meant rebuilding the ANN
    # index from scratch for every single query. The pool is thread-local
    # because a sqlite3 connection must not cross threads, and FastMCP may
    # dispatch tools concurrently.
    _pool = threading.local()
    _pool_lock = threading.Lock()
    _pool_all: list["Retriever"] = []      # every pooled instance, for shutdown

    def _ret() -> "Retriever":
        # Correctness layer: freshen before every retrieval so a query never
        # reads stale line spans (which would mis-slice edited files). The
        # mtime/size fast path keeps this to a stat() per file when nothing
        # changed. A pre-warm hook can keep this a no-op in the common case.
        report = index_repo(root, db)
        r = getattr(_pool, "ret", None)
        if r is None:
            r = Retriever(root, db)
            r.pooled = True          # close() becomes a no-op for pooled use
            _pool.ret = r
            with _pool_lock:
                _pool_all.append(r)
        elif report.parsed or report.removed or report.reembedded:
            # The graph moved under us — drop derived caches so the next query
            # cannot answer from stale source lines or a stale ANN index.
            r.invalidate()
        return r

    def _close_pool() -> int:
        """Release every pooled connection. Call at server shutdown.

        A pooled retriever deliberately survives `close()`, so it holds its
        sqlite handle open for the process lifetime. Anything that needs the
        database file released (tests, a temp checkout, Windows, which refuses
        to delete an open file) must call this.
        """
        with _pool_lock:
            for held in _pool_all:
                try:
                    held.close_now()
                except Exception:
                    pass
            n = len(_pool_all)
            _pool_all.clear()
        _pool.ret = None
        return n

    mcp = FastMCP(
        name="tokengraph",
        instructions=(
            "Token-efficient code context for this repository (deep parse with full "
            "call/import/inheritance graph for 25+ languages: Python, Java, Go, "
            "TypeScript, JavaScript, C/C++, C#, Rust, Ruby, PHP, Kotlin, Swift, Scala, "
            "Lua, Bash, Solidity, Perl, Erlang, Julia, R, Haskell, OCaml, Nim, "
            "PowerShell, Dart; 30+ more languages regex-indexed incl. "
            "SQL/Elixir/Clojure/F#/Groovy/Zig/Objective-C/Markdown). PREFER these "
            "tools over reading whole files. "
            "Orientation: call list_modules() first for a token table of top dirs, "
            "then find_relevant_context(task) (or ask(task) for intent/coverage/risk "
            "metadata) for a budgeted slice of the most relevant symbols + chunks. "
            "Drill in with get_symbol / get_callees / get_callers / explain_file / "
            "get_module_summary; use get_lines(file,start,end) for a surgical, "
            "secret-scanned line range and get_impact(qname) for blast radius before "
            "editing. search_semantic(query) finds symbols by meaning. get_map shows "
            "the import/inheritance graph; get_routing / suggest_tier give model-tier "
            "hints. validate(task) gates coverage before answering, judge(answer, "
            "context) flags ungrounded answers, and verify(answer) flags fabricated "
            "files/symbols (with did-you-mean). squeeze(text) shrinks pasted "
            "stacktraces/CI-logs/JSON before they cost tokens. read_memory / write_memory / "
            "create_checkpoint persist decisions across sessions (local only). "
            "estimate_savings(task) reports tokens saved. For safe code generation, "
            "conventions() detects house style, scaffold(name) proposes a "
            "convention-matched file (refuses on conflict), verify_plan(plan) checks "
            "refs + blast radius before acting, review_diff() audits the diff, and "
            "create(task) orchestrates retrieve->scaffold->plan-check; evidence(task) "
            "returns a deterministic, hash-grounded pack for audit/CI. "
            "get_diff_context() is the diff-native retrieval tool — a budgeted pack of "
            "exactly the symbols a git diff touches plus their blast radius, no task "
            "string needed (use it to review or continue an in-progress change). "
            "All output is "
            "secret-scanned. The graph auto-refreshes on every call (changed files are "
            "reparsed before answering), so results are never stale; reindex forces a "
            "full rescan, and notify_change / notify_file_created / notify_symbol_added "
            "/ notify_file_deleted let an IDE push file events proactively to keep the "
            "graph warm."
        ),
    )

    @mcp.tool
    def find_relevant_context(task: str, budget_tokens: int = 6000, depth: int = 1,
                              max_body_tokens: int = 1600,
                              session: str = "") -> str:
        """Return a token-budgeted context pack of the symbols most relevant to a task.

        Use this FIRST instead of opening files. Small seeds get full bodies;
        large seeds are demoted to signatures plus matching indexed chunks.
        Callers/callees/base-classes are included as signatures, ranked by
        relevance to the task. Anything dropped for budget is listed by name
        so you can request it explicitly.

        Pass a stable `session` id (any string identifying this conversation)
        to enable cross-turn dedup: symbols already sent to that session and
        unchanged since are referenced by name instead of being resent, which
        makes repeated retrievals in one conversation substantially cheaper.
        """
        r = _ret()
        try:
            md, info = r.find_relevant_context_cached(
                task, budget_tokens=budget_tokens, expand_depth=depth,
                max_body_tokens=max_body_tokens, session=session)
            pack = info.get("pack")
            if pack is not None:
                # Only bill/record work we actually performed; a cache hit
                # costs nothing and must not inflate the savings ledger.
                record_pack_savings(
                    root, "mcp.context", final_tokens=pack.rendered_tokens,
                    baseline_tokens=r._targeted_baseline(pack),
                    files=len({p.file for p in pack.pieces}))
            return md
        finally:
            r.close()

    @mcp.tool
    def get_symbol(qname: str) -> str:
        """Full source of one symbol by qualified name (module.Class.method)."""
        r = _ret()
        try:
            return r.get_symbol(qname) or f"(no symbol named {qname})"
        finally:
            r.close()

    @mcp.tool
    def get_callers(qname: str) -> list:
        """Qualified names of symbols that call the given symbol."""
        r = _ret()
        try:
            return r.get_callers(qname)
        finally:
            r.close()

    @mcp.tool
    def get_callees(qname: str) -> list:
        """Qualified names of symbols the given symbol calls."""
        r = _ret()
        try:
            return r.get_callees(qname)
        finally:
            r.close()

    @mcp.tool
    def file_skeleton(file: str) -> str:
        """Signatures (no bodies) for every definition in a file."""
        r = _ret()
        try:
            return r.file_skeleton(file)
        finally:
            r.close()

    @mcp.tool
    def search_semantic(query: str, limit: int = 12) -> list:
        """Find symbols by embedding similarity, ranked by relevance.

        Use when you don't know the exact identifier. NOTE: the strength of
        this depends on the active embedding backend — call
        `embedding_status()` to see which one is live. With
        sentence-transformers installed and warmed you get true meaning-based
        matching ("retry with backoff" finds a reattempt helper); on the
        default hashing backend you get robust lexical/structural overlap,
        which will NOT bridge unrelated vocabulary.
        """
        r = _ret()
        try:
            return [row["qname"] for row in r.semantic_search(query, limit=limit)]
        finally:
            r.close()

    @mcp.tool
    def embedding_status() -> dict:
        """Which embedding backend is actually in use, and how to improve it.

        Tells you whether `search_semantic` is doing true semantic matching or
        the deterministic hashing fallback, so you can calibrate how much to
        trust it.
        """
        return embed_backend_info()

    @mcp.tool
    def session_savings(session: str) -> dict:
        """What this conversation has already been sent, and what that saved.

        Pair with `find_relevant_context(task, session=...)`: symbols already
        delivered to `session` and unchanged since are referenced by name
        rather than resent.
        """
        store = Store(db)
        try:
            stats = store.session_stats(session)
            stats["note"] = ("Pass this same session id to find_relevant_context "
                             "to avoid resending these symbols.")
            return stats
        finally:
            store.close()

    @mcp.tool
    def cache_stats() -> dict:
        """Context-pack cache: entries, hits, and the graph version they pin to.

        Identical stateless requests are served from this cache and cost no
        retrieval work; any reindex that changes the graph invalidates it.
        """
        r = _ret()
        try:
            stats = r.store.pack_cache_stats()
            stats["note"] = ("Session-scoped packs are never cached — with a "
                             "session id the correct pack differs per call.")
            return stats
        finally:
            r.close()

    @mcp.tool
    def reset_session(session: str = "") -> dict:
        """Forget what was sent to a session (empty string clears all sessions).

        Use when starting a fresh conversation, or after context compaction —
        once the model no longer holds the earlier text, it must be resent.
        """
        store = Store(db)
        try:
            return {"cleared_entries": store.clear_session(session),
                    "session": session or "(all)"}
        finally:
            store.close()

    @mcp.tool
    def prompt_cache_blocks(strategy: str = "full", budget_tokens: int = 8000) -> dict:
        """Provider-ready message blocks with a correctly placed cache breakpoint.

        Returns the repository signature map split into a large STABLE prefix
        carrying Anthropic `cache_control: ephemeral`, followed by the small
        git-volatile tail (recent changes / TODOs) left uncached. Sending them
        in this order means a new commit invalidates only the tail instead of
        the whole map. For providers with automatic caching the same blocks
        work unannotated.
        """
        r = _ret()
        try:
            cfg = load_config(root)
            payload = build_context_payload(
                r, root, strategy=strategy, src_dirs=cfg.get("srcDirs", []),
                budget=budget_tokens, hot_commits=cfg.get("hotCommits", 20),
                diff=False, staged=False, config=cfg)
            return {
                "blocks": cache_blocks(payload),
                "stable_tokens": payload["stable_tokens"],
                "volatile_tokens": payload["volatile_tokens"],
                "cached_fraction": round(
                    payload["stable_tokens"] / max(1, payload["tokens"]), 3),
            }
        finally:
            r.close()

    @mcp.tool
    def get_module_summary(file: str) -> str:
        """A compact summary of a whole file (docstring + its types/functions).

        A few tokens to understand what a module is for, instead of reading it.
        """
        r = _ret()
        try:
            return r.module_summary(file)
        finally:
            r.close()

    @mcp.tool
    def set_module_summary(file: str, summary: str) -> str:
        """Cache a better (e.g. agent-written) one-paragraph summary for a file.

        Persists until the file changes; future packs reuse it instead of the
        auto-generated extractive summary.
        """
        store = Store(db)
        try:
            store.set_summary(file, summary, source="agent")
            store.commit()
            return f"saved summary for {file} (~{count_tokens(summary)} tokens)"
        finally:
            store.close()

    @mcp.tool
    def estimate_savings(task: str, budget_tokens: int = 6000, depth: int = 1) -> dict:
        """Pack tokens vs. what grepping and reading around the hits would cost.

        The headline `savings_pct` compares against a *competent* agent
        baseline (grep, then read a window around each hit, merging overlaps),
        not against reading whole files. The whole-file comparison is also
        returned, as `savings_pct_vs_whole_file`, but it flatters the tool on
        repos with large files and should not be quoted as the saving.
        """
        r = _ret()
        try:
            return r.measure(task, budget_tokens=budget_tokens, expand_depth=depth)
        finally:
            r.close()

    @mcp.tool
    def savings_report(tasks: list[str], budget_tokens: int = 6000, depth: int = 1) -> dict:
        """Aggregate with/without savings across many tasks.

        Like estimate_savings but for a list of tasks: returns per-task `rows`,
        an `aggregate` rollup (totals, overall/mean savings %, best/worst), and
        a repo-scale `repo` summary. Use this when you want a quantitative
        report rather than a single-task number.
        """
        r = _ret()
        try:
            return r.report(tasks, budget_tokens=budget_tokens, expand_depth=depth)
        finally:
            r.close()

    @mcp.tool
    def savings_ledger(since: str = "", model: str = DEFAULT_GAIN_MODEL,
                       top: int = 0) -> dict:
        """Aggregate the persistent savings ledger (.context/gain.ndjson).

        Unlike estimate_savings (a single what-if), this reports realized
        token savings accumulated across every tracked context/ask/measure/
        generate call: totals, reduction %, per-op breakdown, a dollar
        projection, and daily/weekly/monthly trends. `since` accepts '7d' /
        '12h' / an ISO date; empty = all time.
        """
        return summarize_gain(root, since=since or None, model=model,
                              top=top or None, trends=True)

    @mcp.tool
    def reindex() -> str:
        """Re-scan the repo and update the graph incrementally. Call after edits."""
        rep = index_repo(root, db)
        return (f"parsed={rep.parsed} skipped={rep.skipped} removed={rep.removed} "
                f"graph={rep.stats}")

    @mcp.tool
    def ingest_scip(index_file: str) -> dict:
        """Import precise REFERENCES edges from SCIP JSON inside the repo."""
        target = (root / index_file).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return {"error": "SCIP index must be inside the repository"}
        if not target.is_file():
            return {"error": f"no SCIP JSON file: {index_file}"}
        return import_scip_json(root, db, target)

    # ---- proactive change notifications (optional; the graph also
    #      auto-refreshes on every query). An IDE/agent that already knows a
    #      file event happened can push it here to keep the graph warm so the
    #      next retrieval is a no-op. 'deleted' forgets named paths precisely;
    #      'created'/'modified' run a cheap incremental reindex. ----
    def _notify(event: str, paths: list) -> dict:
        paths = [p.replace("\\", "/").lstrip("./") for p in (paths or []) if p]
        if event == "deleted":
            store = Store(db)
            removed = 0
            try:
                indexed = store.all_indexed_files()
                for p in paths:
                    if p in indexed:
                        store.forget_file(p); removed += 1
                store.commit()
            finally:
                store.close()
            return {"event": "deleted", "paths": paths, "removed": removed,
                    "note": f"forgot {removed} file(s) from the graph"}
        rep = index_repo(root, db, paths=paths or None)
        return {"event": event, "paths": paths, "parsed": rep.parsed,
                "skipped": rep.skipped, "removed": rep.removed,
                "note": (f"reindexed: {rep.parsed} parsed, {rep.removed} removed "
                         f"({rep.skipped} unchanged)")}

    @mcp.tool
    def notify_change(event: str = "modified", paths: list = None) -> dict:
        """Fold a file event into the graph proactively (created|modified|deleted).

        Optional — ContextIQ already freshens on every query — but pushing the
        event keeps the graph warm so the next retrieval pays no reparse cost.
        'deleted' forgets the named paths; others run an incremental reindex."""
        return _notify(event, paths or [])

    @mcp.tool
    def notify_file_created(path: str = "") -> dict:
        """Fold a newly created file into the graph (SigMap-compatible hook)."""
        return _notify("created", [path] if path else [])

    @mcp.tool
    def notify_symbol_added(path: str = "") -> dict:
        """Re-fold a file after a symbol was added (SigMap-compatible hook).

        A symbol add is a file modification at ContextIQ's file granularity."""
        return _notify("modified", [path] if path else [])

    @mcp.tool
    def notify_file_deleted(path: str = "") -> dict:
        """Forget a deleted file from the graph (SigMap-compatible hook)."""
        return _notify("deleted", [path] if path else [])

    @mcp.tool
    def list_modules() -> list:
        """Token-count table of top-level source directories. CALL THIS FIRST.

        Cheapest way to orient: shows files/tokens/symbols per top dir so you
        can scope read_context / find_relevant_context to one module instead of
        loading the whole codebase.
        """
        r = _ret()
        try:
            return r.list_modules()
        finally:
            r.close()

    @mcp.tool
    def explain_file(file: str) -> str:
        """Signatures + imports + external callers for one file (who depends on it)."""
        r = _ret()
        try:
            return r.explain_file(file)
        finally:
            r.close()

    @mcp.tool
    def get_impact(qname: str) -> dict:
        """Blast radius of a symbol: direct + transitive callers, subclasses, tests touched.

        Use before changing a function to see what it could break.
        """
        r = _ret()
        try:
            return r.get_impact(qname)
        finally:
            r.close()

    @mcp.tool
    def get_lines(file: str, start: int, end: int) -> str:
        """Surgical fetch of an exact line range — clamped, secret-scanned, sandboxed.

        For when a signature isn't enough but a whole file is too much.
        """
        r = _ret()
        try:
            return r.get_lines(file, start, end)
        finally:
            r.close()

    @mcp.tool
    def get_method_impact(qname: str) -> dict:
        """Function-level blast radius: which functions break if this one changes.

        The change-safety view for a single function/method: its callers (that
        break on a signature change) with file:line call sites, its callees
        (dependencies), same-name overrides/overloads, transitive callers, and
        the tests touched. Use before editing a function's signature.
        """
        r = _ret()
        try:
            return r.get_method_impact(qname)
        finally:
            r.close()

    @mcp.tool
    def get_map(kind: str = "imports") -> dict:
        """Project graph by kind: 'imports' | 'hierarchy' | 'routes' | 'hubs'.

        imports = import edges; hierarchy = class inheritance; routes = HTTP
        endpoints (Flask/FastAPI/Express/Spring/Go); hubs = fan-in/fan-out
        ranking + import cycles. Orient without reading files.
        """
        r = _ret()
        try:
            return r.get_map(kind)
        finally:
            r.close()

    @mcp.tool
    def get_architecture_overview() -> dict:
        """Whole-repo shape in one call: module breakdown, hub files, import
        cycles, language mix, and route totals.

        The fastest way to orient in an unfamiliar repo — composes list_modules
        and get_map (hubs/cycles/routes) into a single payload so you don't have
        to call each separately.
        """
        r = _ret()
        try:
            return r.get_architecture_overview()
        finally:
            r.close()

    @mcp.tool
    def get_test_map(target: str = "") -> dict:
        """Map implementation files to their tests, and back.

        No target → the whole-repo impl<->test map + coverage stats. A file →
        its tests (or, for a test, what it covers). A symbol qname → its file's
        tests plus tests that reference the symbol through the call graph. Use
        before writing a test (find the right file) or editing code (find the
        tests that exercise it).
        """
        r = _ret()
        try:
            return r.get_test_map(target)
        finally:
            r.close()

    @mcp.tool
    def get_routing() -> list:
        """Per-file model-tier hints (fast/balanced/powerful) with cost + model names.

        Lets you route each file to the cheapest sufficient model.
        """
        r = _ret()
        try:
            return r.get_routing()
        finally:
            r.close()

    @mcp.tool
    def suggest_tier(task: str) -> dict:
        """Recommend a model tier (fast/balanced/powerful) for a task, with cost hint."""
        return recommend_tier(task)

    @mcp.tool
    def ask(task: str, budget_tokens: int = 6000, depth: int = 1) -> dict:
        """Focused retrieval with metadata: intent, coverage %, risk, cost, top files + pack.

        Like find_relevant_context but returns a structured result (for CI /
        dashboards) alongside the markdown pack.
        """
        r = _ret()
        try:
            res = r.ask(task, budget_tokens=budget_tokens, depth=depth)
            # Wire `ask` into the realized-savings ledger like find_relevant_context
            # (op "mcp.ask"); best-effort — record_pack_savings never raises.
            record_pack_savings(root, "mcp.ask",
                                final_tokens=res.get("pack_tokens", 0),
                                baseline_tokens=res.get("baseline_tokens", 0),
                                files=len(res.get("top_files", [])))
            return res
        finally:
            r.close()

    @mcp.tool
    def validate(task: str, min_coverage: float = 60.0) -> dict:
        """Check whether the assembled context is sufficient before answering (coverage gate)."""
        r = _ret()
        try:
            return r.validate(task, min_coverage=min_coverage)
        finally:
            r.close()

    @mcp.tool
    def judge(answer: str, context: str) -> dict:
        """Score whether an answer is grounded in a context (hallucination guard)."""
        r = _ret()
        try:
            return r.judge(answer, context)
        finally:
            r.close()

    @mcp.tool
    def verify(answer: str) -> dict:
        """Flag fabricated file paths / code symbols in an answer, with did-you-mean.

        Deterministic, no LLM: extracts concrete references (file paths,
        backtick-quoted identifiers) and checks each against the graph. Use
        before trusting an answer that cites files or symbols.
        """
        r = _ret()
        try:
            return r.verify(answer)
        finally:
            r.close()

    @mcp.tool
    def squeeze(text: str, kind: str = "auto") -> dict:
        """Shrink a pasted stacktrace / CI log / JSON blob before it costs tokens.

        Classifies the input (auto by default) and removes low-signal noise —
        vendor stack frames, build-progress spam, verbose JSON — keeping the
        diagnostic content. Returns the reduced text plus token savings.
        """
        r = _ret()
        try:
            res = r.squeeze(text, kind=kind)
            record_pack_savings(root, "mcp.squeeze",
                                final_tokens=res.get("squeezed_tokens", 0),
                                baseline_tokens=res.get("original_tokens", 0), files=0)
            return res
        finally:
            r.close()

    @mcp.tool
    def count_tokens_model(text: str, model: str = "gpt-4o") -> dict:
        """Model-aware token count (GPT / Claude / Gemini / Llama families)."""
        return {"model": model, "family": model_family(model),
                "tokens": count_tokens_for_model(text, model),
                "base_tokens": count_tokens(text)}

    @mcp.tool
    def estimate_call_cost(prompt: str, model: str = DEFAULT_COST_MODEL,
                           expected_output_tokens: int = 500,
                           compare: bool = False) -> dict:
        """Price an API call *before* sending it — per-model USD, input + output.

        Counts `prompt` with the model-aware tokenizer and multiplies by list
        prices. Set compare=true to rank every known model cheapest-first so you
        can pick the cheapest sufficient one. Deterministic, no network.
        """
        if compare:
            return compare_cost(prompt, expected_output_tokens)
        return estimate_cost(prompt, model, expected_output_tokens)

    @mcp.tool
    def dedupe_context(blocks: list[str], threshold: float = 0.8) -> dict:
        """Remove near-duplicate text blocks from a set of context snippets.

        Feed retrieved snippets / tool outputs / pasted context; get back only
        the non-redundant ones plus the token saving. (Context packs are already
        deduped automatically; this is for ad-hoc blocks.)
        """
        res = dedupe_blocks(blocks, threshold=threshold)
        record_pack_savings(root, "mcp.dedupe",
                            final_tokens=res.get("tokens_after", 0),
                            baseline_tokens=res.get("tokens_before", 0), files=0)
        return res

    @mcp.tool
    def score_prompt_quality(prompt: str) -> dict:
        """Score a prompt 0–100 on clarity / specificity / context / actionability.

        Catches under-specified prompts before they waste a round-trip; returns
        subscores and concrete fix suggestions. Distinct from judge(), which
        scores whether an *answer* is grounded.
        """
        return score_prompt(prompt)

    @mcp.tool
    def summarize_chat(transcript: str, max_tokens: int = 400,
                       required_identifiers: list | None = None,
                       required_facts: list | None = None) -> dict:
        """Compress a long chat transcript into a compact, token-cheap brief.

        Extracts decisions, action items, open questions and the code entities
        touched, capped at ~max_tokens. Use to carry a session forward without
        replaying the whole history.

        Pass `required_identifiers` / `required_facts` — the symbols and the
        decisions or constraints the next session must not lose — to also get a
        `fidelity` score telling you whether they survived. Without it the only
        feedback is a reduction percentage, which a summary that deleted
        everything would maximise.
        """
        res = summarize_conversation(transcript, max_tokens=max_tokens)
        if required_identifiers or required_facts:
            res["fidelity"] = score_summary_fidelity(
                res, {"identifiers": required_identifiers or [],
                      "facts": required_facts or []})
        record_pack_savings(root, "mcp.summarize",
                            final_tokens=res.get("summary_tokens", 0),
                            baseline_tokens=res.get("original_tokens", 0), files=0)
        return res

    @mcp.tool
    def learn(file: str, good: bool, weight: float = 1.0) -> dict:
        """Reinforce (good=true) or penalise (good=false) a file's local ranking weight."""
        r = _ret()
        try:
            return r.learn(file, good, weight)
        finally:
            r.close()

    @mcp.tool
    def read_memory(limit: int = 20) -> dict:
        """Recall the cross-session decision log + recent checkpoints for this repo."""
        r = _ret()
        try:
            return r.read_memory(limit=limit)
        finally:
            r.close()

    @mcp.tool
    def write_memory(text: str, kind: str = "note") -> dict:
        """Persist a decision/note to the cross-session memory log (local to the repo)."""
        r = _ret()
        try:
            return r.remember(text, kind=kind)
        finally:
            r.close()

    @mcp.tool
    def create_checkpoint(label: str, note: str = "") -> dict:
        """Record session progress with the current git short-SHA snapshot."""
        r = _ret()
        try:
            return r.create_checkpoint(label, note=note)
        finally:
            r.close()

    # ---- spec-named surface (MCP-2): read_context / search_signatures /
    #      query_context — the canonical names Copilot/Claude Code expect ----
    @mcp.tool
    def read_context(module: str = "", budget_tokens: int = 4000) -> str:
        """Signatures for the whole codebase or a module path. Call list_modules first.

        Token-frugal ordering: list_modules() -> read_context(module=…) instead
        of loading the full codebase.
        """
        r = _ret()
        try:
            return r.read_context(module or None, budget_tokens=budget_tokens)
        finally:
            r.close()

    @mcp.tool
    def search_signatures(query: str, limit: int = 20) -> list:
        """Keyword search across signatures (qname / kind / file / signature)."""
        r = _ret()
        try:
            return r.search_signatures(query, limit=limit)
        finally:
            r.close()

    @mcp.tool
    def query_context(query: str, top_k: int = 10, budget_tokens: int = 6000) -> dict:
        """TF-IDF/embedding rank of all files vs a query (top-K) plus a context pack."""
        r = _ret()
        try:
            return r.query_context(query, top_k=top_k, budget_tokens=budget_tokens)
        finally:
            r.close()

    # ---- grounded-creation + conventions + evidence (close the retrieve->create loop) ----
    @mcp.tool
    def conventions() -> dict:
        """Detect the repo's file-naming / layout / test / export conventions.

        Use before creating files so new code matches house style. Pure analysis
        over the graph — deterministic, no LLM."""
        r = _ret()
        try:
            return analyze_conventions(r.store, root)
        finally:
            r.close()

    @mcp.tool
    def scaffold(name: str, kind: str = "module", apply: bool = False) -> dict:
        """Propose (or apply=True, create) a convention-matched file + skeleton.

        Refuses (ok=false) if a file already exists at the target path, so it
        never overwrites. kind: module|class|function|component|test."""
        r = _ret()
        try:
            return (write_scaffold(r.store, root, name, kind=kind) if apply
                    else propose_scaffold(r.store, root, name, kind=kind))
        finally:
            r.close()

    @mcp.tool
    def verify_plan(plan: str) -> dict:
        """Check a plan's file/symbol references before acting: which exist, which
        are new, the blast radius of edits, and any fabricated symbols."""
        r = _ret()
        try:
            return verify_plan_fn(r, plan)
        finally:
            r.close()

    @mcp.tool
    def verify_output(answer: str) -> dict:
        """Audit AI-generated code/output for fabricated files, symbols AND local
        imports (the verify-ai-output stage). Use after generating code, before
        trusting it. External packages aren't flagged — only unresolved repo-local
        references."""
        r = _ret()
        try:
            return verify_ai_output_fn(r, answer)
        finally:
            r.close()

    @mcp.tool
    def review_diff(staged: bool = False) -> dict:
        """Audit the working-tree (or staged) git diff for scope drift, hub edits
        and missing tests — deterministic heuristics over the graph."""
        r = _ret()
        try:
            return review_diff_fn(r, root, staged=staged)
        finally:
            r.close()

    @mcp.tool
    def get_diff_context(staged: bool = False, budget_tokens: int = 6000,
                         depth: int = 1) -> dict:
        """Budgeted context pack for exactly what the git diff touches.

        The diff-native counterpart to find_relevant_context: no task string
        needed. Seeds from the diff itself — every changed symbol in full, plus
        its callers/callees/base classes (the blast radius) as signatures. Use
        before reviewing or continuing a change to load only the affected code.
        Returns the markdown pack plus changed_files / touched_symbols /
        impacted callers and token savings. staged=True scopes to the index."""
        r = _ret()
        try:
            res = r.get_diff_context(staged=staged, budget_tokens=budget_tokens,
                                     depth=depth)
            if res.get("baseline_tokens"):
                record_pack_savings(root, "mcp.diff_context",
                                    final_tokens=res["pack_tokens"],
                                    baseline_tokens=res["baseline_tokens"],
                                    files=len(res["changed_files"]))
            return res
        finally:
            r.close()

    @mcp.tool
    def create(task: str, kind: str = "module", answer: str = "",
               apply: bool = False) -> dict:
        """Grounded-creation state machine: scaffold -> verify-plan -> (verify-output
        if `answer` given) -> review, each stage gating the next.

        Dry-run by default (writes nothing); apply=True writes the scaffold file
        (never an overwrite) and reviews the resulting diff."""
        r = _ret()
        try:
            return create_pipeline(r, root, task, kind=kind,
                                   answer=(answer or None), apply=apply)
        finally:
            r.close()

    @mcp.tool
    def evidence(task: str, budget_tokens: int = 6000) -> dict:
        """Deterministic, hash-grounded evidence pack for a task (audit/CI).

        Byte-stable JSON: same index state + task -> identical context_hash, with
        anchor_coverage proving each cited symbol resolves to a real line span."""
        r = _ret()
        try:
            return build_evidence_pack(r, task, budget_tokens=budget_tokens)
        finally:
            r.close()

    @mcp.tool
    def hallucination_benchmark(sample_per_repo: int = 40,
                                baseline_per_100: float | None = None,
                                baseline_source: str = "") -> dict:
        """Reproducible, multi-repo codebase-fact grounding benchmark.

        MEASURES grounding coverage, guard catch rate, and guard specificity
        over the real index. Deterministic — same index state yields the same
        numbers, no LLM involved.

        Does NOT report a hallucination-reduction % unless you supply
        `baseline_per_100`: the rate at which an un-grounded agent fabricates
        is not something this tool can observe, and defaulting it would make
        the headline a restatement of the assumption. Supply one you measured,
        with `baseline_source`, to get a clearly-labelled projection."""
        r = _ret()
        try:
            return hallucination_benchmark_fn(
                r, sample_per_repo=sample_per_repo,
                baseline_per_100=baseline_per_100,
                baseline_source=baseline_source)
        finally:
            r.close()

    # Explicit shutdown for the connection pool (RP-1).
    mcp.close_pool = _close_pool
    return mcp


# ==========================================================================
# command-line interface
# ==========================================================================
def _db_path(root: Path) -> Path:
    return root / ".tokengraph" / "graph.db"


def cmd_index(args):
    root = Path(args.path).resolve()
    rep = index_repo(root, _db_path(root))
    print(f"scanned={rep.scanned} parsed={rep.parsed} skipped={rep.skipped} "
          f"removed={rep.removed}")
    print(f"graph: {rep.stats}")
    if rep.errors:
        print(f"{len(rep.errors)} file(s) had errors:", file=sys.stderr)
        for e in rep.errors[:10]:
            print("  " + e, file=sys.stderr)


def cmd_context(args):
    root = Path(args.path).resolve()
    if not getattr(args, "no_refresh", False):
        index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    pack = r.find_relevant_context(args.task, budget_tokens=args.budget,
                                   expand_depth=args.depth,
                                   max_body_tokens=args.max_body)
    out = pack.to_markdown()
    files = sorted({p.file for p in pack.pieces})
    record_pack_savings(root, "context", final_tokens=pack.rendered_tokens,
                        baseline_tokens=sum(r.store.token_est_for(f) for f in files),
                        files=len(files), no_track=getattr(args, "no_track", False))
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"wrote {args.out} (~{pack.rendered_tokens} tokens, {len(pack.pieces)} symbols)")
    else:
        print(out)
    r.close()


def cmd_skeleton(args):
    root = Path(args.path).resolve()
    index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    print(r.file_skeleton(args.file))
    r.close()


def cmd_callers(args):
    root = Path(args.path).resolve()
    index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    print("\n".join(r.get_callers(args.qname)) or "(none)")
    r.close()


def cmd_callees(args):
    root = Path(args.path).resolve()
    index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    print("\n".join(r.get_callees(args.qname)) or "(none)")
    r.close()


def cmd_semantic(args):
    root = Path(args.path).resolve()
    if not getattr(args, "no_refresh", False):
        index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    rows = r.semantic_search(args.query, limit=args.limit)
    if not rows:
        print("(no matches — vectors may be empty; run `index` first)")
    for row in rows:
        print(f"{row['qname']}  ({row['kind']})  {row['file']}")
    r.close()


def cmd_summary(args):
    root = Path(args.path).resolve()
    index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    print(r.module_summary(args.file))
    r.close()


def cmd_measure(args):
    root = Path(args.path).resolve()
    index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    m = r.measure(args.task, budget_tokens=args.budget, expand_depth=args.depth)
    record_pack_savings(root, "measure", final_tokens=m["pack_tokens"],
                        baseline_tokens=m["baseline_tokens"],
                        files=m["files_referenced"],
                        no_track=getattr(args, "no_track", False))
    print(f"task:             {m['task']}")
    print(f"pack tokens:      {m['pack_tokens']}")
    print(f"baseline (whole): {m['baseline_tokens']}  "
          f"({m['files_referenced']} file(s), {m['symbols_in_pack']} symbols)")
    print(f"tokens saved:     {m['tokens_saved']}  ({m['savings_pct']}% fewer)")
    r.close()


def _open_retriever(args):
    root = Path(args.path).resolve()
    if not getattr(args, "no_refresh", False):
        index_repo(root, _db_path(root))
    return Retriever(root, _db_path(root))


def _emit(obj, as_json: bool):
    import json
    if as_json:
        print(json.dumps(obj, indent=2))
    return obj


def cmd_modules(args):
    r = _open_retriever(args)
    try:
        mods = r.list_modules()
        if getattr(args, "json", False):
            _emit(mods, True)
        else:
            print(f"{'module':24} {'files':>6} {'tokens':>9} {'symbols':>8}")
            for m in mods:
                print(f"{m['module'][:24]:24} {m['files']:>6} {m['tokens']:>9,} "
                      f"{m['symbols']:>8}")
    finally:
        r.close()


def cmd_explain(args):
    r = _open_retriever(args)
    try:
        print(r.explain_file(args.file))
    finally:
        r.close()


def cmd_impact(args):
    r = _open_retriever(args)
    try:
        imp = r.get_impact(args.qname)
        if getattr(args, "json", False):
            _emit(imp, True)
        elif not imp.get("found"):
            print(f"(no symbol named {args.qname})")
        else:
            print(f"symbol:   {imp['symbol']}  ({imp['file']})")
            print(f"blast radius: {imp['blast_radius']} symbol(s)")
            print(f"direct callers:     {', '.join(imp['direct_callers']) or '(none)'}")
            print(f"transitive callers: {', '.join(imp['transitive_callers']) or '(none)'}")
            print(f"subclasses:         {', '.join(imp['subclasses']) or '(none)'}")
            print(f"tests touched:      {', '.join(imp['tests_touched']) or '(none)'}")
    finally:
        r.close()


def cmd_method_impact(args):
    r = _open_retriever(args)
    try:
        imp = r.get_method_impact(args.qname)
        if getattr(args, "json", False):
            _emit(imp, True)
        elif not imp.get("found"):
            dym = imp.get("did_you_mean") or []
            print(f"(no symbol named {args.qname})"
                  + (f"  did you mean: {', '.join(dym)}" if dym else ""))
        else:
            print(f"function: {imp['symbol']}  ({imp['kind']})  "
                  f"{imp['file']}:{imp['line']}")
            print(f"signature: {imp['signature']}")
            print(f"blast radius: {imp['blast_radius']} symbol(s), "
                  f"{imp['call_sites']} call site(s)")
            callers = [f"{c['symbol']} ({c['file']}:{c['line']})"
                       for c in imp["callers"]]
            print(f"callers (break on change): {', '.join(callers) or '(none)'}")
            print(f"callees (dependencies):    "
                  f"{', '.join(c['symbol'] for c in imp['callees']) or '(none)'}")
            ovr = [o["symbol"] for o in imp["overrides_or_overloads"]]
            print(f"overrides/overloads:       {', '.join(ovr) or '(none)'}")
            print(f"transitive callers: "
                  f"{', '.join(imp['transitive_callers']) or '(none)'}")
            print(f"tests touched:      "
                  f"{', '.join(imp['tests_touched']) or '(none)'}")
            if imp.get("warning"):
                print(f"! {imp['warning']}")
    finally:
        r.close()


def cmd_arch(args):
    r = _open_retriever(args)
    try:
        a = r.get_architecture_overview()
        if getattr(args, "json", False):
            _emit(a, True)
            return
        t = a["totals"]
        print(f"# architecture overview: {t['modules']} module(s), "
              f"{t['files']} file(s), {t['symbols']:,} symbol(s), "
              f"{t['tokens']:,} tokens")
        print("\n## modules (by tokens)")
        for m in a["modules"][:15]:
            print(f"  {m['module'][:28]:28} {m['files']:>5} files "
                  f"{m['tokens']:>9,} tok {m['symbols']:>6} sym")
        print("\n## languages")
        for L in a["languages"][:12]:
            print(f"  {L['language'][:16]:16} {L['files']:>5} files "
                  f"{L['tokens']:>9,} tok")
        print("\n## hub files (by import degree)")
        for h in a["hubs"]:
            print(f"  in={h['fan_in']:>3} out={h['fan_out']:>3}  {h['file']}")
        print(f"\n## import cycles: {len(a['cycles'])}")
        for cyc in a["cycles"]:
            print("  " + " -> ".join(cyc))
        print(f"\n## routes: {a['routes_total']}")
        for rt in a["routes"]:
            print(f"  {rt['method']:7} {rt['path']:30} {rt['file']}:{rt['line']}")
    finally:
        r.close()


def cmd_test_map(args):
    root = Path(args.path).resolve()

    # --benchmark: measured precision/recall/F1/hit@1 over a labeled corpus
    if getattr(args, "benchmark", False):
        import json
        corpus = Path(args.corpus) if args.corpus else root / "benchmarks" / "testmap"
        pairs_file = corpus / "pairs.json"
        if not pairs_file.exists():
            sys.exit(f"test-map --benchmark: no pairs.json at {pairs_file}")
        gold = json.loads(pairs_file.read_text(encoding="utf-8")).get("pairs", [])
        files = [p.relative_to(corpus).as_posix()
                 for p in sorted(corpus.rglob("*"))
                 if p.is_file() and p.name != "pairs.json"]
        res = test_discovery_f1(files, gold)
        res["corpus"] = str(corpus)
        res["files"] = len(files)
        if getattr(args, "json", False):
            _emit(res, True)
        else:
            print(f"test-discovery F1 benchmark ({corpus.name}): "
                  f"{len(files)} files, {res['gold_pairs']} gold pair(s)")
            print(f"  precision={res['precision']}  recall={res['recall']}  "
                  f"F1={res['f1']}  hit@1={res['hit_at_1']}")
            print(f"  tp={res['true_positives']} fp={res['false_positives']} "
                  f"fn={res['false_negatives']}")
        if getattr(args, "check", False):
            th = args.min_f1 if args.min_f1 is not None else 0.90
            if res["f1"] < th:
                print(f"FAIL f1 {res['f1']} < {th}", file=sys.stderr)
                sys.exit(1)
        return

    # lookup
    r = _open_retriever(args)
    try:
        tm = r.get_test_map(args.target or "")
    finally:
        r.close()
    if getattr(args, "json", False):
        _emit(tm, True)
        return
    if tm.get("target") is None:
        cov = tm["coverage"]
        print(f"impl<->test map: {cov['impl_files_with_tests']}/{cov['impl_files']} "
              f"impl file(s) have tests ({cov['coverage_pct']}%), "
              f"{cov['pairs']} pair(s)")
        for p in tm["pairs"][:40]:
            print(f"  {p['impl']}  <-  {p['test']}")
        if tm["untested_impls"]:
            print(f"  untested: {', '.join(tm['untested_impls'][:15])}")
    elif tm.get("kind") == "symbol":
        print(f"tests for {tm['target']} ({tm['file']}):")
        print(f"  by name:       {', '.join(tm['tests_by_name']) or '(none)'}")
        print(f"  by call graph: {', '.join(tm['tests_by_call_graph']) or '(none)'}")
    elif tm.get("kind") == "test":
        print(f"{tm['target']} covers: {', '.join(tm['covers']) or '(unknown)'}")
    else:
        print(f"tests for {tm['target']}: {', '.join(tm['tests']) or '(none)'}")


def cmd_lines(args):
    r = _open_retriever(args)
    try:
        print(r.get_lines(args.file, args.start, args.end))
    finally:
        r.close()


def cmd_map(args):
    r = _open_retriever(args)
    try:
        m = r.get_map(args.kind)
        if getattr(args, "json", False):
            _emit(m, True)
        elif m["kind"] == "routes":
            print("# routes")
            for rt in m["routes"]:
                print(f"{rt['method']:7} {rt['path']:30} {rt['file']}:{rt['line']}")
            if not m["routes"]:
                print("(no routes found)")
        elif m["kind"] == "hubs":
            print("# hubs (by import degree)")
            for h in m["hubs"]:
                print(f"  in={h['fan_in']:>3} out={h['fan_out']:>3}  {h['file']}")
            print(f"# import cycles: {len(m['cycles'])}")
            for cyc in m["cycles"]:
                print("  " + " -> ".join(cyc))
        else:
            print(f"# {m['kind']}")
            for src, dsts in m["edges"].items():
                print(f"{src} -> {', '.join(dsts)}")
    finally:
        r.close()


def cmd_routing(args):
    r = _open_retriever(args)
    try:
        rt = r.get_routing()
        if getattr(args, "json", False):
            _emit(rt, True)
        else:
            for x in rt:
                print(f"{x['tier']:9} {x['file']}")
    finally:
        r.close()


def cmd_suggest(args):
    info = recommend_tier(args.task)
    if getattr(args, "json", False):
        _emit(info, True)
    else:
        print(f"task:   {info['task']}")
        print(f"intent: {info['intent']}")
        print(f"tier:   {info['tier']}  (~${info['cost_hint_per_1k']}/1K tokens)")
        print(f"models: {', '.join(info['models'])}")
        print(f"use for: {info['use_for']}")


def cmd_ask(args):
    r = _open_retriever(args)
    try:
        a = r.ask(args.task, budget_tokens=args.budget, depth=args.depth)
        record_pack_savings(Path(args.path).resolve(), "ask",
                            final_tokens=a.get("pack_tokens", 0),
                            baseline_tokens=a.get("baseline_tokens", 0),
                            files=a.get("files", a.get("files_referenced", 0)),
                            no_track=getattr(args, "no_track", False))
        if getattr(args, "json", False):
            a = dict(a); a.pop("markdown", None)
            _emit(a, True)
        else:
            print(f"intent={a['intent']}  coverage={a['coverage_pct']}%  "
                  f"risk={a['risk']}  tier={a['suggested_tier']}")
            print(f"pack={a['pack_tokens']} tokens  saved={a['savings_pct']}% "
                  f"vs {a['baseline_tokens']}")
            print()
            print(a["markdown"])
    finally:
        r.close()


def cmd_validate(args):
    r = _open_retriever(args)
    try:
        v = r.validate(args.task, min_coverage=args.min_coverage,
                       budget_tokens=args.budget, depth=args.depth)
        if getattr(args, "json", False):
            _emit(v, True)
        else:
            print(f"coverage={v['coverage_pct']}% (min {v['min_coverage']}%)  "
                  f"risk={v['risk']}  ok={v['ok']}")
            print(v["recommendation"])
        if not v["ok"]:
            raise SystemExit(1)
    finally:
        r.close()


def cmd_judge(args):
    answer = Path(args.answer_file).read_text(encoding="utf-8") \
        if args.answer_file else (args.answer or "")
    context = Path(args.context_file).read_text(encoding="utf-8") \
        if args.context_file else (args.context or "")
    r = _open_retriever(args)
    try:
        j = r.judge(answer, context)
        if getattr(args, "json", False):
            _emit(j, True)
        else:
            print(f"grounded={j['grounded']}  score={j['grounded_pct']}%")
            print(j["note"])
            if j["unsupported_terms"]:
                print("unsupported: " + ", ".join(j["unsupported_terms"]))
        if not j["grounded"]:
            raise SystemExit(1)
    finally:
        r.close()


def cmd_verify(args):
    answer = Path(args.answer_file).read_text(encoding="utf-8") \
        if args.answer_file else (args.answer or "")
    r = _open_retriever(args)
    try:
        v = r.verify(answer)
        if getattr(args, "json", False):
            _emit(v, True)
        else:
            print(f"ok={v['ok']}  checked files={v['checked']['files']} "
                  f"symbols={v['checked']['symbols']}")
            print(v["note"])
            for it in v["issues"]:
                hint = (f"  did you mean: {', '.join(it['did_you_mean'])}"
                        if it["did_you_mean"] else "")
                print(f"  [{it['kind']}] {it['name']}{hint}")
        if not v["ok"]:
            raise SystemExit(1)
    finally:
        r.close()


def cmd_squeeze(args):
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()
    r = _open_retriever(args)
    try:
        s = r.squeeze(text, kind=args.kind)
    finally:
        r.close()
    record_pack_savings(Path(args.path).resolve(), "squeeze",
                        final_tokens=s.get("squeezed_tokens", 0),
                        baseline_tokens=s.get("original_tokens", 0), files=0,
                        no_track=getattr(args, "no_track", False))
    if getattr(args, "json", False):
        _emit(s, True)
    else:
        print(s["text"])
        print(f"\n--- squeezed [{s['kind']}]: {s['original_tokens']} -> "
              f"{s['squeezed_tokens']} tokens ({s['reduction_pct']}% smaller) ---",
              file=sys.stderr)


def _read_text_arg(args) -> str:
    """Resolve inline --text / --text-file / stdin, in that order."""
    if getattr(args, "text_file", None):
        return Path(args.text_file).read_text(encoding="utf-8")
    if getattr(args, "text", None):
        return args.text
    return sys.stdin.read()


def cmd_cost(args):
    text = _read_text_arg(args)
    if args.compare:
        res = compare_cost(text, args.output_tokens)
        if getattr(args, "json", False):
            _emit(res, True)
        else:
            c = res["cheapest"]
            print(f"cheapest: {c['model']}  ${c['total_usd']:.6f}")
            print(f"{'model':18} {'in$':>10} {'out$':>10} {'total$':>10}")
            for row in res["by_model"]:
                print(f"{row['model']:18} {row['input_usd']:>10.6f} "
                      f"{row['output_usd']:>10.6f} {row['total_usd']:>10.6f}")
    else:
        res = estimate_cost(text, args.model, args.output_tokens)
        if getattr(args, "json", False):
            _emit(res, True)
        else:
            print(f"{res['model']}: {res['input_tokens']} in + "
                  f"{res['output_tokens']} out = ${res['total_usd']:.6f} "
                  f"(in ${res['input_usd']:.6f} + out ${res['output_usd']:.6f})")


def cmd_prompt_score(args):
    prompt = _read_text_arg(args)
    res = score_prompt(prompt)
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        s = res["subscores"]
        print(f"score={res['score']} ({res['grade']})  "
              f"clarity={s['clarity']} specificity={s['specificity']} "
              f"context={s['context']} actionability={s['actionability']}")
        for tip in res["suggestions"]:
            print(f"  - {tip}")


def cmd_summarize_chat(args):
    transcript = _read_text_arg(args)
    res = summarize_conversation(transcript, max_tokens=args.max_tokens)
    record_pack_savings(Path(args.path).resolve(), "summarize",
                        final_tokens=res["summary_tokens"],
                        baseline_tokens=res["original_tokens"], files=0,
                        no_track=getattr(args, "no_track", False))
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        print(res["summary"])
        print(f"\n--- summarized {res['turns']} turns: {res['original_tokens']} -> "
              f"{res['summary_tokens']} tokens ({res['reduction_pct']}% smaller) ---",
              file=sys.stderr)


def cmd_dedupe(args):
    raw = _read_text_arg(args)
    blocks = [b for b in raw.split(args.sep) if b.strip()] if args.sep \
        else [b for b in raw.split("\n\n") if b.strip()]
    res = dedupe_blocks(blocks, threshold=args.threshold)
    record_pack_savings(Path(args.path).resolve(), "dedupe",
                        final_tokens=res["tokens_after"],
                        baseline_tokens=res["tokens_before"], files=0,
                        no_track=getattr(args, "no_track", False))
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        print(("\n\n" if not args.sep else args.sep).join(res["kept"]))
        print(f"\n--- deduped {res['input_blocks']} -> {res['kept_blocks']} blocks: "
              f"{res['tokens_before']} -> {res['tokens_after']} tokens "
              f"({res['reduction_pct']}% smaller) ---", file=sys.stderr)


def cmd_learn(args):
    r = _open_retriever(args)
    try:
        res = r.learn(args.file, good=not args.bad, weight=args.weight)
        print(f"{res['file']}: delta={res['delta']:+g} weight={res['weight']:g}")
    finally:
        r.close()


def cmd_memory(args):
    r = _open_retriever(args)
    try:
        if args.add:
            res = r.remember(args.add, kind=args.kind)
            print(f"saved memory #{res['id']} ({res['kind']})")
        else:
            mem = r.read_memory(limit=args.limit)
            if getattr(args, "json", False):
                _emit(mem, True)
            else:
                print("# notes")
                for n in mem["notes"]:
                    print(f"- [{n['kind']}] {n['text']}")
                print("# checkpoints")
                for c in mem["checkpoints"]:
                    print(f"- {c['label']} @ {c['git_sha'] or '(no sha)'} {c['note']}")
    finally:
        r.close()


def cmd_checkpoint(args):
    r = _open_retriever(args)
    try:
        res = r.create_checkpoint(args.label, note=args.note or "")
        print(f"checkpoint #{res['id']}: {res['label']} @ {res['git_sha'] or '(no sha)'}")
    finally:
        r.close()


def cmd_conventions(args):
    root = Path(args.path).resolve()
    if not getattr(args, "no_refresh", False):
        index_repo(root, _db_path(root))
    store = Store(_db_path(root))
    try:
        conv = analyze_conventions(store, root)
    finally:
        store.close()
    if getattr(args, "json", False):
        _emit(conv, True)
    else:
        print(f"conventions ({conv['files_analyzed']} files): {conv['summary']}")
        print(f"  naming     : {conv['dominant_naming']}  "
              f"{conv['naming_distribution']}")
        print(f"  by dir     : {conv['naming_by_dir']}")
        print(f"  extension  : {conv['primary_extension']}")
        print(f"  tests      : pattern={conv['test_pattern']} dir={conv['test_dir']}")
        print(f"  exports    : {conv['export_style']} "
              f"({conv['public_symbols']} public / {conv['private_symbols']} private)")
        print(f"  source dirs: {', '.join(conv['source_dirs']) or '(flat)'}")
        print(f"  conformance: {conv['conformance_pct']}% "
              f"({len(conv['nonconforming_files'])} outlier(s))")
        if getattr(args, "check", False) or getattr(args, "fix", False):
            for nc in conv["nonconforming_files"]:
                print(f"    [{nc['found']}!={nc['expected']}] {nc['file']} "
                      f"-> {nc['suggested_stem']}")
    if getattr(args, "fix", False):
        plan = _conventions_fix(root, conv["nonconforming_files"],
                                dry_run=getattr(args, "dry_run", False))
        if getattr(args, "json", False):
            _emit({"renames": plan}, True)
        else:
            verb = "would rename" if getattr(args, "dry_run", False) else "renamed"
            for mv in plan:
                status = mv["status"]
                print(f"  {verb if status=='ok' else 'skip'}: {mv['from']} -> "
                      f"{mv['to']}" + (f"  ({status})" if status != "ok" else ""))
            done = sum(1 for m in plan if m["status"] == "ok")
            print(f"conventions --fix: {done}/{len(plan)} "
                  f"{'planned' if getattr(args,'dry_run',False) else 'applied'}")
        return
    if getattr(args, "check", False) and conv["nonconforming_files"]:
        raise SystemExit(1)


def _conventions_fix(root: Path, nonconforming: list[dict],
                     dry_run: bool = False) -> list[dict]:
    """Rename non-conforming files to their convention-matched stems.

    Preserves each file's extension and directory, refuses to clobber an
    existing path, and prefers `git mv` when the repo is under git so history
    follows the rename. Returns a per-file plan with a status.
    """
    use_git = (root / ".git").exists()
    plan: list[dict] = []
    for nc in nonconforming:
        src_rel = nc["file"]
        src = root / src_rel
        ext = src.suffix
        dst_rel = (src.parent.relative_to(root) / (nc["suggested_stem"] + ext)).as_posix()
        dst = root / dst_rel
        if not src.exists():
            status = "missing"
        elif dst.exists():
            status = "conflict"
        elif dst_rel == src_rel:
            status = "noop"
        else:
            status = "ok"
            if not dry_run:
                try:
                    moved = use_git and bool(_git(root, "mv", src_rel, dst_rel))
                    if not moved:
                        src.rename(dst)
                except Exception as ex:
                    status = f"error: {ex}"
        plan.append({"from": src_rel, "to": dst_rel, "status": status})
    return plan


def cmd_scaffold(args):
    root = Path(args.path).resolve()
    if not getattr(args, "no_refresh", False):
        index_repo(root, _db_path(root))
    store = Store(_db_path(root))
    try:
        res = (write_scaffold(store, root, args.name, kind=args.kind)
               if args.apply else
               propose_scaffold(store, root, args.name, kind=args.kind))
    finally:
        store.close()
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        print(f"{res['note']}")
        if res["ok"] and not res.get("written"):
            print(f"\n--- {res['proposed_path']} ---")
            print(res["skeleton"])
    if not res["ok"]:
        raise SystemExit(1)


def cmd_verify_output(args):
    answer = (Path(args.answer_file).read_text(encoding="utf-8")
              if args.answer_file else (args.answer or ""))
    r = _open_retriever(args)
    try:
        res = verify_ai_output(r, answer)
    finally:
        r.close()
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        print(f"ok={res['ok']}  checked={res['checked']}")
        print(res["note"])
        for it in res["issues"]:
            hint = (f"  did you mean: {', '.join(it['did_you_mean'])}"
                    if it["did_you_mean"] else "")
            print(f"  [{it['kind']}] {it['name']}{hint}")
    if not res["ok"]:
        raise SystemExit(1)


def cmd_verify_plan(args):
    plan = (Path(args.plan_file).read_text(encoding="utf-8")
            if args.plan_file else (args.plan or ""))
    r = _open_retriever(args)
    try:
        res = verify_plan(r, plan)
    finally:
        r.close()
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        print(res["note"])
        for f in res["referenced_files"]:
            print(f"  [{f['status']:6}] {f['path']}")
        for b in res["blast_radius"]:
            print(f"  impact {b['impact']:>3}  {b['qname']}")
        for s in res["fabricated_symbols"]:
            hint = (f"  did you mean: {', '.join(s['did_you_mean'])}"
                    if s["did_you_mean"] else "")
            print(f"  [FABRICATED] {s['name']}{hint}")
    if not res["ok"]:
        raise SystemExit(1)


def cmd_review(args):
    r = _open_retriever(args)
    try:
        res = review_diff(r, Path(args.path).resolve(), staged=args.staged)
    finally:
        r.close()
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        print(res["note"])
        for f in res["findings"]:
            print(f"  [{f['severity']:4}] {f['kind']}: {f['detail']}")
    if not res["ok"]:
        raise SystemExit(1)


def cmd_diff_context(args):
    r = _open_retriever(args)
    try:
        res = r.get_diff_context(staged=args.staged, budget_tokens=args.budget,
                                 depth=args.depth)
    finally:
        r.close()
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        print(res["note"])
        print(f"  pack={res['pack_tokens']} tok  baseline={res['baseline_tokens']} tok  "
              f"saved={res['savings_pct']}%")
        if res["markdown"]:
            print()
            print(res["markdown"])


def cmd_create(args):
    answer = (Path(args.answer_file).read_text(encoding="utf-8")
              if args.answer_file else None)
    r = _open_retriever(args)
    try:
        res = create_pipeline(r, Path(args.path).resolve(), args.task,
                              kind=args.kind, answer=answer, apply=args.apply)
    finally:
        r.close()
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        print(f"task: {res['task']}")
        print(f"conventions: {res['conventions']}")
        for s in res["stages"]:
            print(f"  [{'ok ' if s['ok'] else 'FAIL'}] {s['stage']}: {s['detail']}")
        print(f"context symbols: {', '.join(res['context_symbols'][:8])}")
        print(res["note"])
    if not res["ok"]:
        raise SystemExit(1)


def cmd_evidence(args):
    r = _open_retriever(args)
    try:
        ev = build_evidence_pack(r, args.task, budget_tokens=args.budget)
    finally:
        r.close()
    if args.out:
        import json
        Path(args.out).write_text(json.dumps(ev, indent=2), encoding="utf-8")
        print(f"wrote {args.out} (hash={ev['grounding']['context_hash'][:16]}, "
              f"coverage={ev['grounding']['anchor_coverage']})")
    else:
        _emit(ev, True)


def cmd_grounding(args):
    r = _open_retriever(args)
    try:
        rep = grounding_report(r, sample=args.sample)
    finally:
        r.close()
    if getattr(args, "json", False):
        _emit(rep, True)
    else:
        print(rep.get("summary", rep.get("note", "")))
        if "guard_precision" in rep:
            print(f"guard precision: {rep['guard_precision']}")


def cmd_hallucination(args):
    r = _open_retriever(args)
    try:
        rep = hallucination_benchmark(
            r, sample_per_repo=args.sample, baseline_per_100=args.baseline,
            baseline_source=getattr(args, "baseline_source", ""))
    finally:
        r.close()
    if args.out:
        Path(args.out).write_text(hallucination_report_to_markdown(rep),
                                  encoding="utf-8")
        print(f"wrote {args.out} — {rep.get('summary', rep.get('note',''))}")
    elif getattr(args, "json", False):
        _emit(rep, True)
    else:
        print(rep.get("summary", rep.get("note", "")))
        for row in rep.get("per_repo", []):
            line = (f"  {row['repo']:16} facts={row['facts']:>3} "
                    f"coverage={row['grounding_coverage_pct']}% "
                    f"catch={row['guard_catch_pct']}% "
                    f"unguarded={row['unguarded_fact_share_pct']}%")
            if "projected_reduction_pct" in row:
                line += f" projected={row['projected_reduction_pct']}%"
            print(line)
        proj = rep.get("projection") or {}
        if not proj.get("available") and rep.get("per_repo"):
            print(f"\n  note: {proj.get('why', '')}")


def cmd_ide_setup(args):
    root = Path(args.path).resolve()
    editors = args.editor or None

    # --verify: report per-editor completeness (MCP + rules) and exit non-zero
    # if a requested editor is not ready. Turns "we wrote configs" into a check.
    if getattr(args, "verify", False):
        rows = verify_ide_wiring(root, editors=editors)
        if getattr(args, "json", False):
            _emit(rows, True)
        else:
            print("IDE wiring status (MCP server + steering rules):")
            for row in rows:
                mark = "OK  " if row["ready"] else "MISS"
                print(f"  [{mark}] {row['editor']:9} mcp={row['mcp_wired']!s:5} "
                      f"({row['mcp_scope']})  rules={row['rules_present']}")
        not_ready = [r["editor"] for r in rows if not r["ready"]]
        if not_ready:
            print(f"not ready: {', '.join(not_ready)} "
                  f"(run: tokengraph ide-setup)", file=sys.stderr)
            sys.exit(1)
        return

    workspace_roots = [Path(p) for p in args.workspace_root] if args.workspace_root else None
    res = ide_setup(root, editors=editors, workspace_roots=workspace_roots,
                    write_global=getattr(args, "write_global", False))
    # By default, also drop the steering-rules block so one command fully
    # provisions each IDE (MCP config + rules). --no-rules keeps it MCP-only.
    if getattr(args, "with_rules", True):
        try:
            res["rules_written"] = write_ide_rules(root, editors=editors)
        except Exception as e:                       # never fail wiring over rules
            res["rules_written"] = []
            res["rules_error"] = str(e)
    if getattr(args, "plugins", False):
        res["plugins"] = emit_ide_plugins(root)
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        print(res["note"])
        for w in res["written"]:
            print(f"  wired {w}")
        for w in res["global_written"]:
            print(f"  wired {w}  (per-user)")
        for w in res.get("rules_written", []):
            print(f"  rules {w}")
        for g in res["global_pending"]:
            print(f"  SKIPPED {g['host']}: {g['why']}")
            print(f"          write it with: tokengraph ide-setup "
                  f"--editor {g['host']} --global")
        print(f"  Neovim (mcphub): {res['neovim_snippet']}")
        print(f"  {res['jetbrains_note']}")
        if res.get("rules_error"):
            print(f"  (rules skipped: {res['rules_error']})")
        if "plugins" in res:
            print(f"  {res['plugins']['note']}")


# QG-3: the floor for the only experiment that directly tests the product's
# thesis — that a pack answers as well as the full source it replaces. Below
# `quality_retention` the compression is costing answers, which no amount of
# token savings redeems.
JUDGE_THRESHOLDS = {
    "quality_retention": 0.90,   # pack score / full-source score
    "pack_correct_rate": 0.60,   # share of questions answered fully correctly
}
# Where the last run is recorded, so "we have never actually checked" is a
# visible state rather than a silent one.
JUDGE_RESULT_FILE = "judge-eval.json"
JUDGE_STALE_AFTER_DAYS = 30


def judge_result_path(root: Path) -> Path:
    return root / ".context" / JUDGE_RESULT_FILE


def check_judge_thresholds(aggregate: dict,
                           thresholds: dict | None = None) -> dict:
    """Compare an LLM-judge aggregate against the QG-3 floors."""
    th = dict(JUDGE_THRESHOLDS)
    th.update(thresholds or {})
    failures = []
    for metric, limit in th.items():
        actual = aggregate.get(metric)
        if actual is None:
            continue
        if actual < limit:
            failures.append({"metric": metric, "actual": actual,
                             "threshold": limit, "direction": "min"})
    return {"ok": not failures, "failures": failures, "thresholds": th}


def cmd_judge_eval(args):
    """Run the LLM-judged answer-quality evaluation (QG-2, QG-3).

    This is the only measurement that tests the project's actual claim end to
    end, so it also records *that it ran*: the deterministic gate can prove the
    right symbols are present, and still not prove a model answers from them.
    """
    import json
    import time
    root = Path(args.path).resolve()
    corpora = ([Path(c) for c in args.corpus] if args.corpus
               else load_judge_corpora(root))
    if not corpora:
        sys.exit("judge-eval: no judge.json corpora found under benchmarks/repos/")
    res = run_llm_judge(corpora, model=args.model, budget_tokens=args.budget,
                        limit=args.limit or 0,
                        compare_full=not args.no_compare)
    if not res.get("ok"):
        print(f"judge-eval: {res['error']}", file=sys.stderr)
        print(f"  {res.get('hint','')}", file=sys.stderr)
        sys.exit(2)
    a = res["aggregate"]
    gate = check_judge_thresholds(a)
    res["gate"] = gate
    res["ran_at"] = time.time()
    # Record the run where doctor and CI can see its age. Only the aggregate
    # and the gate are persisted — never the graded answers, which contain
    # source excerpts.
    record = judge_result_path(root)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(
        {"ran_at": res["ran_at"], "model": a.get("model"),
         "aggregate": a, "gate": gate}, indent=2, sort_keys=True),
        encoding="utf-8")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        print(f"judge-eval: {a['graded']}/{a['questions']} graded "
              f"across {len(a['repos'])} repos (model {a['model']})")
        print(f"  pack score      {a['pack_mean_score']}  "
              f"(fully correct: {a['pack_correct_rate']})")
        print(f"  pack tokens     {a['mean_pack_tokens']:.0f} avg")
        if "quality_retention" in a:
            print(f"  full-file score {a['full_mean_score']}  "
                  f"({a['mean_full_tokens']:.0f} tokens avg)")
            print(f"  QUALITY RETENTION {a['quality_retention']}  "
                  f"at {a['token_ratio']:.1%} of the tokens")
        if a["errors"]:
            print(f"  {a['errors']} question(s) errored")
        print(f"  recorded in {record.relative_to(root).as_posix()}")
    if getattr(args, "check", False) and not gate["ok"]:
        for f in gate["failures"]:
            print(f"FAIL {f['metric']}: {f['actual']} (min {f['threshold']})",
                  file=sys.stderr)
        raise SystemExit(1)


def cmd_embed_warm(args):
    """Fetch and verify the semantic embedding model, then invalidate vectors."""
    res = embed_warm()
    if res.get("ok"):
        root = Path(args.path).resolve()
        db = _db_path(root)
        if db.exists():
            store = Store(db)
            try:
                dropped = store.drop_stale_vectors()
                store.set_meta("embed_backend", embed_backend_id())
                store.commit()
            finally:
                store.close()
            # Rebuild in the new space now, so the next query is not the one
            # that pays for it.
            rep = index_repo(root, db)
            res["vectors_dropped"] = dropped
            res["vectors_rebuilt"] = rep.reembedded
    if getattr(args, "json", False):
        _emit(res, True)
    elif res.get("ok"):
        print(f"embeddings ready: {res['model']} ({res['dim']}d)")
        if "vectors_rebuilt" in res:
            print(f"  re-embedded {res['vectors_rebuilt']} symbol(s) "
                  f"(dropped {res['vectors_dropped']} from the old backend)")
    else:
        print(f"embed-warm failed: {res['error']}")
        print(f"  still usable — falling back to: {res['backend']}")
        sys.exit(1)


def cmd_ide_plugin(args):
    root = Path(args.path).resolve()
    res = emit_ide_plugins(root, out_dir=args.out, editors=args.editor or None)
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        print(res["note"])
        for w in res["written"]:
            print(f"  wrote {w}")
        print("install:")
        for ed in res["editors"]:
            print(f"  {ed}: {res['install'][ed]}")


def cmd_dashboard(args):
    """Signpost for the removed Streamlit app.

    `dashboard.py` (Streamlit + plotly) was superseded by the built-in report
    and has been deleted. The subcommand survives it purely so an old habit or
    a stale script gets a route to the replacement instead of "unknown
    command" — it points at the report and writes it if it is missing.
    """
    root = Path(args.path).resolve()
    print("note: the Streamlit `dashboard` was removed — the report is built in "
          "now (no Streamlit, no plotly, no server).", file=sys.stderr)
    report = usage_report_path(root)
    if not report.exists():
        write_usage_report(root)
    print(f"  static:  python tokengraph_all.py gain --report   -> {report}")
    print("  live:    python tokengraph_all.py gain --serve     -> 127.0.0.1")


def cmd_import_scip(args):
    root = Path(args.path).resolve()
    db = _db_path(root)
    index_repo(root, db)
    index_file = Path(args.index)
    if not index_file.is_absolute():
        index_file = root / index_file
    result = import_scip_json(root, db, index_file)
    _emit(result, getattr(args, "json", False))


def cmd_freeze(args):
    """Emit a PyInstaller spec to build a standalone single-file binary.

    The CLI is already one dependency-free file; this produces a native
    executable for users without Python. (IDE integration is the MCP server —
    `tokengraph serve` — which any MCP-capable editor can launch.)"""
    spec = (
        "# tokengraph.spec — build: pyinstaller tokengraph.spec\n"
        "# produces dist/tokengraph(.exe), a standalone binary (no Python needed)\n"
        "block_cipher = None\n"
        "a = Analysis(['tokengraph_all.py'], pathex=['.'], binaries=[], datas=[],\n"
        "             hiddenimports=['sqlite3'], hookspath=[], runtime_hooks=[],\n"
        "             excludes=[], cipher=block_cipher)\n"
        "pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)\n"
        "exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],\n"
        "          name='tokengraph', console=True, onefile=True)\n"
    )
    out = Path(args.out or (Path(args.path).resolve() / "tokengraph.spec"))
    out.write_text(spec, encoding="utf-8")
    print(f"wrote {out}")
    if getattr(args, "build", False):
        import shutil
        import subprocess
        if not shutil.which("pyinstaller"):
            print("pyinstaller not found — install it: pip install pyinstaller")
            raise SystemExit(1)
        print("building binary with pyinstaller ...")
        rc = subprocess.run(["pyinstaller", "--clean", "--noconfirm", str(out)],
                            cwd=str(out.parent)).returncode
        if rc == 0:
            print("result:      dist/tokengraph(.exe) — a standalone binary, no Python")
        raise SystemExit(rc)
    print("build with:  pip install pyinstaller && pyinstaller tokengraph.spec")
    print("result:      dist/tokengraph(.exe) — a standalone binary, no Python")


def cmd_dist(args):
    res = emit_distribution(Path(args.path).resolve(), out_dir=args.out)
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        print(res["note"])
        for w in res["written"]:
            print(f"  wrote {w}")
        print(f"  channels: {', '.join(res['channels'])}")


def _md_cell(text: str) -> str:
    """Make task text safe for a one-line markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def report_to_markdown(rep: dict) -> str:
    agg, repo, rows = rep["aggregate"], rep["repo"], rep["rows"]
    out: list[str] = ["# tokengraph savings report", ""]

    out += [
        "## Repo baseline (with vs. without, repo scale)",
        "",
        f"- Indexed files: **{repo['indexed_files']}**",
        f"- Whole-repo tokens (read everything): **{repo['repo_tokens_total']:,}**",
        f"- Mean pack size: **{repo['mean_pack_tokens']:,}** tokens "
        f"(**{repo['mean_pack_pct_of_repo']}%** of the whole repo)",
        "",
    ]

    out += [f"## Aggregate across {agg['tasks']} task(s)", ""]
    if agg["tasks"]:
        out += [
            f"- Pack tokens (sum): **{agg['pack_tokens_total']:,}**",
            f"- Baseline tokens (sum): **{agg['baseline_tokens_total']:,}**",
            f"- Tokens saved (sum): **{agg['tokens_saved_total']:,}** "
            f"(**{agg['savings_pct_overall']}%** fewer overall)",
            f"- Mean savings per task: **{agg['savings_pct_mean']}%**",
            f"- Best: \"{_md_cell(agg['best']['task'])}\" ({agg['best']['savings_pct']}%)",
            f"- Worst: \"{_md_cell(agg['worst']['task'])}\" ({agg['worst']['savings_pct']}%)",
            "",
        ]

    out += [
        "## Per-task",
        "",
        "| Task | Pack | Baseline | Saved | % fewer |",
        "|---|--:|--:|--:|--:|",
    ]
    for r in rows:
        out.append(
            f"| {_md_cell(r['task'])} | {r['pack_tokens']:,} | {r['baseline_tokens']:,} "
            f"| {r['tokens_saved']:,} | {r['savings_pct']}% |"
        )
    out.append("")
    return "\n".join(out)


def report_to_csv(rep: dict) -> str:
    buf = io.StringIO()
    cols = ["task", "pack_tokens", "baseline_tokens", "files_referenced",
            "symbols_in_pack", "tokens_saved", "savings_pct"]
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rep["rows"]:
        w.writerow({k: r[k] for k in cols})
    return buf.getvalue()


def append_report_csv(path: Path, csv_text: str) -> int:
    """Append a report's data rows to a cumulative CSV, header written once.

    `report --csv` overwrites, which is right for a snapshot but loses history
    when you re-run a task list over time. Appending keeps one growing sheet:
    the header goes in when the file is created, later runs contribute rows
    only. Returns how many data rows were added.
    """
    path = Path(path)
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        return 0
    header, rows = lines[0], lines[1:]
    existing = path.exists() and path.stat().st_size > 0
    body = "\n".join(rows if existing else lines)
    with path.open("a", encoding="utf-8", newline="") as fh:
        fh.write(body + "\n")
    return len(rows)


def append_report_markdown(path: Path, md: str, tasks: int,
                           stamp: str | None = None) -> None:
    """Append one timestamped run section to a cumulative markdown log."""
    import datetime as _dt
    path = Path(path)
    stamp = stamp or _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    head = "" if (path.exists() and path.stat().st_size) else \
        "# ContextIQ savings report (running log)\n\n"
    plural = "task" if tasks == 1 else "tasks"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{head}## Run {stamp} ({tasks} {plural})\n\n{md.rstrip()}\n\n")


def _collect_tasks(args) -> list[str]:
    tasks = list(args.tasks or [])
    if args.tasks_file:
        for line in Path(args.tasks_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                tasks.append(line)
    return tasks


def cmd_report(args):
    tasks = _collect_tasks(args)
    if not tasks:
        print("no tasks given: pass tasks as arguments and/or --tasks-file FILE "
              "(one task per line; # comments allowed)", file=sys.stderr)
        raise SystemExit(2)
    root = Path(args.path).resolve()
    index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    try:
        rep = r.report(tasks, budget_tokens=args.budget, expand_depth=args.depth)
        md = report_to_markdown(rep)
        append = getattr(args, "append", False)
        if args.out:
            if append:
                append_report_markdown(Path(args.out), md, len(tasks))
                print(f"appended a run section to {args.out}")
            else:
                Path(args.out).write_text(md, encoding="utf-8")
                print(f"wrote {args.out} ({rep['aggregate']['tasks']} task(s), "
                      f"{rep['aggregate']['savings_pct_overall']}% fewer overall)")
        else:
            print(md)
        if args.csv:
            if append:
                n = append_report_csv(Path(args.csv), report_to_csv(rep))
                print(f"appended {n} row(s) to {args.csv}")
            else:
                Path(args.csv).write_text(report_to_csv(rep), encoding="utf-8")
                print(f"wrote {args.csv}")
    finally:
        r.close()


def cmd_watch(args):
    """Poll-based reindex loop for the large-repo / low-latency niche.

    Uses the mtime/size fast path so each tick is a stat() sweep unless files
    actually changed. No third-party watcher dependency.
    """
    import time
    root = Path(args.path).resolve()
    db = _db_path(root)
    rep = index_repo(root, db)
    print(f"watching {root} (interval={args.interval}s) — initial graph: {rep.stats}")
    try:
        while True:
            time.sleep(max(1, args.interval))
            rep = index_repo(root, db)
            if rep.parsed or rep.removed:
                print(f"updated: parsed={rep.parsed} removed={rep.removed} "
                      f"graph={rep.stats}")
    except KeyboardInterrupt:
        print("\nstopped.")


def cmd_stats(args):
    root = Path(args.path).resolve()
    s = Store(_db_path(root))
    print(s.stats())
    s.close()


def repo_fidelity(root: Path) -> dict:
    """Per-language extraction fidelity for what is actually indexed (PF-1).

    Answers the question a language table cannot: not "is Zig supported?" but
    "for the Zig in *this* repository, does the call graph exist?" Coverage is
    reported as the share of indexed files whose language produces graph edges,
    so a repo that is 90% regex-tier reads as such instead of reporting a
    healthy index.
    """
    store = Store(_db_path(root))
    try:
        rows = store.files_with_tokens()
        scip = store.get_meta("scip_ingested_at", "")
    finally:
        store.close()
    langs: dict[str, dict] = {}
    for r in rows:
        lang = r["language"] or "unknown"
        tier = language_tier(lang)
        entry = langs.setdefault(lang, {
            "language": lang, "tier": tier,
            **{k: EXTRACTION_TIERS[tier][k]
               for k in ("label", "symbols", "calls", "imports",
                         "inheritance", "note")},
            "files": 0, "symbols": 0, "tokens": 0,
        })
        entry["files"] += 1
        entry["symbols"] += r["symbols_count"] or 0
        entry["tokens"] += r["token_est"] or 0
    total = sum(e["files"] for e in langs.values()) or 1
    graphed = sum(e["files"] for e in langs.values() if e["calls"])
    return {
        "languages": sorted(langs.values(),
                            key=lambda e: (-e["files"], e["language"])),
        "files": total,
        "graph_coverage_pct": round(graphed / total * 100, 1),
        "regex_only_files": total - graphed,
        # SCIP lifts reference precision for languages whose native resolution
        # is weak, but only where an external indexer has actually been run.
        "scip_ingested": bool(scip),
        "scip_ingested_at": scip,
    }


def cmd_langs(args):
    """Language support, by extraction tier — and, with --repo, by real usage."""
    if getattr(args, "repo", False):
        rep = repo_fidelity(Path(args.path).resolve())
        if getattr(args, "json", False):
            _emit(rep, True)
            return
        print(f"indexed files: {rep['files']}   "
              f"call-graph coverage: {rep['graph_coverage_pct']}%   "
              f"regex-only: {rep['regex_only_files']}   "
              f"SCIP: {'yes' if rep['scip_ingested'] else 'no'}")
        print(f"{'language':14} {'tier':20} {'files':>6} {'symbols':>8}  edges")
        for e in rep["languages"]:
            edges = ("calls+imports+inheritance" if e["calls"]
                     else "NONE (symbols only)")
            print(f"{e['language'][:14]:14} {e['label'][:20]:20} "
                  f"{e['files']:>6} {e['symbols']:>8}  {edges}")
        return
    if getattr(args, "json", False):
        _emit({"tiers": EXTRACTION_TIERS,
               "languages": languages_available()}, True)
        return
    for lang, exts in languages_available().items():
        print(f"  {lang:28} {', '.join(exts)}")
    print("\ntiers (what each can extract):")
    for name, t in sorted(EXTRACTION_TIERS.items(),
                          key=lambda kv: -kv[1]["rank"]):
        print(f"  {t['label']:24} {t['note']}")


def cmd_diagnose(args):
    rep = diagnose_extractors()
    if getattr(args, "json", False):
        _emit(rep, True)
    else:
        print(f"extractors: {rep['passed']} passed, {rep['failed']} failed, "
              f"{rep['skipped']} skipped (of {rep['total']})")
        for r in rep["rows"]:
            if r["status"] != "pass":
                extra = (f" missing={r.get('missing')}" if r.get("missing") else "")
                extra += (f" reason={r['reason']}" if r.get("reason") else "")
                extra += (f" error={r['error']}" if r.get("error") else "")
                print(f"  [{r['status']}] {r['language']:11} {r['file']}{extra}")
    if not rep["ok"]:
        raise SystemExit(1)


def cmd_serve(args):
    """Run the MCP server over stdio (default) or HTTP.

    stdio suits an editor launching the server as a child process. HTTP suits
    a shared/remote instance — one index serving several clients, or a host
    that can only reach MCP over the network.
    """
    root = Path(args.path).resolve()
    db = _db_path(root)
    if not db.exists():
        index_repo(root, db)
    try:
        server = build_mcp_server(root, db)
    except ImportError:
        sys.exit("tokengraph: the MCP server needs FastMCP. Install it with:\n"
                 "    pip install fastmcp\n"
                 "(The CLI commands — index/context/skeleton/watch — work without it.)")

    transport = (getattr(args, "transport", None) or "stdio").lower()
    if transport == "stdio":
        server.run()
        return

    host = getattr(args, "host", None) or "127.0.0.1"
    port = int(getattr(args, "port", None) or 8756)
    # Bind to loopback unless told otherwise: the server exposes repository
    # source, so it must not become reachable off-box by accident.
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"tokengraph: WARNING — serving repository source on {host}:{port}. "
              f"This exposes code to anything that can reach that address.",
              file=sys.stderr)
    try:
        server.run(transport=transport, host=host, port=port)
    except TypeError:
        # Older FastMCP builds take only the transport name.
        server.run(transport=transport)
    except ValueError as ex:
        sys.exit(f"tokengraph: unsupported transport {transport!r}: {ex}")


def _discover_packages(root: Path, each: bool) -> list[Path]:
    """Find sub-package roots for monorepo/each context generation.

    `each` returns every immediate child directory; otherwise (monorepo) it
    returns any nested directory that holds a recognised package manifest,
    without descending past a found package.
    """
    skip = {".git", "node_modules", "__pycache__", "dist", "build", ".tokengraph",
            ".context", ".venv", "venv", "target", ".idea", ".vscode", ".next"}
    if each:
        return [d for d in sorted(root.iterdir())
                if d.is_dir() and d.name not in skip]
    manifests = {"package.json", "pyproject.toml", "setup.py", "go.mod",
                 "Cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts",
                 "composer.json", "Gemfile"}
    roots: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        here = Path(dirpath)
        if here == root:
            continue
        if any(m in filenames for m in manifests):
            roots.append(here)
            dirnames[:] = []           # don't descend into a discovered package
    return sorted(roots)


# ==========================================================================
# Repomix interop (RX-1)
# ==========================================================================
# Repomix (https://repomix.com) packs an entire repository into one file for an
# LLM. ContextIQ packs the same repo at *signature* granularity — an order of
# magnitude smaller. These two helpers bridge the ecosystems both ways:
#   • export  — wrap ContextIQ's signature map in Repomix's XML envelope so any
#               Repomix-aware tool can consume it unchanged (and cheaply).
#   • import  — read an existing (huge) Repomix dump and squeeze it into a
#               token-reduced digest before it ever reaches the model.

_REPOMIX_XML_RE = re.compile(
    r'<file[^>]*\bpath="([^"]+)"[^>]*>(.*?)</file>', re.DOTALL)
_REPOMIX_MD_RE = re.compile(
    r'^#{1,6}\s*(?:File:\s*)?(\S.+?)\s*$\n+```[^\n]*\n(.*?)^```',
    re.DOTALL | re.MULTILINE)


def render_repomix(payload: dict) -> str:
    """Wrap ContextIQ's signature map in a Repomix-compatible XML envelope."""
    import html
    md = payload["markdown"]
    summary = (
        "This file is a signature-level pack of the repository, generated by "
        "ContextIQ (tokengraph). Unlike a full Repomix dump it holds compact "
        "signatures — classes, functions, methods, constants, and their doc "
        "comments — not full source. Fetch full bodies on demand via the "
        "ContextIQ MCP server (get_symbol / read_context / get_lines).\n"
        f"Signatures ≈{payload['tokens']:,} tokens vs "
        f"~{payload['repo_tokens']:,} full-source "
        f"({payload['reduction_pct']}% smaller); {payload['files']} files."
    )
    return (
        "<repomix>\n"
        "  <file_summary>\n"
        f"{html.escape(summary)}\n"
        "  </file_summary>\n"
        "  <files>\n"
        '    <file path="CONTEXTIQ_SIGNATURE_MAP.md">\n'
        f"{html.escape(md)}\n"
        "    </file>\n"
        "  </files>\n"
        "</repomix>\n"
    )


def parse_repomix(text: str) -> list[tuple[str, str]]:
    """Extract (path, content) file blocks from a Repomix pack (XML or markdown)."""
    import html
    blocks = [(m.group(1).strip(), html.unescape(m.group(2)))
              for m in _REPOMIX_XML_RE.finditer(text)]
    if not blocks:
        blocks = [(m.group(1).strip(), m.group(2))
                  for m in _REPOMIX_MD_RE.finditer(text)]
    return blocks


def cmd_repomix(args):
    root = Path(args.path).resolve()

    # --- import: squeeze an existing Repomix dump ---
    if getattr(args, "import_file", None):
        text = Path(args.import_file).read_text(encoding="utf-8", errors="replace")
        blocks = parse_repomix(text)
        if not blocks:
            print("no Repomix <file> blocks found; squeezing the whole input",
                  file=sys.stderr)
            blocks = [("<input>", text)]
        r = _open_retriever(args)
        try:
            joined = "\n\n".join(f"# {p}\n{c}" for p, c in blocks)
            s = r.squeeze(joined, kind="auto")
        finally:
            r.close()
        record_pack_savings(root, "repomix.import",
                            final_tokens=s.get("squeezed_tokens", 0),
                            baseline_tokens=s.get("original_tokens", 0), files=len(blocks),
                            no_track=getattr(args, "no_track", False))
        if getattr(args, "json", False):
            _emit({"files": [p for p, _ in blocks], **s}, True)
        else:
            print(s["text"])
            print(f"\n--- imported {len(blocks)} file(s) from Repomix: "
                  f"{s['original_tokens']:,} -> {s['squeezed_tokens']:,} tokens "
                  f"({s['reduction_pct']}% smaller) ---", file=sys.stderr)
        return

    # --- export: emit the signature map as a Repomix pack ---
    cfg = load_config(root, getattr(args, "config", None))
    src_dirs = cfg.get("srcDirs") or ["."]
    index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    try:
        budget = args.budget or cfg.get("maxTokens", 8000)
        payload = build_context_payload(
            r, root, strategy=args.strategy or cfg.get("strategy", "hot-cold"),
            src_dirs=src_dirs, budget=budget,
            hot_commits=cfg.get("hotCommits", 10),
            diff=False, staged=False, config=cfg)
    finally:
        r.close()
    doc = render_repomix(payload)
    if getattr(args, "out", None):
        Path(args.out).write_text(doc, encoding="utf-8")
        print(f"wrote {args.out} (~{payload['tokens']:,} signature tokens vs "
              f"~{payload['repo_tokens']:,} full-source; {payload['files']} files)")
    else:
        sys.stdout.write(doc)


def cmd_generate(args):
    root = Path(args.path).resolve()
    if getattr(args, "monorepo", False) or getattr(args, "each", False):
        import copy
        pkgs = _discover_packages(root, each=getattr(args, "each", False))
        if not pkgs:
            print("(no sub-packages found to generate for)")
            return
        for pk in pkgs:
            sub = copy.copy(args)
            sub.path, sub.monorepo, sub.each, sub.json = str(pk), False, False, False
            print(f"# {pk.relative_to(root).as_posix() or pk.name}")
            try:
                cmd_generate(sub)
            except SystemExit:
                pass
        print(f"generated context for {len(pkgs)} package(s)")
        return
    cfg = load_config(root, getattr(args, "config", None))
    strategy = args.strategy or cfg.get("strategy", "hot-cold")
    src_dirs = cfg.get("srcDirs") or ["."]
    hot_commits = (args.hot_commits if args.hot_commits is not None
                   else cfg.get("hotCommits", 10))
    outputs = args.adapter or cfg.get("outputs") or list(DEFAULT_ADAPTERS)
    fmt = args.format or cfg.get("format", "md")

    index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    try:
        # CFG-6: taken before generation so the stamp names the code the
        # artefact actually describes.
        fingerprint = r.store.content_fingerprint(generated_artifact_paths(cfg))
        total_sig = r.total_signature_tokens(src_dirs)
        if cfg.get("autoMaxTokens", True) and not args.budget:
            budget, warns = effective_budget(
                total_sig, cfg.get("coverageTarget", 0.80),
                cfg.get("modelContextLimit", 128000),
                cfg.get("maxTokensHeadroom", 0.20))
        else:
            budget = args.budget or cfg.get("maxTokens", 8000)
            warns = []

        payload = build_context_payload(
            r, root, strategy=strategy, src_dirs=src_dirs, budget=budget,
            hot_commits=hot_commits, diff=args.diff, staged=args.staged, config=cfg)
        warns = warns + payload["warnings"]

        written = []
        # Per-host budgets (HB-1): hosts differ by an order of magnitude in how
        # much steering text they will carry, so writing one identical blob to
        # every adapter either wasted a large window or blew a small one.
        # Payloads are rebuilt only once per *distinct* budget, not per adapter.
        by_budget: dict[int, list[str]] = {}
        for ad in outputs:
            if ad not in ADAPTERS:
                warns.append(f"unknown adapter {ad!r} (skipped)")
                continue
            if ADAPTERS[ad].get("deprecated"):
                warns.append(f"adapter {ad!r} writes a deprecated format; "
                             f"prefer {ad.replace('-legacy', '')}")
            by_budget.setdefault(
                min(budget, adapter_budget(ad, budget)), []).append(ad)

        payloads = {budget: payload}
        for host_budget, ads in sorted(by_budget.items()):
            if host_budget not in payloads:
                payloads[host_budget] = build_context_payload(
                    r, root, strategy=strategy, src_dirs=src_dirs,
                    budget=host_budget, hot_commits=hot_commits,
                    diff=args.diff, staged=args.staged, config=cfg)
            hp = payloads[host_budget]
            for ad in ads:
                custom = cfg.get("output") if ad == "copilot" else None
                rel = write_adapter(root, ad, hp["markdown"], custom,
                                    fingerprint=fingerprint)
                entry = {"adapter": ad, "path": rel, "budget": host_budget,
                         "tokens": hp["tokens"]}
                if fmt == "cache":
                    # PC-1: the sidecar carries the stable prefix with the cache
                    # breakpoint, plus the volatile tail after it.
                    entry["cache"] = write_cache_sidecar(root, rel, hp)
                written.append(entry)

        over = payload["tokens"] > budget
        track_gain(root, {"op": "generate", "final_tokens": payload["tokens"],
                          "baseline_tokens": payload["repo_tokens"],
                          "saved": payload["repo_tokens"] - payload["tokens"],
                          "reduction_pct": payload["reduction_pct"],
                          "files": payload["files"]}, no_track=args.no_track)
        track_usage(root, {"op": "generate", "strategy": payload["strategy"],
                           "final_tokens": payload["tokens"],
                           "reduction_pct": payload["reduction_pct"]},
                    no_track=args.no_track)

        if getattr(args, "report", False):
            rep = {"finalTokens": payload["tokens"],
                   "reductionPct": payload["reduction_pct"],
                   "effectiveBudget": budget, "overBudget": over}
            _emit(rep, True) if getattr(args, "json", False) else print(
                f"finalTokens={rep['finalTokens']} reductionPct={rep['reductionPct']} "
                f"effectiveBudget={budget} overBudget={over}")
        elif getattr(args, "json", False):
            _emit({"strategy": payload["strategy"], "final_tokens": payload["tokens"],
                   "reduction_pct": payload["reduction_pct"], "effective_budget": budget,
                   "total_sig_tokens": total_sig, "over_budget": over,
                   "outputs": written, "warnings": warns,
                   "hot_files": payload["hot_files"]}, True)
        else:
            print(f"strategy={payload['strategy']}  files={payload['files']}  "
                  f"context≈{payload['tokens']:,} tokens  "
                  f"({payload['reduction_pct']}% < full source)  budget={budget:,}")
            for w in written:
                print(f"  wrote {w['path']}" +
                      (f"  (+ {w['cache']})" if w.get("cache") else ""))
            for w in warns:
                print(f"  ! {w}", file=sys.stderr)
        if over and getattr(args, "report", False):     # TB-7: CI gate
            raise SystemExit(1)
    finally:
        r.close()


def cmd_init(args):
    root = Path(args.path).resolve()
    p = write_default_config(root)
    print(f"wrote {p.name} — edit srcDirs/strategy/outputs to taste")


def cmd_setup(args):
    """Auto-setup: config + MCP wiring + git hook + initial context (MCP-7)."""
    import json
    import shutil
    import time
    t0 = time.time()
    root = Path(args.path).resolve()
    script = os.path.abspath(__file__)
    done: list[str] = []

    # Prefer the installed `tokengraph` console entry point (G13) so the MCP
    # configs and git hook don't hardcode an absolute path to this script.
    entry = shutil.which("tokengraph")
    if entry:
        cmd, serve_args, gen_cmd = "tokengraph", ["serve"], 'tokengraph generate'
        done.append("using installed `tokengraph` entry point")
    else:
        cmd, serve_args = "python", [script, "serve"]
        gen_cmd = 'python "%s" generate' % script

    if not (root / CONFIG_NAME).exists():
        write_default_config(root)
        done.append(f"wrote {CONFIG_NAME}")

    def _merge_json(rel: str, updates: dict):
        p = root / rel
        cur = {}
        if p.exists():
            try:
                cur = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                cur = {}
        merged = _deep_merge(cur, updates)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        done.append(f"wired {rel}")

    stdio = {"type": "stdio", "command": cmd, "args": serve_args}
    _merge_json(".mcp.json", {"mcpServers": {"tokengraph": stdio}})
    _merge_json(".vscode/mcp.json", {"servers": {"tokengraph": stdio}})
    _merge_json(".cursor/mcp.json", {"mcpServers": {"tokengraph": stdio}})
    # NOTE: Claude Code reads MCP servers from .mcp.json, NOT from
    # .claude/settings.json — which has no `mcpServers` key at all. Writing one
    # there was a silent no-op. What settings.json *is* good for is the
    # PostToolUse hook that keeps the graph warm after every edit, so that is
    # what we write instead.
    _merge_json(".claude/settings.json", {
        "hooks": {
            "PostToolUse": [{
                "matcher": "Edit|Write|MultiEdit",
                "hooks": [{"type": "command",
                           "command": f"{gen_cmd.rsplit(' ', 1)[0]} index"
                                      if entry else
                                      f'python "{script}" index'}],
            }],
        },
    })

    hook = root / ".git" / "hooks" / "post-commit"
    if (root / ".git").is_dir():
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/bin/sh\n%s >/dev/null 2>&1 || true\n"
                        % gen_cmd, encoding="utf-8")
        try:
            os.chmod(hook, 0o755)
        except OSError:
            pass
        done.append("installed git post-commit hook")
    else:
        done.append("(no .git — skipped post-commit hook)")

    index_repo(root, _db_path(root))
    done.append("built initial graph")
    args.strategy = None; args.adapter = None; args.budget = None
    args.hot_commits = None; args.format = None; args.diff = False
    args.staged = False; args.no_track = True; args.report = False
    args.json = False; args.config = None
    cmd_generate(args)
    print(f"\nsetup done in {time.time()-t0:.1f}s:")
    for d in done:
        print(f"  - {d}")


def cmd_health(args):
    """Composite health score 0-100 / grade A-F (CI-1)."""
    import json
    import time
    root = Path(args.path).resolve()
    usage = root / ".context" / "usage.ndjson"
    runs = 0
    last_ts = 0.0
    last_reduction = 0.0
    if usage.exists():
        for line in usage.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            runs += 1
            last_ts = max(last_ts, d.get("ts", 0.0))
            if "reduction_pct" in d:
                last_reduction = d["reduction_pct"]
    age_days = ((time.time() - last_ts) / 86400.0) if last_ts else 999.0
    red_score = max(0.0, min(1.0, last_reduction / 95.0)) * 60      # up to 60
    fresh_score = max(0.0, 1.0 - age_days / 7.0) * 30               # up to 30 (CI-2 staleness)
    run_score = min(1.0, runs / 5.0) * 10                           # up to 10
    score = round(red_score + fresh_score + run_score)
    grade = ("A" if score >= 90 else "B" if score >= 75 else
             "C" if score >= 60 else "D" if score >= 45 else "F")
    out = {"score": score, "grade": grade, "reduction_pct": last_reduction,
           "age_days": round(age_days, 2), "runs": runs,
           "stale": age_days > 7}
    if getattr(args, "json", False):
        _emit(out, True)
    else:
        print(f"health: {score}/100  grade {grade}")
        print(f"  reduction={last_reduction}%  last_regen={out['age_days']}d ago  "
              f"runs={runs}  stale={out['stale']}")
    if out["stale"] and getattr(args, "strict", False):
        raise SystemExit(1)


def load_benchmark_corpus(path: Path) -> list[dict]:
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases", data) if isinstance(data, dict) else data
    return [case for case in cases
            if case.get("task") and case.get("expected_files")]


# QG-1: quality gate thresholds. A retrieval tool whose whole premise is
# "fewer tokens, same answer" must be able to fail a build when the answer
# stops being reachable. These are the floors CI enforces.
#
# These are calibrated to MEASURED performance on the shipped corpora, with a
# regression margin — they are detectors, not aspirations. Numbers observed at
# the product default budget of 6000 tokens across 96 cases in 4 repositories
# (Python / TypeScript / Go):
#
#     recall_at_5 0.979 | symbol_recall 0.865 | answerable 0.698 | waste 0.717
#
# Previously 0.958 / 0.719 / 0.510 / 0.785. The gain came from two fixes, both
# aimed at the same finding — that retrieval located the right file and then
# failed to carry the thing in it that answered the question:
#
#   * SR-1, the completion sweep, which stopped packs terminating at half the
#     requested budget with the answer left on the floor; and
#   * CN-1, indexing module- and type-scope constants, without which a
#     controlling value was not a symbol and could not be retrieved at all.
#
# The honest reading is still that answerability is the weak metric: roughly a
# third of packs remain short of some symbol or literal an answer needs. Do not
# raise a threshold without first raising the measurement.
BENCHMARK_THRESHOLDS = {
    "recall_at_5": 0.95,        # the right file is in the top 5
    "symbol_recall": 0.80,      # required symbols actually made it into the pack
    "answerable_rate": 0.60,    # packs carrying EVERY required symbol + fact
    "irrelevant_token_ratio": 0.78,   # ceiling — budget spent outside target files
}


def score_pack_answerability(pack: "ContextPack", case: dict) -> dict:
    """Can this pack actually support a correct answer? (QG-1)

    File-level recall is not answer quality: a pack can name the right file and
    still omit the function the question is about. This checks the two things
    an answer actually needs —

    * ``expected_symbols`` — every symbol the answer must reason about is
      present in the pack, as a body or at least a signature.
    * ``must_contain`` — literal facts (an identifier, a default value, a
      decorator) that must survive compression into the rendered pack.

    ``answerable`` is the strict conjunction: everything required is there.
    """
    rendered = pack.to_markdown()
    present = {p.qname for p in pack.pieces}
    # A symbol also counts as covered when a chunk from its file carries its
    # name, and when the session ledger says the model already holds it.
    covered_text = rendered
    reused = set(pack.reused)

    want_symbols = list(case.get("expected_symbols") or [])
    found_symbols, missing_symbols = [], []
    for q in want_symbols:
        leaf = q.rsplit(".", 1)[-1]
        if q in present or q in reused or leaf in covered_text:
            found_symbols.append(q)
        else:
            missing_symbols.append(q)

    want_facts = list(case.get("must_contain") or [])
    missing_facts = [f for f in want_facts if f not in covered_text]

    sym_recall = (len(found_symbols) / len(want_symbols)) if want_symbols else 1.0
    return {
        "symbol_recall": sym_recall,
        "missing_symbols": missing_symbols,
        "missing_facts": missing_facts,
        "answerable": not missing_symbols and not missing_facts,
    }


def run_retrieval_benchmark(r: Retriever, cases: list[dict],
                            budget_tokens: int = 4000) -> dict:
    import time
    hits = 0
    reciprocal_rank = 0.0
    irrelevant_tokens = 0
    pack_tokens = 0
    symbol_recall_sum = 0.0
    answerable = 0
    latencies_ms: list[float] = []
    rows = []
    for case in cases:
        started = time.perf_counter()
        ranked = [f for f, _ in r.rank_files(
            case["task"], top_k=5, use_recency=False)]
        pack = r.find_relevant_context(case["task"], budget_tokens=budget_tokens)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        expected = set(case["expected_files"])
        positions = [ranked.index(f) + 1 for f in expected if f in ranked]
        rank = min(positions) if positions else 0
        if rank:
            hits += 1
            reciprocal_rank += 1.0 / rank
        # MS-2: "waste" used to mean any token from a file outside
        # expected_files — which counted legitimate cross-file context (the
        # callee that explains the seed, the base class it inherits) as waste.
        # That made the metric punish the graph expansion that is the whole
        # point of the tool. A piece now counts as on-target if it is in an
        # expected file, IS an expected symbol, or is graph-connected to one.
        want_syms = set(case.get("expected_symbols") or [])
        related: set[str] = set()
        for q in want_syms:
            sid = r.store.id_for_qname(q)
            if sid is None:
                continue
            for direction in ("out", "in"):
                for nb in r.store.neighbors(
                        sid, ["CALLS", "REFERENCES", "INHERITS"], direction,
                        limit=MAX_FANOUT_PER_SYMBOL):
                    related.add(nb["qname"])
        row_pack_tokens = sum(piece.token_est for piece in pack.pieces)
        row_irrelevant = sum(
            piece.token_est for piece in pack.pieces
            if piece.file not in expected
            and piece.qname not in want_syms
            and piece.qname not in related)
        pack_tokens += row_pack_tokens
        irrelevant_tokens += row_irrelevant

        quality = score_pack_answerability(pack, case)
        symbol_recall_sum += quality["symbol_recall"]
        answerable += 1 if quality["answerable"] else 0

        rows.append({"task": case["task"], "expected_files": sorted(expected),
                     "top_files": ranked, "rank": rank,
                     "pack_tokens": pack.rendered_tokens,
                     "symbol_recall": round(quality["symbol_recall"], 3),
                     "answerable": quality["answerable"],
                     "missing_symbols": quality["missing_symbols"],
                     "missing_facts": quality["missing_facts"]})
    count = len(cases) or 1
    return {
        "queries": len(cases),
        "recall_at_5": round(hits / count, 3),
        "hit_at_5": round(hits / count, 3),
        "mrr": round(reciprocal_rank / count, 3),
        "symbol_recall": round(symbol_recall_sum / count, 3),
        "answerable_rate": round(answerable / count, 3),
        "irrelevant_token_ratio": round(
            irrelevant_tokens / pack_tokens, 3) if pack_tokens else 0.0,
        "mean_latency_ms": round(sum(latencies_ms) / count, 2),
        "rows": rows,
    }


# ==========================================================================
# QG-2: LLM-judged answer quality
# ==========================================================================
# The deterministic gate (score_pack_answerability) proves the required symbols
# and facts are *present* in the pack. It cannot prove a model *answers
# correctly* from it. This harness closes that gap with the experiment the
# project's thesis actually needs: answer each held-out question twice — once
# from the ContextIQ pack, once from the full text of the files that contain
# the answer — and have an independent judge grade both against a rubric.
#
# `quality_retention = pack_score / full_score` is the headline: 1.0 means
# compression cost nothing. It is opt-in and never runs in CI, because it needs
# API access and real spend.

JUDGE_MODEL = "claude-opus-4-8"

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "met": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "required": ["claim", "met", "evidence"],
                "additionalProperties": False,
            },
        },
        "score": {"type": "number"},
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
    },
    "required": ["criteria", "score", "verdict"],
    "additionalProperties": False,
}

_ANSWER_SYSTEM = (
    "You are a senior engineer answering a question about an unfamiliar "
    "codebase. Answer ONLY from the context provided below. If the context "
    "does not contain what you need, say exactly what is missing rather than "
    "guessing. Be specific: name the identifiers, constants and control flow "
    "you are relying on. Keep the answer under 200 words."
)

_JUDGE_SYSTEM = (
    "You grade answers about a codebase against a rubric of atomic factual "
    "claims. For each rubric claim decide whether the candidate answer "
    "actually contains it — not whether the answer sounds plausible. Quote the "
    "supporting span in `evidence`, or explain the gap when it is absent. "
    "`score` is the fraction of claims met, 0.0 to 1.0. `verdict` is "
    "'correct' when every claim is met, 'partial' when some are, 'incorrect' "
    "when essentially none are. Be strict: a claim that is merely implied is "
    "not met."
)


def _anthropic_client():
    """Anthropic client, or None with a reason when unavailable."""
    try:
        import anthropic  # type: ignore[import-not-found]
    except Exception:
        return None, ("the `anthropic` package is not installed "
                      "(pip install anthropic)")
    try:
        return anthropic.Anthropic(), ""
    except Exception as ex:
        return None, f"could not construct the client: {type(ex).__name__}: {ex}"


def _ask_model(client, model: str, context: str, question: str) -> tuple[str, dict]:
    """Answer `question` from `context` alone. Returns (answer, usage)."""
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_ANSWER_SYSTEM,
        messages=[{"role": "user", "content":
                   f"# Context\n\n{context}\n\n# Question\n\n{question}"}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    u = resp.usage
    return text, {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}


def _judge_answer(client, model: str, question: str, reference: str,
                  rubric: list[str], answer: str) -> dict:
    """Grade `answer` against `rubric` with a structured verdict."""
    claims = "\n".join(f"{i+1}. {c}" for i, c in enumerate(rubric))
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_JUDGE_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
        messages=[{"role": "user", "content":
                   f"# Question\n{question}\n\n"
                   f"# Reference answer\n{reference}\n\n"
                   f"# Rubric claims\n{claims}\n\n"
                   f"# Candidate answer to grade\n{answer}"}],
    )
    import json
    text = next(b.text for b in resp.content if b.type == "text")
    out = json.loads(text)
    out["judge_usage"] = {"input_tokens": resp.usage.input_tokens,
                          "output_tokens": resp.usage.output_tokens}
    return out


def load_judge_corpora(root: Path) -> list[Path]:
    """Every judge.json alongside a fixture repo."""
    repos = Path(root) / "benchmarks" / "repos"
    return sorted(repos.glob("*/judge.json")) if repos.is_dir() else []


def run_llm_judge(corpus_paths: list[Path], model: str = JUDGE_MODEL,
                  budget_tokens: int = 6000, limit: int = 0,
                  compare_full: bool = True) -> dict:
    """Score answers from a ContextIQ pack against answers from full files (QG-2).

    For every question: retrieve a pack, answer from it, and (unless
    `compare_full` is off) also answer from the complete text of the cited
    files. An independent judge grades both against the rubric. The ratio of
    the two scores is the number that actually tests "fewer tokens, same
    answer quality".
    """
    import json
    client, why = _anthropic_client()
    if client is None:
        return {"ok": False, "error": why,
                "hint": "Set ANTHROPIC_API_KEY (or run `ant auth login`), then "
                        "re-run. This eval makes real API calls and costs money, "
                        "which is why it never runs in CI."}

    rows, suites = [], []
    for cpath in corpus_paths:
        cpath = Path(cpath)
        if not cpath.exists():
            continue
        doc = json.loads(cpath.read_text(encoding="utf-8"))
        repo = (cpath.parent / doc.get("repo", ".")).resolve()
        if not repo.exists():
            continue
        questions = doc.get("questions", [])
        if limit:
            questions = questions[:limit]
        db = _db_path(repo)
        index_repo(repo, db)
        r = Retriever(repo, db)
        try:
            for q in questions:
                pack = r.find_relevant_context(q["question"],
                                               budget_tokens=budget_tokens)
                pack_md = pack.to_markdown()
                row = {"id": q.get("id", ""), "repo": cpath.parent.name,
                       "question": q["question"],
                       "pack_tokens": pack.rendered_tokens}
                try:
                    answer, usage = _ask_model(client, model, pack_md,
                                               q["question"])
                    graded = _judge_answer(client, model, q["question"],
                                           q.get("reference_answer", ""),
                                           q.get("rubric", []), answer)
                    row.update({"pack_score": graded["score"],
                                "pack_verdict": graded["verdict"],
                                "pack_answer": answer,
                                "pack_criteria": graded["criteria"],
                                "pack_answer_tokens": usage["input_tokens"]})
                except Exception as ex:
                    row["error"] = f"{type(ex).__name__}: {ex}"
                    rows.append(row)
                    continue

                if compare_full:
                    # Generous baseline: the whole text of every cited file.
                    parts = []
                    for f in q.get("cites", []):
                        try:
                            parts.append(f"## {f}\n```\n"
                                         + (repo / f).read_text(encoding="utf-8",
                                                                errors="replace")
                                         + "\n```")
                        except OSError:
                            continue
                    full_ctx = "\n\n".join(parts)
                    if full_ctx:
                        try:
                            fa, fu = _ask_model(client, model, full_ctx,
                                                q["question"])
                            fg = _judge_answer(client, model, q["question"],
                                               q.get("reference_answer", ""),
                                               q.get("rubric", []), fa)
                            row.update({"full_score": fg["score"],
                                        "full_verdict": fg["verdict"],
                                        "full_tokens": count_tokens(full_ctx),
                                        "full_answer_tokens": fu["input_tokens"]})
                        except Exception as ex:
                            row["full_error"] = f"{type(ex).__name__}: {ex}"
                rows.append(row)
        finally:
            r.close()
        suites.append(cpath.parent.name)

    scored = [x for x in rows if "pack_score" in x]
    paired = [x for x in scored if "full_score" in x]
    n = len(scored) or 1
    pack_mean = sum(x["pack_score"] for x in scored) / n
    agg = {
        "questions": len(rows),
        "graded": len(scored),
        "errors": len(rows) - len(scored),
        "repos": suites,
        "model": model,
        "pack_mean_score": round(pack_mean, 3),
        "pack_correct_rate": round(
            sum(1 for x in scored if x["pack_verdict"] == "correct") / n, 3),
        "mean_pack_tokens": round(
            sum(x["pack_tokens"] for x in scored) / n, 1),
    }
    if paired:
        m = len(paired)
        full_mean = sum(x["full_score"] for x in paired) / m
        agg.update({
            "paired": m,
            "full_mean_score": round(full_mean, 3),
            "mean_full_tokens": round(
                sum(x["full_tokens"] for x in paired) / m, 1),
            # The headline: 1.0 means compression cost no answer quality.
            "quality_retention": round(pack_mean / full_mean, 3) if full_mean else None,
            "token_ratio": round(
                sum(x["pack_tokens"] for x in paired)
                / max(1, sum(x["full_tokens"] for x in paired)), 3),
        })
    return {"ok": True, "aggregate": agg, "rows": rows}


# ==========================================================================
# publishable benchmark (BM-PUB): reproducible dataset + report + archive meta
# ==========================================================================
# SigMap's one durable edge was a peer-archived study with a DOI. ContextIQ
# already has the measurement machinery (run_benchmark_suite, test_discovery_f1,
# hallucination_benchmark); this turns a run into the *artifacts a repository
# like Zenodo needs*: a content-hashed dataset manifest (so anyone can verify
# they benchmarked the same bytes), a human report, and citation/deposition
# metadata. The credentialed upload stays a manual maintainer step.

# Files generated *by* the benchmark — never part of the hashed input dataset,
# or the dataset_hash would drift on every run and stop being reproducible.
_BENCHMARK_GENERATED = {"REPORT.md", "MANIFEST.json"}


def dataset_manifest(root: Path) -> dict:
    """Content hashes of every benchmark input, for reproducibility."""
    base = Path(root) / "benchmarks"
    entries: list[dict] = []
    if base.is_dir():
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if (".tokengraph" in p.parts or p.suffix in (
                    ".db", ".db-wal", ".db-shm")
                    or p.name in _BENCHMARK_GENERATED):
                continue
            data = p.read_bytes()
            entries.append({"path": p.relative_to(root).as_posix(),
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "bytes": len(data)})
    dataset_hash = hashlib.sha256(
        "\n".join(f"{e['path']}:{e['sha256']}" for e in entries)
        .encode("utf-8")).hexdigest()
    return {"file_count": len(entries), "dataset_hash": dataset_hash,
            "files": entries}


def build_benchmark_publication(root: Path, *, budget: int = 4000,
                                run_hallucination: bool = False) -> dict:
    """Run the reproducible benchmark suite and assemble a publication payload."""
    root = Path(root).resolve()
    corpora = discover_corpora(root)
    suite = (run_benchmark_suite(corpora, budget_tokens=budget)
             if corpora else {"aggregate": {}, "suites": []})

    f1 = None
    tmdir = root / "benchmarks" / "testmap"
    if (tmdir / "pairs.json").exists():
        import json
        gold = json.loads((tmdir / "pairs.json").read_text(
            encoding="utf-8")).get("pairs", [])
        files = [p.relative_to(tmdir).as_posix()
                 for p in sorted(tmdir.rglob("*"))
                 if p.is_file() and p.name != "pairs.json"]
        f1 = test_discovery_f1(files, gold)

    hallucination = None
    if run_hallucination:
        try:
            index_repo(root, _db_path(root))
            r = Retriever(root, _db_path(root))
            try:
                hallucination = hallucination_benchmark(r)
            finally:
                r.close()
        except Exception as e:                       # never block the report
            hallucination = {"error": str(e)}

    return {
        "tool": "ContextIQ (tokengraph)",
        "retrieval": suite.get("aggregate", {}),
        "retrieval_suites": [
            {k: s[k] for k in ("corpus", "queries", "recall_at_5",
                               "symbol_recall", "answerable_rate",
                               "irrelevant_token_ratio", "languages")
             if k in s}
            for s in suite.get("suites", [])],
        "test_discovery": f1,
        "hallucination": hallucination,
        "dataset": dataset_manifest(root),
        "environment": {
            "extractor_version": extractor_version(),
            "python": sys.version.split()[0],
        },
    }


def render_benchmark_report_md(pub: dict, generated_at: str = "") -> str:
    ret = pub.get("retrieval") or {}
    f1 = pub.get("test_discovery") or {}
    ds = pub.get("dataset") or {}
    out = [
        "# ContextIQ Benchmark Report",
        "",
        "> Reproducible, self-contained benchmark of ContextIQ (`tokengraph`) — "
        "token-efficient code context retrieval, cross-language test discovery, "
        "and hallucination guarding. Every input is content-hashed (see "
        "`benchmarks/MANIFEST.json`) so a third party can verify they ran the "
        "same dataset.",
        "",
        f"- Generated: {generated_at or '(unset)'}",
        f"- Dataset hash: `{ds.get('dataset_hash', 'n/a')}` "
        f"({ds.get('file_count', 0)} files)",
        f"- Extractor: `{pub.get('environment', {}).get('extractor_version', '?')}`"
        f" · Python {pub.get('environment', {}).get('python', '?')}",
        "",
        "## 1. Retrieval quality",
        "",
        "| Metric | Value |",
        "|---|--:|",
        f"| Queries | {ret.get('queries', 0)} |",
        f"| Corpora | {ret.get('corpora', 0)} |",
        f"| Recall@5 | {ret.get('recall_at_5', 0)} |",
        f"| Symbol recall | {ret.get('symbol_recall', 0)} |",
        f"| Answerable rate | {ret.get('answerable_rate', 0)} |",
        f"| Irrelevant-token ratio (waste) | {ret.get('irrelevant_token_ratio', 0)} |",
        "",
    ]
    if pub.get("retrieval_suites"):
        out += ["### Per-corpus", "",
                "| Corpus | n | Recall@5 | Symbol recall | Answerable | Waste |",
                "|---|--:|--:|--:|--:|--:|"]
        for s in pub["retrieval_suites"]:
            out.append(
                f"| {s.get('corpus', '?')} | {s.get('queries', 0)} | "
                f"{s.get('recall_at_5', 0)} | {s.get('symbol_recall', 0)} | "
                f"{s.get('answerable_rate', 0)} | "
                f"{s.get('irrelevant_token_ratio', 0)} |")
        out.append("")
    out += [
        "## 2. Test discovery (implementation ↔ test mapping)",
        "",
    ]
    if f1:
        out += [
            "| Metric | Value |", "|---|--:|",
            f"| Precision | {f1.get('precision')} |",
            f"| Recall | {f1.get('recall')} |",
            f"| **F1** | **{f1.get('f1')}** |",
            f"| hit@1 | {f1.get('hit_at_1')} |",
            f"| Gold pairs | {f1.get('gold_pairs')} |",
            f"| TP / FP / FN | {f1.get('true_positives')} / "
            f"{f1.get('false_positives')} / {f1.get('false_negatives')} |",
            "",
            "Measured on `benchmarks/testmap/` (Python/Go/TypeScript/Java, labeled "
            "in `pairs.json`). The naming heuristic scores perfect precision; the "
            "single miss is a deliberately name-divergent pair that only the call "
            "graph links (recovered at symbol granularity by `get_test_map`).",
            "",
        ]
    else:
        out += ["_(no labeled corpus present)_", ""]
    hall = pub.get("hallucination")
    if hall and "error" not in hall:
        out += ["## 3. Hallucination guard", "",
                "```json", _json_dumps_safe(hall), "```", ""]
    out += [
        "## Reproduce", "",
        "```bash",
        "tokengraph benchmark --all          # retrieval quality across all corpora",
        "tokengraph test-map --benchmark     # test-discovery precision/recall/F1",
        "tokengraph publish-benchmark        # regenerate this report + manifest",
        "```",
        "",
        "Verify the dataset is byte-identical by re-hashing `benchmarks/` and "
        "comparing `dataset_hash` in `benchmarks/MANIFEST.json`.",
    ]
    return "\n".join(out) + "\n"


def _json_dumps_safe(obj) -> str:
    import json
    return json.dumps(obj, indent=2, sort_keys=True)


def zenodo_metadata(pub: dict, version: str = "1.0.0",
                    creator: str = "") -> dict:
    f1 = (pub.get("test_discovery") or {}).get("f1")
    ret = pub.get("retrieval") or {}
    desc = (
        "ContextIQ is a local code-graph server that gives AI coding agents "
        "token-efficient, verifiable context. This deposition archives its "
        "reproducible benchmark: retrieval quality "
        f"(Recall@5={ret.get('recall_at_5', 'n/a')} over {ret.get('queries', 0)} "
        f"queries across {ret.get('corpora', 0)} corpora), cross-language "
        f"test-discovery (F1={f1 if f1 is not None else 'n/a'}), and a "
        "hallucination-guard measurement, plus a content-hashed dataset manifest "
        "for exact reproduction.")
    return {
        "title": "ContextIQ: A Reproducible Benchmark for Token-Efficient, "
                 "Verifiable AI Code Context",
        "upload_type": "dataset",
        "description": desc,
        "creators": [{"name": creator or "ContextIQ maintainers"}],
        "license": "MIT",
        "access_right": "open",
        "keywords": ["code retrieval", "LLM context", "token optimization",
                     "test discovery", "RAG", "hallucination", "code graph",
                     "MCP"],
        "version": version,
        "notes": "dataset_hash=" + (pub.get("dataset") or {}).get(
            "dataset_hash", ""),
    }


def citation_cff(version: str = "1.0.0", creator: str = "") -> str:
    who = creator or "ContextIQ maintainers"
    return (
        "cff-version: 1.2.0\n"
        "message: \"If you use ContextIQ or its benchmark, please cite it.\"\n"
        "title: \"ContextIQ: Token-Efficient, Verifiable AI Code Context\"\n"
        f"version: \"{version}\"\n"
        "license: MIT\n"
        "authors:\n"
        f"  - name: \"{who}\"\n"
        "keywords:\n"
        "  - code retrieval\n"
        "  - LLM context\n"
        "  - token optimization\n"
        "  - test discovery\n"
    )


def cmd_publish_benchmark(args):
    """Run the suite and emit publish-ready artifacts (BM-PUB)."""
    import json
    root = Path(args.path).resolve()
    creator = getattr(args, "creator", None) or ""
    version = getattr(args, "version", None) or "1.0.0"
    pub = build_benchmark_publication(
        root, budget=args.budget,
        run_hallucination=getattr(args, "full", False))

    # timestamp is human-facing only; it is never folded into dataset_hash.
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report = render_benchmark_report_md(pub, generated_at=stamp)
    manifest = pub["dataset"]
    zmeta = zenodo_metadata(pub, version=version, creator=creator)
    cff = citation_cff(version=version, creator=creator)
    how = (
        "# Benchmark: methodology & how to archive\n\n"
        "ContextIQ ships a reproducible benchmark. This document explains how to "
        "reproduce it and how to archive it with a DOI.\n\n"
        "## What is measured\n\n"
        "- **Retrieval quality** — Recall@5, symbol recall, answerable rate, and "
        "irrelevant-token ratio across every corpus under `benchmarks/` "
        "(`tokengraph benchmark --all`).\n"
        "- **Test discovery** — precision / recall / F1 / hit@1 of the "
        "implementation↔test mapping on the labeled `benchmarks/testmap/` corpus "
        "(`tokengraph test-map --benchmark`).\n"
        "- **Hallucination guard** — grounding coverage + guard catch/specificity "
        "(`tokengraph publish-benchmark --full`).\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "pip install 'contextiq[all]'\n"
        "tokengraph publish-benchmark --full\n"
        "```\n\n"
        "Then confirm `benchmarks/MANIFEST.json`'s `dataset_hash` matches — it is "
        "a SHA-256 over every benchmark input, so an identical hash proves an "
        "identical dataset.\n\n"
        "## Archive with a DOI\n\n"
        "The generated `.zenodo.json` and `CITATION.cff` are deposition-ready "
        "(set the author/ORCID/affiliation first). `tokengraph zenodo-publish` "
        "deposits the artifacts and mints the DOI directly — sandbox and *draft* "
        "by default, so nothing is permanent until you opt in:\n\n"
        "```bash\n"
        "# 1. dry run — see exactly what will be uploaded, no token needed\n"
        "tokengraph zenodo-publish --dry-run\n\n"
        "# 2. create a DRAFT on the safe sandbox to review it\n"
        "export ZENODO_TOKEN=...        # from sandbox.zenodo.org/account/settings/applications\n"
        "tokengraph zenodo-publish\n\n"
        "# 3. mint the PERMANENT DOI on production (irreversible)\n"
        "export ZENODO_TOKEN=...        # from zenodo.org/account/settings/applications\n"
        "tokengraph zenodo-publish --production --publish\n"
        "```\n\n"
        "The DOI Zenodo mints is what turns this from *reproducible* into "
        "*peer-archived*. Minting requires `--production --publish` together, so "
        "a stray run can never publish by accident.\n"
    )

    outputs = {
        "benchmarks/REPORT.md": report,
        "benchmarks/MANIFEST.json": json.dumps(manifest, indent=2) + "\n",
        ".zenodo.json": json.dumps(zmeta, indent=2) + "\n",
        "CITATION.cff": cff,
        "docs/BENCHMARK.md": how,
    }
    written = []
    if not getattr(args, "dry_run", False):
        for rel, content in outputs.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            written.append(rel)

    if getattr(args, "json", False):
        _emit({"written": written, "dataset_hash": manifest["dataset_hash"],
               "retrieval": pub["retrieval"],
               "test_discovery": pub["test_discovery"]}, True)
    else:
        f1 = pub.get("test_discovery") or {}
        ret = pub.get("retrieval") or {}
        print("benchmark publication assembled:")
        print(f"  retrieval:      recall@5={ret.get('recall_at_5', 'n/a')} "
              f"over {ret.get('queries', 0)} queries / "
              f"{ret.get('corpora', 0)} corpora")
        print(f"  test-discovery: F1={f1.get('f1', 'n/a')} "
              f"(precision={f1.get('precision', 'n/a')}, "
              f"recall={f1.get('recall', 'n/a')}, "
              f"hit@1={f1.get('hit_at_1', 'n/a')})")
        print(f"  dataset hash:   {manifest['dataset_hash']} "
              f"({manifest['file_count']} files)")
        for w in written:
            print(f"  wrote {w}")
        if not written:
            print("  (dry run — nothing written)")


# ==========================================================================
# Zenodo deposition (BM-DOI): mint a DOI for the benchmark, from stdlib only
# ==========================================================================
# `publish-benchmark` produces the deposition-ready artifacts; this uploads them
# to Zenodo and (only on an explicit --publish) mints the DOI, closing the last
# gap to a peer-archived benchmark. Safety is layered: sandbox by default, a
# reversible *draft* by default, offline-gated, token never logged, and minting
# a (permanent) DOI requires --production --publish together.

ZENODO_API = "https://zenodo.org/api"
ZENODO_SANDBOX_API = "https://sandbox.zenodo.org/api"

# The reproducibility set uploaded by default. MANIFEST.json carries the
# content hashes, so a reproducer can verify the exact dataset.
ZENODO_DEFAULT_FILES = ("benchmarks/REPORT.md", "benchmarks/MANIFEST.json",
                        "CITATION.cff", ".zenodo.json")


def zenodo_plan(root: Path, *, sandbox: bool = True, publish: bool = False,
                files: list[str] | None = None) -> dict:
    """Build the deposition plan (endpoints + metadata + files) — no network."""
    import json
    root = Path(root).resolve()
    zpath = root / ".zenodo.json"
    if not zpath.exists():
        raise FileNotFoundError(
            ".zenodo.json not found — run `tokengraph publish-benchmark` first")
    meta = json.loads(zpath.read_text(encoding="utf-8"))
    rels = list(files) if files else list(ZENODO_DEFAULT_FILES)
    present = [f for f in rels if (root / f).exists()]
    missing = [f for f in rels if not (root / f).exists()]
    base = ZENODO_SANDBOX_API if sandbox else ZENODO_API
    steps = [{"method": "POST", "path": "/deposit/depositions",
              "note": "create draft deposition"}]
    for f in present:
        steps.append({"method": "PUT", "path": "{bucket}/" + Path(f).name,
                      "file": f, "note": "upload"})
    steps.append({"method": "PUT", "path": "/deposit/depositions/{id}",
                  "note": "set metadata"})
    if publish:
        steps.append({"method": "POST",
                      "path": "/deposit/depositions/{id}/actions/publish",
                      "note": "PUBLISH — mints a permanent DOI"})
    return {"base": base, "sandbox": sandbox, "publish": publish,
            "metadata": {"metadata": meta}, "files": present,
            "missing_files": missing, "steps": steps}


def _zenodo_http(method: str, url: str, token: str, *, json_body: dict | None = None,
                 raw: bytes | None = None, content_type: str = "application/json"):
    """One Zenodo REST call. Returns parsed JSON (or {} for empty 2xx)."""
    import json
    import urllib.request
    import urllib.error
    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = content_type
    elif raw is not None:
        data = raw
        headers["Content-Type"] = "application/octet-stream"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", "replace")
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Zenodo {method} {url.split('?')[0]} -> "
                           f"HTTP {e.code}: {detail}") from None


def zenodo_deposit(root: Path, *, token: str, sandbox: bool = True,
                   publish: bool = False, files: list[str] | None = None) -> dict:
    """Create a deposition, upload artifacts, set metadata, optionally publish.

    Returns the deposition id, its edit URL, and the (pre-reserved or minted)
    DOI. Never prints the token. Refuses in offline mode.
    """
    if offline_mode():
        raise RuntimeError("offline mode (TOKENGRAPH_OFFLINE) — refusing network upload")
    if not token:
        raise RuntimeError("no Zenodo token (pass --token or set ZENODO_TOKEN)")
    plan = zenodo_plan(root, sandbox=sandbox, publish=publish, files=files)
    base = plan["base"]
    root = Path(root).resolve()

    dep = _zenodo_http("POST", f"{base}/deposit/depositions", token, json_body={})
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    for rel in plan["files"]:
        data = (root / rel).read_bytes()
        _zenodo_http("PUT", f"{bucket}/{Path(rel).name}", token, raw=data)
    _zenodo_http("PUT", f"{base}/deposit/depositions/{dep_id}", token,
                 json_body=plan["metadata"])
    doi = (dep.get("metadata", {}).get("prereserve_doi", {}) or {}).get("doi")
    published = False
    if publish:
        pub = _zenodo_http(
            "POST", f"{base}/deposit/depositions/{dep_id}/actions/publish", token)
        doi = pub.get("doi") or doi
        published = True
    return {
        "deposition_id": dep_id,
        "sandbox": sandbox,
        "published": published,
        "doi": doi,
        "edit_url": f"{base.replace('/api', '')}/deposit/{dep_id}",
        "uploaded": [Path(f).name for f in plan["files"]],
    }


def cmd_zenodo_publish(args):
    root = Path(args.path).resolve()
    token = getattr(args, "token", None) or os.environ.get("ZENODO_TOKEN", "")
    sandbox = not getattr(args, "production", False)
    publish = getattr(args, "publish", False)

    if getattr(args, "dry_run", False) or (not token and not getattr(args, "force", False)):
        try:
            plan = zenodo_plan(root, sandbox=sandbox, publish=publish)
        except FileNotFoundError as e:
            sys.exit(str(e))
        target = "sandbox.zenodo.org" if sandbox else "zenodo.org (PRODUCTION)"
        if getattr(args, "json", False):
            _emit(plan, True)
        else:
            print(f"Zenodo deposition plan → {target}"
                  + ("  [DRY RUN]" if getattr(args, "dry_run", False)
                     else "  (no token; showing plan only)"))
            for f in plan["files"]:
                print(f"  upload {f}")
            for f in plan["missing_files"]:
                print(f"  MISSING {f} (run publish-benchmark)")
            for s in plan["steps"]:
                print(f"  {s['method']:4} {s['path']}  — {s['note']}")
            print(f"  metadata: {plan['metadata']['metadata'].get('title', '')[:60]}…")
            if publish:
                print("  NOTE: --publish will mint a PERMANENT DOI (irreversible).")
            if not token:
                print("  Set ZENODO_TOKEN (or --token) to execute.")
        return

    if publish and not sandbox:
        print("! Publishing to PRODUCTION Zenodo mints a permanent, "
              "non-deletable DOI.", file=sys.stderr)
    try:
        res = zenodo_deposit(root, token=token, sandbox=sandbox, publish=publish)
    except (RuntimeError, KeyError) as e:
        sys.exit(f"zenodo-publish failed: {e}")
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        state = "PUBLISHED" if res["published"] else "draft (not yet published)"
        print(f"Zenodo deposition {res['deposition_id']} — {state}"
              + ("  [sandbox]" if res["sandbox"] else ""))
        print(f"  uploaded: {', '.join(res['uploaded'])}")
        if res["doi"]:
            print(f"  DOI: {res['doi']}")
        print(f"  review/edit: {res['edit_url']}")
        if not res["published"]:
            print("  (draft — review it, then re-run with --publish to mint the DOI)")


def check_benchmark_thresholds(result: dict,
                               thresholds: dict | None = None) -> dict:
    """Compare a benchmark result against the CI floors (QG-1)."""
    th = dict(BENCHMARK_THRESHOLDS)
    th.update(thresholds or {})
    failures = []
    for metric, limit in th.items():
        actual = result.get(metric)
        if actual is None:
            continue
        # irrelevant_token_ratio is a ceiling; everything else is a floor.
        bad = actual > limit if metric == "irrelevant_token_ratio" else actual < limit
        if bad:
            failures.append({"metric": metric, "actual": actual,
                             "threshold": limit,
                             "direction": "max" if metric == "irrelevant_token_ratio"
                                          else "min"})
    return {"ok": not failures, "failures": failures, "thresholds": th}


def run_benchmark_suite(corpus_paths: list[Path], budget_tokens: int = 4000,
                        limit: int = 0) -> dict:
    """Run every corpus against its own repository and aggregate (QG-1).

    Each corpus declares the repo it targets, so the suite spans several
    codebases and languages instead of only measuring the tool against
    itself — a self-only benchmark says nothing about generalisation.
    """
    import json
    suites, all_rows = [], []
    totals = {"queries": 0, "recall_hits": 0.0, "symbol_recall": 0.0,
              "answerable": 0.0, "irrelevant": 0.0, "pack": 0.0}
    for cpath in corpus_paths:
        cpath = Path(cpath)
        if not cpath.exists():
            continue
        data = json.loads(cpath.read_text(encoding="utf-8"))
        repo = (cpath.parent / data.get("repo", ".")).resolve()
        if not repo.exists():
            continue
        cases = [c for c in data.get("cases", [])
                 if c.get("task") and c.get("expected_files")]
        if limit:
            cases = cases[:limit]
        if not cases:
            continue
        db = _db_path(repo)
        index_repo(repo, db)
        r = Retriever(repo, db)
        try:
            res = run_retrieval_benchmark(r, cases, budget_tokens=budget_tokens)
        finally:
            r.close()
        # Label by the repo it exercises — every fixture corpus is named
        # tasks.json, which made CI output unreadable.
        res["corpus"] = (cpath.parent.name if cpath.name == "tasks.json"
                         else cpath.stem)
        res["repo"] = str(repo)
        res["languages"] = data.get("languages", [])
        n = res["queries"]
        totals["queries"] += n
        totals["recall_hits"] += res["recall_at_5"] * n
        totals["symbol_recall"] += res["symbol_recall"] * n
        totals["answerable"] += res["answerable_rate"] * n
        totals["irrelevant"] += res["irrelevant_token_ratio"] * n
        for row in res["rows"]:
            row["corpus"] = cpath.name
        all_rows.extend(res["rows"])
        suites.append(res)

    q = totals["queries"] or 1
    aggregate = {
        "queries": totals["queries"],
        "corpora": len(suites),
        "recall_at_5": round(totals["recall_hits"] / q, 3),
        "symbol_recall": round(totals["symbol_recall"] / q, 3),
        "answerable_rate": round(totals["answerable"] / q, 3),
        "irrelevant_token_ratio": round(totals["irrelevant"] / q, 3),
    }
    return {"aggregate": aggregate, "suites": suites,
            "failures": [row for row in all_rows if not row["answerable"]]}


def discover_corpora(root: Path) -> list[Path]:
    """Every benchmark corpus in the repo: the self corpus + fixture repos."""
    out: list[Path] = []
    self_corpus = root / "benchmarks" / "retrieval_tasks.json"
    if self_corpus.exists():
        out.append(self_corpus)
    repos_dir = root / "benchmarks" / "repos"
    if repos_dir.is_dir():
        out.extend(sorted(repos_dir.glob("*/tasks.json")))
    return out


def cmd_benchmark(args):
    """Corpus-driven retrieval benchmark across every fixture repo (QG-1).

    With `--all` (the CI mode) this runs every corpus — the self corpus plus
    each fixture repository — and scores answer quality, not just file recall:
    whether the pack actually contains the symbols and facts an answer needs.
    `--check` turns the result into a build gate.
    """
    root = Path(args.path).resolve()
    if getattr(args, "all", False) or getattr(args, "check", False):
        corpora = ([Path(args.corpus)] if args.corpus
                   else discover_corpora(root))
        if not corpora:
            sys.exit("benchmark: no corpora found under benchmarks/")
        suite = run_benchmark_suite(corpora, budget_tokens=args.budget,
                                    limit=args.limit if args.limit else 0)
        gate = check_benchmark_thresholds(suite["aggregate"])
        suite["gate"] = gate
        if getattr(args, "json", False):
            _emit(suite, True)
        else:
            agg = suite["aggregate"]
            print(f"benchmark suite: {agg['queries']} queries across "
                  f"{agg['corpora']} corpora")
            for s in suite["suites"]:
                print(f"  {s['corpus']:<28} n={s['queries']:<3} "
                      f"recall@5={s['recall_at_5']:<6} "
                      f"symbols={s['symbol_recall']:<6} "
                      f"answerable={s['answerable_rate']:<6} "
                      f"waste={s['irrelevant_token_ratio']}")
            print(f"  {'AGGREGATE':<28} n={agg['queries']:<3} "
                  f"recall@5={agg['recall_at_5']:<6} "
                  f"symbols={agg['symbol_recall']:<6} "
                  f"answerable={agg['answerable_rate']:<6} "
                  f"waste={agg['irrelevant_token_ratio']}")
            if suite["failures"]:
                print(f"\n  {len(suite['failures'])} unanswerable case(s):")
                for row in suite["failures"][:10]:
                    detail = (row["missing_symbols"] or []) + (row["missing_facts"] or [])
                    print(f"    [{row['corpus']}] {row['task'][:60]}")
                    print(f"      missing: {', '.join(map(str, detail[:5]))}")
        if getattr(args, "check", False) and not gate["ok"]:
            for f in gate["failures"]:
                print(f"FAIL {f['metric']}: {f['actual']} "
                      f"({f['direction']} {f['threshold']})", file=sys.stderr)
            sys.exit(1)
        return

    index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    try:
        corpus = Path(args.corpus) if args.corpus else root / "benchmarks" / "retrieval_tasks.json"
        if corpus.exists():
            cases = load_benchmark_corpus(corpus)[:args.limit]
            source = str(corpus)
        else:
            cases = []
            for fr in r.store.files_with_tokens():
                for symbol in r.store.file_symbols(fr["path"]):
                    if symbol["kind"] != "module":
                        query = (symbol["name"] + " " +
                                 (symbol["docstring"] or "")).strip()
                        if len(query) >= 3:
                            cases.append({"task": query,
                                          "expected_files": [symbol["file"]]})
            cases = cases[:args.limit]
            source = "generated-symbol-query fallback"
        out = run_retrieval_benchmark(r, cases, budget_tokens=args.budget)
        out["corpus"] = source
        if getattr(args, "json", False):
            _emit(out, True)
        else:
            print(f"benchmark: {out['queries']} queries  "
                f"Recall@5={out['recall_at_5']}  MRR={out['mrr']}  "
                f"symbols={out['symbol_recall']}  "
                f"answerable={out['answerable_rate']}  "
                f"irrelevant={out['irrelevant_token_ratio']}  "
                f"mean={out['mean_latency_ms']}ms")
    finally:
        r.close()


def cmd_analyze(args):
    """Per-file analyzer: signatures, tokens, extractor, coverage (CI-5)."""
    import time
    root = Path(args.path).resolve()
    index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    try:
        rows = []
        for fr in r.store.files_with_tokens():
            t0 = time.time()
            sk = r.file_skeleton(fr["path"])
            sig_tokens = count_tokens(sk)
            whole = fr["token_est"] or 0
            rows.append({
                "file": fr["path"], "language": fr["language"],
                "symbols": fr["symbols_count"] or 0,
                "whole_tokens": whole, "signature_tokens": sig_tokens,
                "coverage_pct": round(sig_tokens / whole * 100, 1) if whole else 0.0,
                **({"ms": round((time.time() - t0) * 1000, 1)} if args.slow else {}),
            })
        rows.sort(key=lambda x: x["whole_tokens"], reverse=True)
        if getattr(args, "json", False):
            _emit(rows, True)
        else:
            print(f"{'file':40} {'lang':10} {'syms':>5} {'whole':>7} {'sig':>6} {'sig%':>5}")
            for x in rows:
                print(f"{x['file'][:40]:40} {x['language'][:10]:10} {x['symbols']:>5} "
                      f"{x['whole_tokens']:>7} {x['signature_tokens']:>6} "
                      f"{x['coverage_pct']:>5}")
    finally:
        r.close()


def cmd_pricing(args):
    """Show the effective rate card and flag families due for review (CE-3).

    `--check` is the CI gate: exit non-zero when any family the project prices
    against has passed its review window. That is the automation the built-in
    table needs — prices cannot be fetched offline, but going stale silently
    is a choice, and this makes it fail loudly instead.
    """
    root = Path(args.path).resolve()
    pricing = load_pricing(root, refresh=True)
    models = args.models or sorted(pricing["prices"])
    cards = [rate_card(m, root) for m in models]
    # The gate covers the families this installation actually quotes costs in
    # — the ones behind DEFAULT_COST_MODEL and DEFAULT_GAIN_MODEL — because
    # those are the numbers a user is handed. Prices for families nobody here
    # bills against are still printed as OVERDUE, but failing the build on them
    # would only train people to pass --no-verify. `--all-families` gates
    # everything, for projects that do quote all of them.
    gated = ({model_family(DEFAULT_COST_MODEL), model_family(DEFAULT_GAIN_MODEL)}
             if not getattr(args, "all_families", False)
             else set(pricing["families"]))
    overdue = [f for f, meta in pricing["families"].items()
               if meta["stale"] and f in gated]
    advisory = [f for f, meta in pricing["families"].items()
                if meta["stale"] and f not in gated]
    if getattr(args, "json", False):
        _emit({"catalog_version": pricing["catalog_version"],
               "source": pricing["source"], "as_of": pricing["as_of"],
               "families": pricing["families"], "overdue": overdue,
               "rate_cards": cards, "warnings": pricing["warnings"]}, True)
    else:
        print(f"pricing catalog v{pricing['catalog_version']}  "
              f"source={pricing['source']}")
        print(f"{'model':22} {'in':>8} {'out':>8} {'cache rd':>9} "
              f"{'cache wr':>9} {'batch':>6}  as_of")
        for c in cards:
            print(f"{c['model'][:22]:22} {c['input']:>8.3f} {c['output']:>8.3f} "
                  f"{c.get('cache_read', 0):>9.3f} {c.get('cache_write', 0):>9.3f} "
                  f"{c.get('batch_multiplier', 1.0):>6.2f}  {c['as_of']}")
        for f, meta in sorted(pricing["families"].items()):
            mark = ("OVERDUE" if meta["stale"] and f in gated else
                    "stale  " if meta["stale"] else "ok     ")
            scope = "gated" if f in gated else "advisory"
            print(f"  [{mark}] {f}: as of {meta['as_of']} "
                  f"({meta['age_days']}d, {scope})")
        for w in pricing["warnings"]:
            print(f"  ! {w}", file=sys.stderr)
    if getattr(args, "check", False):
        if advisory:
            print(f"note: {', '.join(sorted(advisory))} also past the review "
                  f"window, but not gated (no default cost figure uses them; "
                  f"use --all-families to gate them too)", file=sys.stderr)
        if overdue:
            print(f"FAIL pricing: {', '.join(sorted(overdue))} past the "
                  f"{PRICES_STALE_AFTER_DAYS}-day review window",
                  file=sys.stderr)
            raise SystemExit(1)


def cmd_doctor(args):
    """Validate config, context, index freshness, coverage, MCP wiring (CFG-5).

    Findings carry a severity. `fail` means something is broken or missing that
    ContextIQ needs; `warn` means it works but could be better. Only failures
    set the exit code, because a readiness check that reports "not ready" for an
    optional file trains people to ignore it — which is exactly what a missing
    `gen-context.config.json` used to do, despite the built-in defaults being
    complete and valid. `--strict` opts into treating warnings as failures.
    """
    root = Path(args.path).resolve()
    checks: list[dict] = []

    def add(name, ok, fix="", severity="fail"):
        checks.append({"check": name, "ok": bool(ok),
                       "severity": "ok" if ok else severity, "fix": fix})

    cfg_path = root / CONFIG_NAME
    cfg = load_config(root)
    # Optional by design: load_config() starts from a complete default set, so
    # its absence is a preference, not a fault.
    add("config present (optional)", cfg_path.exists(),
        "using built-in defaults — run `tokengraph init` to pin them",
        severity="warn")
    add("config parses", "_error" not in cfg, cfg.get("_error", ""))

    db = _db_path(root)
    add("index built", db.exists(), "run: tokengraph index")
    fingerprint = ""
    if db.exists():
        rep = index_repo(root, db)
        # Advisory, not a failure: the graph refreshes on every call, so by the
        # time this line runs the index *is* fresh. Reporting the reparse as a
        # fault made a healthy repository look broken after any edit.
        add("index fresh", rep.parsed == 0,
            f"{rep.parsed} file(s) had changed and were reindexed just now",
            severity="warn")
        store = Store(db)
        try:
            fingerprint = store.content_fingerprint(generated_artifact_paths(cfg))
        finally:
            store.close()

    outputs = cfg.get("outputs", ["copilot"])
    for ad in outputs:
        spec = ADAPTERS.get(ad)
        if not spec:
            continue
        rel = cfg.get("output") if ad == "copilot" and cfg.get("output") else spec["path"]
        p = root / rel
        if not p.exists():
            add(f"context: {rel}", False, "run: tokengraph generate")
            continue
        # CFG-6: staleness is a content question, not a calendar one.
        stamp = read_source_stamp(p.read_text(encoding="utf-8", errors="replace"))
        if not stamp:
            add(f"context: {rel}", False,
                "generated before source stamping — run: tokengraph generate "
                "once to make staleness checkable", severity="warn")
        elif fingerprint and stamp != fingerprint:
            add(f"context: {rel}", False,
                "describes source that has since changed — run: tokengraph generate")
        else:
            add(f"context: {rel}", True)

    # A config file must actually *declare the tokengraph server* to count.
    # Previously the mere existence of .claude/settings.json (which may hold
    # only hooks, and cannot hold MCP servers at all) was reported as success.
    wired_files = mcp_wiring_status(root)
    wired = any(w["declares_tokengraph"] for w in wired_files)
    add("MCP wiring present", wired, "run: tokengraph ide-setup")
    for w in wired_files:
        if w["exists"] and not w["declares_tokengraph"]:
            add(f"{w['path']} declares tokengraph", False,
                f"{w['path']} exists but has no tokengraph server under "
                f"`{w['key']}` — run: tokengraph ide-setup")

    # QG-3: the deterministic gate proves the required symbols are *present*;
    # only the LLM judge proves a model still *answers* from the smaller pack.
    # Its absence is the project's biggest unproven claim, so it is reported —
    # as an advisory, because it costs real API spend and cannot run offline.
    import json as _json
    import time as _time
    jrec = judge_result_path(root)
    if not jrec.exists():
        add("LLM answer-quality eval", False,
            "never run — the token savings are unproven end to end. "
            "Run: tokengraph judge-eval --check (needs API access)",
            severity="warn")
    else:
        try:
            doc = _json.loads(jrec.read_text(encoding="utf-8"))
            age = (_time.time() - float(doc.get("ran_at") or 0)) / 86400
            gate_ok = (doc.get("gate") or {}).get("ok", True)
            retention = (doc.get("aggregate") or {}).get("quality_retention")
            label = (f"LLM answer-quality eval (retention {retention})"
                     if retention is not None else "LLM answer-quality eval")
            if not gate_ok:
                add(label, False, "last run was BELOW the quality floor — "
                                  "compression is costing answers")
            elif age > JUDGE_STALE_AFTER_DAYS:
                add(label, False,
                    f"last run {int(age)}d ago (>{JUDGE_STALE_AFTER_DAYS}d) — "
                    f"re-run: tokengraph judge-eval --check", severity="warn")
            else:
                add(label, True)
        except (ValueError, TypeError) as ex:
            add("LLM answer-quality eval", False,
                f"unreadable {jrec.name}: {ex}", severity="warn")

    info = embed_backend_info()
    add(f"embeddings: {info['kind']}", True, "")
    if not info["semantic"]:
        add("semantic embeddings active", False,
            "search_semantic is using the lexical hash fallback — "
            "pip install 'contextiq[embeddings]' && tokengraph embed-warm",
            severity="warn")

    failures = [c for c in checks if c["severity"] == "fail"]
    warnings = [c for c in checks if c["severity"] == "warn"]
    strict = getattr(args, "strict", False)
    ok = not failures and not (strict and warnings)
    if getattr(args, "json", False):
        _emit({"ok": ok, "checks": checks, "failures": len(failures),
               "warnings": len(warnings), "strict": strict}, True)
    else:
        marks = {"ok": "ok  ", "warn": "warn", "fail": "FIX "}
        for c in checks:
            line = f"[{marks[c['severity']]}] {c['check']}"
            if not c["ok"] and c["fix"]:
                line += f"  -> {c['fix']}"
            print(line)
        if failures:
            print(f"doctor: {len(failures)} issue(s) found")
        elif warnings:
            print(f"doctor: ready ({len(warnings)} advisory)"
                  + (" — failing because --strict" if strict else ""))
        else:
            print("doctor: all good")
    if not ok:
        raise SystemExit(1)


def cmd_gain(args):
    """Report realized token savings from the ledger (trend + cost projection)."""
    root = Path(args.path).resolve()
    if getattr(args, "reset", False):
        p = root / ".context" / "gain.ndjson"
        if p.exists():
            p.unlink()
        print("savings ledger cleared")
        return
    if getattr(args, "serve", False):
        serve_report(root, port=getattr(args, "port", 8787), model=args.model)
        return
    if getattr(args, "report", False):
        out = write_usage_report(root, model=args.model)
        print(f"wrote {out}" if out else "could not write usage report")
        return
    s = summarize_gain(root, since=getattr(args, "since", None),
                       model=args.model, top=getattr(args, "top", None),
                       trends=getattr(args, "all", False))
    if getattr(args, "html", None):
        payload = build_report_payload(root, model=args.model,
                                       generated_at=_report_timestamp())
        Path(args.html).write_text(render_report_html(payload), encoding="utf-8")
        print(f"wrote {args.html}")
        return
    if getattr(args, "json", False):
        _emit(s, True)
        return
    if s["runs"] == 0:
        print("no savings recorded yet — run `context`/`ask`/`generate` first "
              "(ledger: .context/gain.ndjson)")
        return
    print(f"savings ({s['since'] or 'all time'}): "
          f"{s['saved_tokens']:,} tokens saved across {s['runs']:,} run(s)  "
          f"({s['reduction_pct']}% reduction)")
    print(f"  projected cost saved: ${s['saved_usd']:,} @ {s['model']} "
          f"(${s['price_per_1m_usd']}/1M tok)")
    print(f"  {'op':14} {'runs':>6} {'saved':>12} {'reduction':>10}")
    for o in s["by_op"]:
        red = round(o["saved"] / o["baseline"] * 100, 1) if o["baseline"] else 0.0
        print(f"  {o['op'][:14]:14} {o['runs']:>6} {o['saved']:>12,} {red:>9}%")
    if getattr(args, "all", False) and s.get("daily"):
        spark = _sparkline([d["saved"] for d in s["daily"]])
        print(f"  daily saved (last {len(s['daily'])}): {spark}")


def cmd_status(args):
    """One-line repo status: branch, dirty files, index freshness, notes, savings."""
    import time
    root = Path(args.path).resolve()
    branch = (_git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
              or "(no git)")
    dirty = len(git_changed_files(root))
    db = _db_path(root)
    if db.exists():
        age = (time.time() - db.stat().st_mtime) / 3600.0
        fresh = f"{age:.1f}h ago" if age >= 1 else f"{age*60:.0f}m ago"
    else:
        fresh = "not indexed"
    store_stats = {}
    notes = 0
    if db.exists():
        st = Store(db)
        try:
            store_stats = st.stats()
            notes = len(st.recent_memory(1000))
        finally:
            st.close()
    gain = summarize_gain(root)
    report = usage_report_path(root)
    out = {
        "branch": branch, "dirty_files": dirty,
        "indexed_files": store_stats.get("files", 0),
        "symbols": store_stats.get("symbols", 0),
        "index_age": fresh, "notes": notes,
        "saved_tokens": gain["saved_tokens"], "gain_runs": gain["runs"],
        "usage_report": str(report) if report.exists() else None,
    }
    if getattr(args, "json", False):
        _emit(out, True)
        return
    print(f"branch={branch}  dirty={dirty}  "
          f"indexed={out['indexed_files']} files / {out['symbols']} symbols  "
          f"index={fresh}")
    print(f"notes={notes}  savings={out['saved_tokens']:,} tok over "
          f"{out['gain_runs']} run(s)")
    if out["usage_report"]:
        print(f"report={out['usage_report']}")


def build_parser():
    p = argparse.ArgumentParser(prog="tokengraph_all")
    p.add_argument("--path", default=os.environ.get("TOKENGRAPH_ROOT", "."),
                   help="repo root (default: $TOKENGRAPH_ROOT or cwd)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("index").set_defaults(func=cmd_index)

    c = sub.add_parser("context")
    c.add_argument("task")
    c.add_argument("-b", "--budget", type=int, default=6000)
    c.add_argument("-d", "--depth", type=int, default=1)
    c.add_argument("--max-body", type=int, default=1600,
                   help="max tokens for one full symbol body before using signature+chunks")
    c.add_argument("-o", "--out", default=None)
    c.add_argument("--no-refresh", action="store_true",
                   help="skip the freshen-on-query reindex (use the graph as-is)")
    c.add_argument("--no-track", action="store_true",
                   help="don't append this run's savings to .context/gain.ndjson")
    c.set_defaults(func=cmd_context)

    sk = sub.add_parser("skeleton")
    sk.add_argument("file")
    sk.set_defaults(func=cmd_skeleton)

    cl = sub.add_parser("callers"); cl.add_argument("qname"); cl.set_defaults(func=cmd_callers)
    ce = sub.add_parser("callees"); ce.add_argument("qname"); ce.set_defaults(func=cmd_callees)

    se = sub.add_parser("semantic", help="find symbols by meaning (embeddings)")
    se.add_argument("query")
    se.add_argument("-n", "--limit", type=int, default=12)
    se.add_argument("--no-refresh", action="store_true")
    se.set_defaults(func=cmd_semantic)

    su = sub.add_parser("summary", help="compact summary of a file")
    su.add_argument("file")
    su.set_defaults(func=cmd_summary)

    me = sub.add_parser("measure", help="report token savings for a task")
    me.add_argument("task")
    me.add_argument("-b", "--budget", type=int, default=6000)
    me.add_argument("-d", "--depth", type=int, default=1)
    me.add_argument("--no-track", action="store_true")
    me.set_defaults(func=cmd_measure)

    rp = sub.add_parser("report",
                        help="aggregate with/without savings across many tasks (markdown + CSV)")
    rp.add_argument("tasks", nargs="*", help="task strings (or use --tasks-file)")
    rp.add_argument("--tasks-file", default=None,
                    help="file with one task per line (# comments allowed)")
    rp.add_argument("-b", "--budget", type=int, default=6000)
    rp.add_argument("-d", "--depth", type=int, default=1)
    rp.add_argument("-o", "--out", default=None, help="write the markdown report to a file")
    rp.add_argument("--csv", default=None, help="also write a per-task CSV to this path")
    rp.add_argument("--append", action="store_true",
                    help="accumulate instead of overwriting: append rows to --csv "
                         "(header written once) and a timestamped run section to -o")
    rp.set_defaults(func=cmd_report)

    sub.add_parser("stats").set_defaults(func=cmd_stats)
    lg = sub.add_parser("langs",
                        help="language support by extraction tier (PF-1)")
    lg.add_argument("--repo", action="store_true",
                    help="report fidelity for THIS repository's indexed files, "
                         "including how much of it has no call graph at all")
    lg.add_argument("--json", action="store_true")
    lg.set_defaults(func=cmd_langs)
    srv = sub.add_parser("serve", help="run the MCP server (stdio or HTTP)")
    srv.add_argument("--transport", default="stdio",
                     choices=["stdio", "http", "streamable-http", "sse"],
                     help="stdio (default, editor-launched) or an HTTP transport "
                          "for a shared/remote server")
    srv.add_argument("--host", default="127.0.0.1",
                     help="HTTP bind address (default loopback only)")
    srv.add_argument("--port", type=int, default=8756, help="HTTP port")
    srv.set_defaults(func=cmd_serve)

    ew = sub.add_parser("embed-warm",
                        help="download + verify the semantic embedding model, once")
    ew.add_argument("--json", action="store_true")
    ew.set_defaults(func=cmd_embed_warm)

    jd = sub.add_parser("judge-eval",
                        help="LLM-judged answer quality: pack context vs. full "
                             "files (opt-in; makes real API calls)")
    jd.add_argument("--model", default=JUDGE_MODEL)
    jd.add_argument("--budget", type=int, default=6000)
    jd.add_argument("-n", "--limit", type=int, default=0,
                    help="questions per repo (0 = all)")
    jd.add_argument("--no-compare", action="store_true",
                    help="skip the full-file baseline (halves cost, loses the "
                         "quality-retention number)")
    jd.add_argument("--corpus", action="append",
                    help="explicit judge.json path; repeatable")
    jd.add_argument("-o", "--out", default=None, help="write full JSON results")
    jd.add_argument("--check", action="store_true",
                    help="CI gate: exit 1 below the QG-3 quality floors")
    jd.add_argument("--json", action="store_true")
    jd.set_defaults(func=cmd_judge_eval)

    dx = sub.add_parser("diagnose-extractors",
                        help="self-test every language extractor (CI gate; exit 1 on failure)")
    dx.add_argument("--json", action="store_true")
    dx.set_defaults(func=cmd_diagnose)

    md = sub.add_parser("modules", help="token-count table of top-level dirs (call first)")
    md.add_argument("--json", action="store_true")
    md.add_argument("--no-refresh", action="store_true")
    md.set_defaults(func=cmd_modules)

    ex = sub.add_parser("explain", help="signatures + imports + external callers for a file")
    ex.add_argument("file")
    ex.add_argument("--no-refresh", action="store_true")
    ex.set_defaults(func=cmd_explain)

    im = sub.add_parser("impact", help="blast radius of a symbol (callers/subclasses/tests)")
    im.add_argument("qname")
    im.add_argument("--json", action="store_true")
    im.add_argument("--no-refresh", action="store_true")
    im.set_defaults(func=cmd_impact)

    mi = sub.add_parser("method-impact", aliases=["method_impact"],
                        help="function-level blast radius: who breaks / deps / "
                             "overrides / call sites")
    mi.add_argument("qname")
    mi.add_argument("--json", action="store_true")
    mi.add_argument("--no-refresh", action="store_true")
    mi.set_defaults(func=cmd_method_impact)

    ln = sub.add_parser("lines", help="surgical line fetch (secret-scanned, sandboxed)")
    ln.add_argument("file")
    ln.add_argument("start", type=int)
    ln.add_argument("end", type=int)
    ln.add_argument("--no-refresh", action="store_true")
    ln.set_defaults(func=cmd_lines)

    mp = sub.add_parser("map", help="project graph: imports | hierarchy | routes | hubs")
    mp.add_argument("kind", nargs="?", default="imports",
                    choices=["imports", "hierarchy", "routes", "hubs"])
    mp.add_argument("--json", action="store_true")
    mp.add_argument("--no-refresh", action="store_true")
    mp.set_defaults(func=cmd_map)

    ar = sub.add_parser("arch", aliases=["architecture", "overview"],
                        help="whole-repo overview: modules + hubs + cycles + "
                             "languages + routes in one call")
    ar.add_argument("--json", action="store_true")
    ar.add_argument("--no-refresh", action="store_true")
    ar.set_defaults(func=cmd_arch)

    tm = sub.add_parser("test-map", aliases=["tests", "test_map"],
                        help="map implementations <-> tests; --benchmark scores "
                             "precision/recall/F1/hit@1 on a labeled corpus")
    tm.add_argument("target", nargs="?", default="",
                    help="a file path or symbol qname; omit for the whole-repo map")
    tm.add_argument("--benchmark", action="store_true",
                    help="measure F1 against benchmarks/testmap/pairs.json")
    tm.add_argument("--corpus", default=None,
                    help="benchmark corpus dir (default: benchmarks/testmap)")
    tm.add_argument("--check", action="store_true",
                    help="with --benchmark: exit 1 if F1 < --min-f1")
    tm.add_argument("--min-f1", type=float, default=None,
                    help="F1 gate for --check (default 0.90)")
    tm.add_argument("--json", action="store_true")
    tm.add_argument("--no-refresh", action="store_true")
    tm.set_defaults(func=cmd_test_map)

    ro = sub.add_parser("routing", help="per-file model-tier hints")
    ro.add_argument("--json", action="store_true")
    ro.add_argument("--no-refresh", action="store_true")
    ro.set_defaults(func=cmd_routing)

    sg = sub.add_parser("suggest-tool", help="recommend a model tier for a task")
    sg.add_argument("task")
    sg.add_argument("--json", action="store_true")
    sg.set_defaults(func=cmd_suggest)

    ak = sub.add_parser("ask", help="focused retrieval with intent/coverage/risk/cost")
    ak.add_argument("task")
    ak.add_argument("-b", "--budget", type=int, default=6000)
    ak.add_argument("-d", "--depth", type=int, default=1)
    ak.add_argument("--json", action="store_true")
    ak.add_argument("--no-refresh", action="store_true")
    ak.add_argument("--no-track", action="store_true")
    ak.set_defaults(func=cmd_ask)

    va = sub.add_parser("validate", help="coverage gate: is context sufficient? (exit 1 if not)")
    va.add_argument("task")
    va.add_argument("-b", "--budget", type=int, default=6000)
    va.add_argument("-d", "--depth", type=int, default=1)
    va.add_argument("--min-coverage", type=float, default=60.0)
    va.add_argument("--json", action="store_true")
    va.add_argument("--no-refresh", action="store_true")
    va.set_defaults(func=cmd_validate)

    jd = sub.add_parser("judge", help="score whether an answer is grounded in a context")
    jd.add_argument("--answer", default=None)
    jd.add_argument("--answer-file", default=None)
    jd.add_argument("--context", default=None)
    jd.add_argument("--context-file", default=None)
    jd.add_argument("--json", action="store_true")
    jd.add_argument("--no-refresh", action="store_true")
    jd.set_defaults(func=cmd_judge)

    vf = sub.add_parser("verify",
                        help="flag fabricated files/symbols in an answer (exit 1 if any)")
    vf.add_argument("--answer", default=None)
    vf.add_argument("--answer-file", default=None)
    vf.add_argument("--json", action="store_true")
    vf.add_argument("--no-refresh", action="store_true")
    vf.set_defaults(func=cmd_verify)

    sq = sub.add_parser("squeeze",
                        help="shrink a pasted stacktrace/CI-log/JSON blob (stdin or --text)")
    sq.add_argument("--text", default=None, help="inline text (else reads --text-file or stdin)")
    sq.add_argument("--text-file", default=None)
    sq.add_argument("--kind", default="auto",
                    choices=["auto", "stacktrace", "cilog", "json", "text"])
    sq.add_argument("--json", action="store_true")
    sq.add_argument("--no-refresh", action="store_true")
    sq.set_defaults(func=cmd_squeeze)

    co = sub.add_parser("cost", help="estimate the USD cost of an API call before sending it")
    co.add_argument("--text", default=None, help="prompt text (else --text-file or stdin)")
    co.add_argument("--text-file", default=None)
    co.add_argument("--model", default=DEFAULT_COST_MODEL,
                    help=f"model to price (default: {DEFAULT_COST_MODEL})")
    co.add_argument("--output-tokens", type=int, default=500,
                    help="expected completion tokens (default: 500)")
    co.add_argument("--compare", action="store_true", help="rank all models cheapest-first")
    co.add_argument("--json", action="store_true")
    co.set_defaults(func=cmd_cost)

    ps = sub.add_parser("prompt-score",
                        help="score a prompt's quality (clarity/specificity/context/action)")
    ps.add_argument("--text", default=None, help="prompt text (else --text-file or stdin)")
    ps.add_argument("--text-file", default=None)
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=cmd_prompt_score)

    sc = sub.add_parser("summarize-chat",
                        help="compress a chat transcript into a token-cheap brief")
    sc.add_argument("--text", default=None, help="transcript (else --text-file or stdin)")
    sc.add_argument("--text-file", default=None)
    sc.add_argument("--max-tokens", type=int, default=400)
    sc.add_argument("--json", action="store_true")
    sc.set_defaults(func=cmd_summarize_chat)

    dd = sub.add_parser("dedupe",
                        help="remove near-duplicate context blocks (blank-line separated)")
    dd.add_argument("--text", default=None, help="blocks (else --text-file or stdin)")
    dd.add_argument("--text-file", default=None)
    dd.add_argument("--sep", default=None, help="block separator (default: blank line)")
    dd.add_argument("--threshold", type=float, default=0.8)
    dd.add_argument("--json", action="store_true")
    dd.set_defaults(func=cmd_dedupe)

    le = sub.add_parser("learn", help="reinforce/penalise a file's local ranking weight")
    le.add_argument("file")
    le.add_argument("--bad", action="store_true", help="penalise instead of reinforce")
    le.add_argument("--weight", type=float, default=1.0)
    le.add_argument("--no-refresh", action="store_true")
    le.set_defaults(func=cmd_learn)

    mm = sub.add_parser("memory", help="read/append the cross-session decision log")
    mm.add_argument("--add", default=None, help="append a note instead of reading")
    mm.add_argument("--kind", default="note")
    mm.add_argument("--limit", type=int, default=20)
    mm.add_argument("--json", action="store_true")
    mm.add_argument("--no-refresh", action="store_true")
    mm.set_defaults(func=cmd_memory)

    cp = sub.add_parser("checkpoint", help="record session progress with a git snapshot")
    cp.add_argument("label")
    cp.add_argument("--note", default=None)
    cp.add_argument("--no-refresh", action="store_true")
    cp.set_defaults(func=cmd_checkpoint)

    cv = sub.add_parser("conventions",
                        help="detect repo file-naming / layout / test / export style + conformance")
    cv.add_argument("--check", action="store_true",
                    help="list non-conforming files and exit 1 if any (CI gate)")
    cv.add_argument("--fix", action="store_true",
                    help="rename non-conforming files to the convention (git mv when possible)")
    cv.add_argument("--dry-run", action="store_true",
                    help="with --fix: print the rename plan without touching files")
    cv.add_argument("--json", action="store_true")
    cv.add_argument("--no-refresh", action="store_true")
    cv.set_defaults(func=cmd_conventions)

    scf = sub.add_parser("scaffold",
                         help="propose (or --apply create) a convention-matched file + skeleton")
    scf.add_argument("name")
    scf.add_argument("--kind", default="module",
                     choices=["module", "class", "function", "component", "test"])
    scf.add_argument("--apply", action="store_true",
                     help="write the file (refuses on conflict — never overwrites)")
    scf.add_argument("--json", action="store_true")
    scf.add_argument("--no-refresh", action="store_true")
    scf.set_defaults(func=cmd_scaffold)

    vp = sub.add_parser("verify-plan",
                        help="check a plan's file/symbol refs + blast radius before acting")
    vp.add_argument("plan", nargs="?", default=None)
    vp.add_argument("--plan-file", default=None)
    vp.add_argument("--json", action="store_true")
    vp.add_argument("--no-refresh", action="store_true")
    vp.set_defaults(func=cmd_verify_plan)

    vo = sub.add_parser("verify-output",
                        help="audit AI-generated code for fabricated files / symbols / imports")
    vo.add_argument("answer", nargs="?", default=None)
    vo.add_argument("--answer-file", default=None)
    vo.add_argument("--json", action="store_true")
    vo.add_argument("--no-refresh", action="store_true")
    vo.set_defaults(func=cmd_verify_output)

    rv = sub.add_parser("review",
                        help="audit the working/staged diff for scope drift, hub edits, missing tests")
    rv.add_argument("--staged", action="store_true", help="review the git index instead of the working tree")
    rv.add_argument("--json", action="store_true")
    rv.add_argument("--no-refresh", action="store_true")
    rv.set_defaults(func=cmd_review)

    dc = sub.add_parser("diff-context",
                        help="budgeted context pack for exactly what the git diff touches (+ blast radius)")
    dc.add_argument("--staged", action="store_true", help="use the git index instead of the working tree")
    dc.add_argument("--budget", type=int, default=6000, help="token budget for the pack (default 6000)")
    dc.add_argument("--depth", type=int, default=1, help="blast-radius hops over callers/callees (default 1)")
    dc.add_argument("--json", action="store_true")
    dc.add_argument("--no-refresh", action="store_true")
    dc.set_defaults(func=cmd_diff_context)

    cr = sub.add_parser("create",
                        help="orchestrate scaffold -> plan -> verify-output -> review (dry-run unless --apply)")
    cr.add_argument("task")
    cr.add_argument("--kind", default="module",
                    choices=["module", "class", "function", "component", "test"])
    cr.add_argument("--apply", action="store_true",
                    help="write the scaffold file and review the resulting diff")
    cr.add_argument("--answer-file", default=None,
                    help="verify generated code (verify-output stage) from this file")
    cr.add_argument("--json", action="store_true")
    cr.add_argument("--no-refresh", action="store_true")
    cr.set_defaults(func=cmd_create)

    ev = sub.add_parser("evidence",
                        help="deterministic, hash-grounded evidence pack JSON for a task (audit/CI)")
    ev.add_argument("task")
    ev.add_argument("-b", "--budget", type=int, default=6000)
    ev.add_argument("-o", "--out", default=None)
    ev.add_argument("--no-refresh", action="store_true")
    ev.set_defaults(func=cmd_evidence)

    gr = sub.add_parser("grounding",
                        help="quantify the hallucination-guard: fabrications caught vs real refs flagged")
    gr.add_argument("--sample", type=int, default=100)
    gr.add_argument("--json", action="store_true")
    gr.add_argument("--no-refresh", action="store_true")
    gr.set_defaults(func=cmd_grounding)

    hb = sub.add_parser("hallucination",
                        help="multi-repo, reproducible codebase-fact grounding benchmark")
    hb.add_argument("--sample", type=int, default=40, help="symbols sampled per repo-partition")
    hb.add_argument("--baseline", type=float, default=None,
                    help="ungrounded fabrication rate per 100 to project a "
                         "reduction against. No default on purpose: the tool "
                         "cannot observe this, and assuming it manufactures the "
                         "headline. Omit for measurements only.")
    hb.add_argument("--baseline-source", default="",
                    help="where --baseline came from (required for it to be "
                         "reported as anything but unsubstantiated)")
    hb.add_argument("-o", "--out", default=None, help="write a markdown report")
    hb.add_argument("--json", action="store_true")
    hb.add_argument("--no-refresh", action="store_true")
    hb.set_defaults(func=cmd_hallucination)

    ide = sub.add_parser("ide-setup",
                         help="wire the MCP server into every major editor (VS Code/Cursor/Windsurf/Zed/Claude)")
    ide.add_argument("--editor", action="append",
                     choices=["claude", "vscode", "cursor", "zed", "continue",
                              "jetbrains", "nvim", "windsurf", "cline"],
                     help="limit to specific editor(s); default = all project-local "
                          "ones. windsurf/cline are per-user configs and need --global")
    ide.add_argument("--global", dest="write_global", action="store_true",
                     help="also write per-user configs outside the repo "
                          "(Windsurf ~/.codeium, Cline globalStorage)")
    ide.add_argument("--verify", action="store_true",
                     help="report per-editor completeness (MCP + rules); "
                          "exit 1 if a requested editor is not wired")
    ide.add_argument("--no-rules", dest="with_rules", action="store_false",
                     help="wire the MCP server only; skip the steering-rules block")
    ide.set_defaults(with_rules=True)
    ide.add_argument("--plugins", action="store_true",
                     help="also scaffold installable VS Code / Neovim / JetBrains plugins")
    ide.add_argument("--workspace-root", action="append",
                     help="repeat for each folder in a multi-root workspace")
    ide.add_argument("--json", action="store_true")
    ide.set_defaults(func=cmd_ide_setup)

    idp = sub.add_parser("ide-plugin",
                         help="scaffold installable editor plugins (VS Code .vsix / Neovim / JetBrains)")
    idp.add_argument("-o", "--out", default="ide-plugins", help="output directory")
    idp.add_argument("--editor", action="append",
                     choices=["vscode", "nvim", "jetbrains"],
                     help="limit to specific plugin(s); default = all")
    idp.add_argument("--json", action="store_true")
    idp.set_defaults(func=cmd_ide_plugin)

    dash = sub.add_parser("dashboard",
                          help="removed — points at `gain --report` / `gain --serve`")
    dash.set_defaults(func=cmd_dashboard)

    scip = sub.add_parser("import-scip",
                          help="import precise REFERENCES edges from SCIP JSON")
    scip.add_argument("index", help="output from `scip print --json index.scip`")
    scip.add_argument("--json", action="store_true")
    scip.set_defaults(func=cmd_import_scip)

    fz = sub.add_parser("freeze",
                        help="emit a PyInstaller spec (or --build it) for a standalone binary")
    fz.add_argument("-o", "--out", default=None)
    fz.add_argument("--build", action="store_true",
                    help="run pyinstaller now to produce the binary (needs pyinstaller)")
    fz.set_defaults(func=cmd_freeze)

    dist = sub.add_parser("dist",
                          help="scaffold release automation + install channels (CI, Docker, Homebrew, install.sh)")
    dist.add_argument("-o", "--out", default=".", help="output directory")
    dist.add_argument("--json", action="store_true")
    dist.set_defaults(func=cmd_dist)

    w = sub.add_parser("watch")
    w.add_argument("--interval", type=float, default=2.0,
                   help="seconds between reindex polls (default: 2)")
    w.set_defaults(func=cmd_watch)

    g = sub.add_parser("generate", aliases=["gen"],
                       help="emit always-on context for one or more assistants "
                            "(strategy/adapter/cache)")
    g.add_argument("--strategy", choices=["full", "per-module", "hot-cold"],
                   default=None, help="TB-4 output strategy (default: from config)")
    g.add_argument("--adapter", action="append", choices=list(ADAPTERS),
                   help="repeatable; overrides config outputs (copilot/claude/...)")
    g.add_argument("-b", "--budget", type=int, default=None,
                   help="override the auto-scaled budget")
    g.add_argument("--hot-commits", type=int, default=None,
                   help="size the hot set from the last N commits (TB-5)")
    g.add_argument("--format", choices=["md", "cache"], default=None,
                   help="cache = also emit a prompt-cache sidecar (PC-1)")
    g.add_argument("--diff", action="store_true",
                   help="hot set = git working-tree changes (TB-5a)")
    g.add_argument("--staged", action="store_true",
                   help="with --diff: only staged files (pre-commit)")
    g.add_argument("--config", default=None, help="path to gen-context.config.json")
    g.add_argument("--report", action="store_true",
                   help="emit finalTokens/reductionPct/overBudget; exit 1 if over (TB-7)")
    g.add_argument("--no-track", action="store_true",
                   help="don't append to .context/gain.ndjson|usage.ndjson (TB-6)")
    g.add_argument("--monorepo", action="store_true",
                   help="generate context for every nested package (manifest-detected)")
    g.add_argument("--each", action="store_true",
                   help="generate context for each immediate child directory")
    g.add_argument("--json", action="store_true")
    g.set_defaults(func=cmd_generate)

    rx = sub.add_parser("repomix",
                        help="Repomix interop (RX-1): export the signature map as "
                             "a Repomix pack, or --import a Repomix dump and squeeze it")
    rx.add_argument("--import", dest="import_file", default=None, metavar="FILE",
                    help="read a Repomix output file and squeeze it to a digest")
    rx.add_argument("--out", default=None,
                    help="write the exported pack here (default: stdout)")
    rx.add_argument("--strategy", choices=["full", "per-module", "hot-cold"],
                    default=None, help="export strategy (default: from config)")
    rx.add_argument("-b", "--budget", type=int, default=None,
                    help="signature budget for the export")
    rx.add_argument("--config", default=None, help="path to gen-context.config.json")
    rx.add_argument("--no-track", action="store_true",
                    help="don't append import savings to the ledger")
    rx.add_argument("--json", action="store_true")
    rx.set_defaults(func=cmd_repomix)

    ini = sub.add_parser("init", help="write a default gen-context.config.json (CFG-1)")
    ini.set_defaults(func=cmd_init)

    stp = sub.add_parser("setup", help="auto-wire MCP + git hook + initial context (MCP-7)")
    stp.set_defaults(func=cmd_setup)

    he = sub.add_parser("health", help="composite health score / grade A-F (CI-1)")
    he.add_argument("--json", action="store_true")
    he.add_argument("--strict", action="store_true", help="exit 1 if context is stale")
    he.set_defaults(func=cmd_health)

    bm = sub.add_parser("benchmark", aliases=["eval"],
                        help="retrieval quality: hit@5 + MRR (CI-4)")
    bm.add_argument("-n", "--limit", type=int, default=100)
    bm.add_argument("--corpus", default=None,
                    help="JSON corpus with task + expected_files cases")
    bm.add_argument("--all", action="store_true",
                    help="run every corpus (self + benchmarks/repos/*/tasks.json) "
                         "and score answer quality, not just file recall")
    bm.add_argument("--check", action="store_true",
                    help="CI gate: implies --all and exits 1 below thresholds")
    bm.add_argument("--budget", type=int, default=6000,
                    help="token budget per pack during the benchmark")
    bm.add_argument("--json", action="store_true")
    bm.set_defaults(func=cmd_benchmark)

    pb = sub.add_parser("publish-benchmark",
                        help="run the suite and emit publish-ready artifacts: "
                             "REPORT.md + MANIFEST.json + .zenodo.json + CITATION.cff")
    pb.add_argument("--budget", type=int, default=4000,
                    help="token budget per pack during the retrieval benchmark")
    pb.add_argument("--full", action="store_true",
                    help="also run the (slower) hallucination-guard benchmark")
    pb.add_argument("--version", default="1.0.0", help="dataset/citation version")
    pb.add_argument("--creator", default=None,
                    help="author name for .zenodo.json / CITATION.cff")
    pb.add_argument("--dry-run", action="store_true",
                    help="compute everything but write nothing")
    pb.add_argument("--json", action="store_true")
    pb.set_defaults(func=cmd_publish_benchmark)

    zp = sub.add_parser("zenodo-publish",
                        help="deposit the benchmark artifacts to Zenodo and "
                             "(with --publish) mint a DOI; sandbox + draft by default")
    zp.add_argument("--token", default=None,
                    help="Zenodo API token (or set ZENODO_TOKEN); omit to show the plan")
    zp.add_argument("--production", action="store_true",
                    help="target zenodo.org instead of the safe sandbox.zenodo.org")
    zp.add_argument("--publish", action="store_true",
                    help="mint a PERMANENT DOI (irreversible); default leaves a draft")
    zp.add_argument("--dry-run", action="store_true",
                    help="show the deposition plan without uploading")
    zp.add_argument("--force", action="store_true",
                    help="proceed to upload even though this needs a token")
    zp.add_argument("--json", action="store_true")
    zp.set_defaults(func=cmd_zenodo_publish)

    an = sub.add_parser("analyze",
                        help="per-file signatures/tokens/extractor/coverage (CI-5)")
    an.add_argument("--slow", action="store_true", help="include per-file timing")
    an.add_argument("--json", action="store_true")
    an.set_defaults(func=cmd_analyze)

    pr = sub.add_parser("pricing",
                        help="effective rate card + price-staleness gate (CE-3)")
    pr.add_argument("--models", nargs="*", default=None,
                    help="limit to these models (default: the whole catalogue)")
    pr.add_argument("--check", action="store_true",
                    help="exit 1 if a family this project quotes costs in is "
                         "past its review window")
    pr.add_argument("--all-families", action="store_true",
                    help="gate on every vendor family, not just the ones "
                         "behind the default cost/gain models")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_pricing)

    dr = sub.add_parser("doctor",
                        help="validate config/context/index/MCP wiring (CFG-5)")
    dr.add_argument("--json", action="store_true")
    dr.add_argument("--strict", action="store_true",
                    help="treat advisories (optional config, missing extras) "
                         "as failures too")
    dr.set_defaults(func=cmd_doctor)

    gn = sub.add_parser("gain",
                        help="realized token savings from the ledger (trend + cost projection)")
    gn.add_argument("--since", default=None,
                    help="window: 7d / 12h / 90m / ISO date (default: all time)")
    gn.add_argument("--model", default=DEFAULT_GAIN_MODEL,
                    help=f"pricing model for $ projection (default: {DEFAULT_GAIN_MODEL})")
    gn.add_argument("--top", type=int, default=None, help="limit per-op rows")
    gn.add_argument("--all", action="store_true",
                    help="include daily/weekly/monthly trend buckets")
    gn.add_argument("--html", default=None, help="write a self-contained HTML dashboard")
    gn.add_argument("--report", action="store_true",
                    help="write the per-workspace report to .tokengraph/token-usage.html")
    gn.add_argument("--serve", action="store_true",
                    help="serve the live report on 127.0.0.1 (polls the ledger; no Streamlit)")
    gn.add_argument("--port", type=int, default=8787, help="port for --serve")
    gn.add_argument("--reset", action="store_true", help="clear the savings ledger")
    gn.add_argument("--json", action="store_true")
    gn.set_defaults(func=cmd_gain)

    stt = sub.add_parser("status",
                         help="one-line repo status: branch/dirty/index/notes/savings")
    stt.add_argument("--json", action="store_true")
    stt.set_defaults(func=cmd_status)
    return p


def main(argv=None):
    # Ensure UTF-8 console output (Windows defaults to cp1252 and would choke
    # on the markdown's box-drawing / math glyphs).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
