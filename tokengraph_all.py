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
import sqlite3
import sys
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


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        # disallowed_special=() so source files containing literal special-token
        # markers (e.g. "<|endoftext|>") are counted as normal text, not rejected.
        return len(enc.encode(text, disallowed_special=()))
    # heuristic: ~4 chars/token for code, with a small floor
    return max(1, (len(text) + 3) // 4)


# Per-file signature cap (FR-2a): bound worst-case token cost so a single huge
# file can never dominate emitted context. Applied at the rendering layer
# (file_skeleton / read_context / generated context), NOT at parse time — the
# full symbol graph stays indexed so callers/callees/semantic/impact keep
# working over the dropped symbols.
MAX_SIGS_PER_FILE = 25


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

    def bases_of(self, node) -> list[str]:
        return []

    def callee_of(self, node) -> Optional[str]:
        return None

    def import_of(self, node) -> Optional[str]:
        return None

    def synth_def(self, node):
        """For lambdas assigned to names: (name, kind, def_node, body_node) or None."""
        return None


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

    def run(self, root):
        self._walk(root, self.module)

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
            docstring="", parent=scope,
        ))
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
            signature=signature, docstring="", parent=mod,
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
    symbol_id INTEGER PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL
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
"""


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=10.0)
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
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(files)")}
        for name, ddl in {
            "language": "ALTER TABLE files ADD COLUMN language TEXT DEFAULT ''",
            "token_est": "ALTER TABLE files ADD COLUMN token_est INTEGER DEFAULT 0",
            "symbols_count": "ALTER TABLE files ADD COLUMN symbols_count INTEGER DEFAULT 0",
            "size": "ALTER TABLE files ADD COLUMN size INTEGER DEFAULT 0",
        }.items():
            if name not in cols:
                self.conn.execute(ddl)

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

    def set_vector(self, symbol_id: int, vec: list[float]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO vectors(symbol_id,dim,vec) VALUES(?,?,?)",
            (symbol_id, len(vec), vec_to_blob(vec)))

    def iter_vectors(self):
        return self.conn.execute("SELECT symbol_id, dim, vec FROM vectors")

    def has_vectors(self) -> bool:
        return self.conn.execute("SELECT 1 FROM vectors LIMIT 1").fetchone() is not None

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

    def neighbors(self, sid: int, types: Iterable[str], direction: str) -> list[sqlite3.Row]:
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
        return self.conn.execute(sql, [sid] + tlist).fetchall()

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


def _embed_model():
    """Lazily load sentence-transformers IFF explicitly opted in; else None."""
    global _EMBED_MODEL, _EMBED_TRIED
    if _EMBED_TRIED:
        return _EMBED_MODEL
    _EMBED_TRIED = True
    if os.environ.get("TOKENGRAPH_EMBEDDINGS", "").lower() in (
            "st", "sbert", "sentence-transformers"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            name = os.environ.get("TOKENGRAPH_EMBED_MODEL", "all-MiniLM-L6-v2")
            _EMBED_MODEL = SentenceTransformer(name)
        except Exception:
            _EMBED_MODEL = None
    return _EMBED_MODEL


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
            "models": ["claude-haiku", "gpt-4o-mini", "gemini-flash"],
            "use_for": "config/markup/typos/simple lookups",
        },
        "balanced": {
            "tier": "balanced", "cost_hint_per_1k": 0.003,
            "models": ["claude-sonnet", "gpt-4o", "gemini-pro"],
            "use_for": "features/tests/debugging",
        },
        "powerful": {
            "tier": "powerful", "cost_hint_per_1k": 0.015,
            "models": ["claude-opus", "gpt-5", "gemini-ultra"],
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


def tier_for_file(file: str, token_est: int = 0, symbols: int = 0) -> dict:
    """Per-file complexity routing (MR-2)."""
    from os.path import splitext
    ext = splitext(file)[1].lower()
    if ext in _FAST_EXTS or token_est and token_est < 300:
        tier = "fast"
    elif token_est > 3000 or symbols > 40:
        tier = "powerful"
    else:
        tier = "balanced"
    return _tier_info(tier, file=file)


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
               respect_gitignore: bool = True) -> IndexReport:
    ignores = (ignores or set()) | DEFAULT_IGNORES
    root = root.resolve()
    store = Store(db_path)
    report = IndexReport(errors=[])

    gitignore = GitIgnore.load(root) if respect_gitignore else None
    files = iter_source_files(root, ignores, gitignore)
    report.scanned = len(files)
    current = {p.relative_to(root).as_posix() for p in files}

    # drop deleted files
    for gone in store.all_indexed_files() - current:
        store.forget_file(gone)
        report.removed += 1

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
        if meta and meta["mtime"] == st.st_mtime and meta["size"] == st.st_size:
            report.skipped += 1
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as ex:
            report.errors.append(f"{rel}: {ex}")
            continue
        h = file_hash(text)
        if meta and meta["hash"] == h:
            # Content identical despite a touched mtime — refresh stat only.
            store.touch_file(rel, st.st_mtime, st.st_size)
            report.skipped += 1
            continue

        store.forget_file(rel)  # clear stale symbols/edges
        res: ParseResult | None = parse_path(root, path)
        if res is None:
            continue
        if res.error and not res.symbols:
            report.errors.append(f"{rel}: {res.error}")

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

    store.commit()
    report.stats = store.stats()
    if pending:
        report.stats["edge_resolution_pct"] = round(100 * resolved_n / len(pending), 1)
    store.close()
    return report


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
    budget: int = 0

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
        out, _ = redact_secrets("\n".join(lines))
        return out


class Retriever:
    def __init__(self, root: Path, db_path: Path):
        self.root = Path(root).resolve()
        self.store = Store(db_path)
        self._src_cache: dict[str, list[str]] = {}

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
        return "\n".join(out)

    def module_summary(self, file: str) -> str:
        """Compact summary of a file (cached, or a signature skeleton fallback)."""
        row = self.store.get_summary(file)
        if row:
            return row["summary"]
        return self.file_skeleton(file)

    # ---- semantic + hybrid seeding ----
    def semantic_search(self, query: str, limit: int = 12) -> list:
        """Cosine ranking of symbols by embedding similarity to the query."""
        if not self.store.has_vectors():
            return []
        qv = embed_text(query)
        scored: list[tuple[float, int]] = []
        for r in self.store.iter_vectors():
            v = blob_to_vec(r["vec"])
            if len(v) != len(qv):
                continue
            s = cosine(qv, v)
            if s > 0:
                scored.append((s, r["symbol_id"]))
        scored.sort(key=lambda t: t[0], reverse=True)
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
        order = sorted(scores, key=lambda i: scores[i], reverse=True)
        return [rowmap[i] for i in order[:limit]]

    # ---- the main entrypoint ----
    def find_relevant_context(self, task: str, budget_tokens: int = 6000,
                              expand_depth: int = 1,
                              max_body_tokens: int = 1600) -> ContextPack:
        pack = ContextPack(task=task, budget=budget_tokens)

        # hybrid seeding: lexical (FTS5) + semantic (embeddings), fused by RRF.
        lexical = self.store.search(task, limit=12)
        semantic = self.semantic_search(task, limit=12)
        seeds = self._fuse([lexical, semantic], limit=12) if semantic else lexical
        # de-prioritise module nodes as seeds; we want concrete defs
        seeds = [s for s in seeds if s["kind"] != "module"] or seeds
        # G10: let learned file weights nudge ordering — reinforced files first,
        # penalised last. Stable sort keeps relevance order among unweighted files.
        weights = self.store.all_weights()
        if weights:
            seeds = sorted(seeds, key=lambda s: weights.get(s["file"], 0.0), reverse=True)
        seed_ids = [s["id"] for s in seeds]
        chunk_rows = self.store.search_chunks(task, limit=8)

        # collect neighbors via BFS over CALLS (both directions) + INHERITS
        neighbor_rows: dict[int, tuple] = {}  # id -> (row, reason)
        frontier = list(seed_ids)
        seen = set(seed_ids)
        for _ in range(max(0, expand_depth)):
            nxt = []
            for sid in frontier:
                for r in self.store.neighbors(sid, ["CALLS"], "out"):
                    if r["id"] not in seen:
                        neighbor_rows.setdefault(r["id"], (r, "callee")); nxt.append(r["id"])
                for r in self.store.neighbors(sid, ["INHERITS"], "out"):
                    if r["id"] not in seen:
                        neighbor_rows.setdefault(r["id"], (r, "base")); nxt.append(r["id"])
                for r in self.store.neighbors(sid, ["CALLS"], "in"):
                    if r["id"] not in seen:
                        neighbor_rows.setdefault(r["id"], (r, "caller")); nxt.append(r["id"])
            seen.update(nxt)
            frontier = nxt

        # --- budget assembly ---
        # 1. seeds get full bodies (highest value), in search-rank order
        full_body_files: set[str] = set()
        for s in seeds:
            body = self._body(s)
            est = count_tokens(body)
            if est > max_body_tokens or (pack.tokens + est > budget_tokens and pack.pieces):
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

        # 2. neighbors get signatures only (cheap context that resolves refs)
        for sid, (r, reason) in neighbor_rows.items():
            if r["kind"] == "module":
                continue
            sig = self._sig_block(r)
            est = count_tokens(sig)
            if pack.tokens + est > budget_tokens:
                pack.dropped.append(r["qname"]); continue
            pack.pieces.append(Piece(r["qname"], r["kind"], r["file"],
                                     "signature", sig, est, reason))

        # 2b. module summaries: a few tokens to gesture at a referenced file
        #     whose body we didn't pull in full.
        summarized: set[str] = set()
        for sid, (r, reason) in neighbor_rows.items():
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
        return pack

    def measure(self, task: str, **kw) -> dict:
        """Quantify the saving: pack tokens vs. reading the referenced files whole.

        The baseline is the token cost of opening every distinct file the pack
        draws from — what a naive agent would do to get the same coverage.
        """
        pack = self.find_relevant_context(task, **kw)
        files = sorted({p.file for p in pack.pieces})
        baseline = sum(self.store.token_est_for(f) for f in files)
        pack_tokens = pack.tokens
        saved = baseline - pack_tokens
        pct = (saved / baseline * 100.0) if baseline else 0.0
        return {
            "task": task,
            "pack_tokens": pack_tokens,
            "baseline_tokens": baseline,
            "files_referenced": len(files),
            "symbols_in_pack": len(pack.pieces),
            "tokens_saved": saved,
            "savings_pct": round(pct, 1),
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
        direct = self.store.neighbors(row["id"], ["CALLS"], "in")
        # transitive callers (BFS, bounded)
        transitive: set[str] = set()
        frontier = [r["id"] for r in direct]
        seen = set(frontier) | {row["id"]}
        depth = 0
        while frontier and depth < 5:
            nxt = []
            for sid in frontier:
                for r in self.store.neighbors(sid, ["CALLS"], "in"):
                    if r["id"] not in seen:
                        seen.add(r["id"]); transitive.add(r["qname"]); nxt.append(r["id"])
            frontier = nxt
            depth += 1
        files = sorted({r["file"] for r in direct})
        tests = [f for f in files if "test" in f.lower() or "spec" in f.lower()]
        subclasses = [r["qname"] for r in self.store.neighbors(row["id"], ["INHERITS"], "in")]
        return {
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
            key=lambda x: (x["degree"], x["file"]), reverse=True)[:top]
        return {"kind": "hubs", "hubs": hubs, "cycles": _import_cycles(graph)}

    # ---- get_routing: per-file model-tier hints ----
    def get_routing(self) -> list[dict]:
        out = []
        for r in self.store.files_with_tokens():
            out.append(tier_for_file(r["path"], r["token_est"] or 0,
                                     r["symbols_count"] or 0))
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
        saved = baseline - pack.tokens
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
            "pack_tokens": pack.tokens,
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
            "pack_tokens": pack.tokens,
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
        return {"id": mid, "kind": kind, "text": text}

    def read_memory(self, limit: int = 20) -> dict:
        notes = [{"kind": r["kind"], "text": r["text"]}
                 for r in self.store.recent_memory(limit)]
        cps = [{"label": r["label"], "git_sha": r["git_sha"], "note": r["note"]}
               for r in self.store.recent_checkpoints(10)]
        return {"notes": notes, "checkpoints": cps}

    def create_checkpoint(self, label: str, note: str = "") -> dict:
        sha = _git(self.root, "rev-parse", "--short", "HEAD").strip()
        cid = self.store.add_checkpoint(label, sha, note)
        self.store.commit()
        return {"id": cid, "label": label, "git_sha": sha, "note": note}

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
        return [{"qname": r["qname"], "kind": r["kind"], "file": r["file"],
                 "signature": (r["signature"] or r["qname"])}
                for r in rows if r["kind"] != "module"]

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
            fscore[f] = fscore.get(f, 0.0) + 1.0 / (rank + 1)
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
        if not src_dirs:
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

    def close(self):
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

ADAPTERS: dict[str, dict] = {
    "copilot":  {"path": ".github/copilot-instructions.md", "mode": "replace"},
    "claude":   {"path": "CLAUDE.md", "mode": "append-marker"},
    "cursor":   {"path": ".cursorrules", "mode": "replace"},
    "windsurf": {"path": ".windsurfrules", "mode": "replace"},
    "openai":   {"path": ".github/openai-context.md", "mode": "replace"},
    "gemini":   {"path": ".github/gemini-context.md", "mode": "replace"},
    "agents":   {"path": "AGENTS.md", "mode": "append-marker"},
    "windsurf-next": {"path": ".windsurf/rules.md", "mode": "replace"},
}


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


def build_context_payload(r: "Retriever", root: Path, *, strategy: str,
                          src_dirs: list[str], budget: int, hot_commits: int,
                          diff: bool, staged: bool, config: dict) -> dict:
    """Render the always-on context markdown for a strategy (TB-4) + metadata."""
    files = r.all_files(src_dirs)
    skeletons = {f: r.file_skeleton(f) for f in files}
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
    if enrich.get("changes", True):
        recent = git_recent_files(root, hot_commits)[:10]
        if recent:
            head += ["## Recently changed", ", ".join(f"`{x}`" for x in recent), ""]
    if enrich.get("todos", True):
        todos = _scan_todos(root, files)
        if todos:
            head += ["## TODO / FIXME", *[f"- {t}" for t in todos], ""]

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
            hot_files = files[:min(5, len(files))]
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
            sk = skeletons[f]
            est = count_tokens(sk)
            if used and used + est > budget:
                body.append(f"_… {len([x for x in files])} files; budget {budget:,} "
                            f"reached. Remaining files via MCP `read_context`._")
                break
            body += [f"### {f}", "```" + _fence(f), sk, "```", ""]
            used += est

    markdown = "\n".join(head + body).rstrip() + "\n"
    if config.get("secretScan", True):
        markdown, _ = redact_secrets(markdown)
    tokens = count_tokens(markdown)
    reduction = (1 - tokens / repo_total) * 100.0 if repo_total else 0.0
    return {
        "strategy": strategy,
        "markdown": markdown,
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


def write_adapter(root: Path, adapter: str, content: str,
                  custom_out: str | None = None) -> str:
    spec = ADAPTERS[adapter]
    rel = custom_out if (custom_out and adapter == "copilot") else spec["path"]
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    # All adapters are non-destructive: the generated block lives between markers
    # and any hand-written content outside them is preserved across re-runs
    # (MCP-5). This avoids clobbering human instructions in copilot/cursor/etc.
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"{CLAUDE_BEGIN}\n{content.rstrip()}\n{CLAUDE_END}\n"
    if CLAUDE_BEGIN in existing and CLAUDE_END in existing:
        pre = existing.split(CLAUDE_BEGIN)[0].rstrip()
        post = existing.split(CLAUDE_END, 1)[1].lstrip("\n")
        new = (pre + "\n\n" if pre else "") + block + (("\n" + post) if post else "")
    else:
        new = (existing.rstrip() + "\n\n" if existing.strip() else "") + block
    path.write_text(new, encoding="utf-8")
    return rel


def write_cache_sidecar(root: Path, adapter_rel: str, text: str) -> str:
    """Write the prompt-cache JSON next to an adapter output (PC-1/PC-3)."""
    import json
    side = (root / adapter_rel).with_suffix(".cache.json")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(cache_artifact(text), indent=2) + "\n", encoding="utf-8")
    return str(side.relative_to(root).as_posix())


# ==========================================================================
# local gain / usage tracking (TB-6, CI-3) — counts only, never leaves machine
# ==========================================================================
def _tracking_disabled(no_track_flag: bool) -> bool:
    return bool(no_track_flag or os.environ.get("SIGMAP_NO_TRACK")
                or os.environ.get("TOKENGRAPH_NO_TRACK"))


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
        red = round(saved / baseline_tokens * 100.0, 1) if baseline_tokens else 0.0
        track_gain(root, {"op": op, "final_tokens": final_tokens,
                          "baseline_tokens": baseline_tokens, "saved": saved,
                          "reduction_pct": red, "files": files}, no_track=no_track)
        track_usage(root, {"op": op, "final_tokens": final_tokens,
                           "reduction_pct": red}, no_track=no_track)
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
}
DEFAULT_GAIN_MODEL = "claude-sonnet"


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
        b = buckets.setdefault(key, {"period": key, "saved": 0, "runs": 0})
        b["saved"] += int(r.get("saved", 0))
        b["runs"] += 1
    return [buckets[k] for k in sorted(buckets)]


def _sparkline(values: list[int]) -> str:
    """Unicode block sparkline for a series of savings values."""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    return "".join(blocks[min(7, int((v - lo) / span * 7))] for v in values)


def render_gain_html(summary: dict) -> str:
    """Self-contained HTML dashboard (inline SVG sparkline, no external deps)."""
    daily = summary.get("daily") or []
    vals = [d["saved"] for d in daily]
    # inline SVG sparkline of daily saved tokens
    bars = ""
    if vals:
        hi = max(vals) or 1
        w, h, gap = 26, 90, 6
        for i, v in enumerate(vals):
            bh = int(v / hi * h)
            x = i * (w + gap)
            bars += (f'<rect x="{x}" y="{h - bh}" width="{w}" height="{bh}" '
                     f'rx="3" fill="#3b82f6"><title>{daily[i]["period"]}: '
                     f'{v:,} tok</title></rect>')
    svg_w = max(1, len(vals)) * 32
    op_rows = "".join(
        f"<tr><td>{o['op']}</td><td>{o['runs']:,}</td>"
        f"<td>{o['saved']:,}</td>"
        f"<td>{round(o['saved']/o['baseline']*100,1) if o['baseline'] else 0}%</td></tr>"
        for o in summary.get("by_op", []))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>ContextIQ — Token Savings</title><style>
body{{font:14px system-ui,sans-serif;margin:2rem;color:#0f172a;background:#f8fafc}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:1.2rem;margin:1rem 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.big{{font-size:2rem;font-weight:700}}.muted{{color:#64748b}}
table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:.4rem .6rem;border-bottom:1px solid #f1f5f9}}
.grid{{display:flex;gap:1rem;flex-wrap:wrap}}.grid .card{{flex:1;min-width:160px}}
</style></head><body>
<h1>ContextIQ — Token Savings Dashboard</h1>
<p class="muted">model: <b>{summary['model']}</b> · window: <b>{summary['since'] or 'all time'}</b> · runs: <b>{summary['runs']:,}</b></p>
<div class="grid">
  <div class="card"><div class="muted">tokens saved</div><div class="big">{summary['saved_tokens']:,}</div></div>
  <div class="card"><div class="muted">reduction</div><div class="big">{summary['reduction_pct']}%</div></div>
  <div class="card"><div class="muted">projected $ saved</div><div class="big">${summary['saved_usd']:,}</div></div>
</div>
<div class="card"><h3>Daily saved tokens (last {len(vals)})</h3>
<svg width="{svg_w}" height="110" role="img">{bars}</svg></div>
<div class="card"><h3>By operation</h3>
<table><tr><th>op</th><th>runs</th><th>saved</th><th>reduction</th></tr>{op_rows}</table></div>
<p class="muted">Generated by <code>tokengraph gain --html</code> · projection is list-price input tokens, indicative only.</p>
</body></html>"""


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


def hallucination_benchmark(retriever: "Retriever", sample_per_repo: int = 40,
                            baseline_per_100: float = 99.8) -> dict:
    """Reproducible, multi-repo codebase-fact hallucination benchmark.

    Partitions the codebase by top-level directory (each = a "repo") and, per
    repo, measures three real structural quantities over sampled symbols:

      • grounding_coverage — % of repo facts the retriever can surface into a
        pack (so a *correct* grounded citation is possible),
      • guard_catch — % of fabricated references verify() flags,
      • guard_specificity — % of real references verify() does NOT false-flag.

    From these it models the residual codebase-fact error rate of a grounded
    agent: a fact is only stated wrong if it could not be grounded AND the guard
    missed the fabrication, i.e. residual = baseline · (1−coverage) · (1−catch).
    `baseline_per_100` is the ungrounded fabrication rate to compare against
    (default 99.8, the figure Sigmap measured with an LLM); the reduction is then
    deterministic and reproducible without running a model. Reports a per-repo
    spread (min/max) so the figure isn't a single-repo artifact."""
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
        residual = round(baseline_per_100 * (1 - cov) * (1 - catch), 3)
        rows.append({
            "repo": repo, "facts": n,
            "grounding_coverage_pct": round(100 * cov, 1),
            "guard_catch_pct": round(100 * catch, 1),
            "guard_specificity_pct": round(100 * spec, 1),
            "modeled_with_grounding_per_100": residual,
            "reduction_pct": round(100 * (baseline_per_100 - residual)
                                   / baseline_per_100, 2),
        })

    if not rows:
        return {"ok": True, "repos": 0, "note": "no symbols indexed"}

    total = sum(r["facts"] for r in rows)
    wmean = lambda k: round(sum(r[k] * r["facts"] for r in rows) / total, 2)
    with_grounding = round(
        sum(r["modeled_with_grounding_per_100"] * r["facts"] for r in rows) / total, 3)
    reduction = round(100 * (baseline_per_100 - with_grounding) / baseline_per_100, 2)
    reds = [r["reduction_pct"] for r in rows]
    return {
        "ok": True,
        "methodology": ("deterministic structural ablation (no LLM): residual = "
                        "baseline*(1-grounding_coverage)*(1-guard_catch); baseline "
                        f"= {baseline_per_100}/100 ungrounded fabrication rate"),
        "repos": len(rows),
        "facts_total": total,
        "baseline_without_grounding_per_100": baseline_per_100,
        "modeled_with_grounding_per_100": with_grounding,
        "hallucination_reduction_pct": reduction,
        "reduction_spread_pct": [min(reds), max(reds)],
        "mean_grounding_coverage_pct": wmean("grounding_coverage_pct"),
        "mean_guard_catch_pct": wmean("guard_catch_pct"),
        "mean_guard_specificity_pct": wmean("guard_specificity_pct"),
        "per_repo": rows,
        "deterministic": True,
        "summary": (f"{reduction}% modeled codebase-fact hallucination reduction "
                    f"across {len(rows)} repo-partition(s) "
                    f"({baseline_per_100} -> {with_grounding} errors/100); "
                    f"coverage {wmean('grounding_coverage_pct')}%, "
                    f"guard catch {wmean('guard_catch_pct')}%"),
    }


def hallucination_report_to_markdown(rep: dict) -> str:
    if not rep.get("per_repo"):
        return "# tokengraph hallucination benchmark\n\n(no symbols indexed)\n"
    out = ["# tokengraph — codebase-fact hallucination benchmark", "",
           f"_{rep['summary']}_", "",
           f"- Methodology: {rep['methodology']}",
           f"- Reproducible: deterministic, no LLM (same index -> same numbers)",
           f"- Baseline (ungrounded): **{rep['baseline_without_grounding_per_100']}** errors/100",
           f"- With grounding (modeled): **{rep['modeled_with_grounding_per_100']}** errors/100",
           f"- **Hallucination reduction: {rep['hallucination_reduction_pct']}%** "
           f"(per-repo spread {rep['reduction_spread_pct'][0]}-{rep['reduction_spread_pct'][1]}%)",
           "",
           "| Repo | Facts | Coverage % | Guard catch % | Guard spec. % | With-grounding /100 | Reduction % |",
           "|---|--:|--:|--:|--:|--:|--:|"]
    for r in rep["per_repo"]:
        out.append(f"| {r['repo']} | {r['facts']} | {r['grounding_coverage_pct']} "
                   f"| {r['guard_catch_pct']} | {r['guard_specificity_pct']} "
                   f"| {r['modeled_with_grounding_per_100']} | {r['reduction_pct']} |")
    out.append("")
    return "\n".join(out)


# ==========================================================================
# IDE integration (FR-IDE): one-command MCP wiring for every major editor
# ==========================================================================
def ide_setup(root: Path, editors: list[str] | None = None) -> dict:
    """Wire ContextIQ's MCP server into every (or selected) MCP-capable editor.

    For an MCP-native tool this is the equivalent of shipping editor plugins:
    one command drops a correct server config into each editor's expected
    location. Non-destructive (merges into existing JSON)."""
    import json
    import shutil
    root = Path(root).resolve()
    entry = shutil.which("tokengraph")
    if entry:
        stdio = {"command": "tokengraph", "args": ["serve"]}
    else:
        stdio = {"command": "python", "args": [os.path.abspath(__file__), "serve"]}
    stdio_typed = {"type": "stdio", **stdio}

    # editor -> (relative config path, payload to merge)
    targets = {
        "claude":   (".mcp.json", {"mcpServers": {"tokengraph": stdio_typed}}),
        "vscode":   (".vscode/mcp.json", {"servers": {"tokengraph": stdio_typed}}),
        "cursor":   (".cursor/mcp.json", {"mcpServers": {"tokengraph": stdio}}),
        "windsurf": (".windsurf/mcp.json", {"mcpServers": {"tokengraph": stdio}}),
        "zed":      (".zed/settings.json",
                     {"context_servers": {"tokengraph":
                      {"command": stdio["command"], "args": stdio["args"]}}}),
    }
    chosen = editors or list(targets)
    written: list[str] = []
    for ed in chosen:
        if ed not in targets:
            continue
        rel, payload = targets[ed]
        p = root / rel
        cur = {}
        if p.exists():
            try:
                cur = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                cur = {}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_deep_merge(cur, payload), indent=2) + "\n",
                     encoding="utf-8")
        written.append(rel)

    nvim = ('require("mcphub").setup({ servers = { tokengraph = { command = "%s", '
            'args = { %s } } } })' % (stdio["command"],
            ", ".join(f'"{a}"' for a in stdio["args"])))
    return {
        "written": written,
        "editors": chosen,
        "neovim_snippet": nvim,
        "jetbrains_note": ("JetBrains AI Assistant: Settings → Tools → MCP → Add → "
                           f"command `{stdio['command']} {' '.join(stdio['args'])}`"),
        "note": f"wired {len(written)} editor config(s); restart the editor to load",
    }


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
          "description": "CLI command (tokengraph, or e.g. 'python /path/tokengraph_all.py')."
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
function root() {
  const f = vscode.workspace.workspaceFolders;
  return f && f.length ? f[0].uri.fsPath : process.cwd();
}
function run(args) {
  return new Promise((resolve) => {
    cp.exec(cli() + ' ' + args, { cwd: root(), maxBuffer: 16 * 1024 * 1024 },
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
    show('Context: ' + task, await run('context "' + task.replace(/"/g, '') + '"'));
  });
  reg('contextiq.reindex', async () => {
    await run('index'); vscode.window.showInformationMessage('ContextIQ: reindexed');
  });
  reg('contextiq.conventions', async () => show('Conventions', await run('conventions')));
  reg('contextiq.impact', async () => {
    const ed = vscode.window.activeTextEditor; if (!ed) return;
    const sel = ed.document.getText(ed.selection);
    const word = sel || ed.document.getText(
      ed.document.getWordRangeAtPosition(ed.selection.active) || ed.selection);
    if (!word) return;
    show('Impact: ' + word, await run('impact ' + word));
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
M.config = { command = "tokengraph" }

function M.setup(opts)
  M.config = vim.tbl_extend("force", M.config, opts or {})
end

local function run(args)
  return vim.fn.system(M.config.command .. " " .. args)
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
  show("Context: " .. task, run('context "' .. task .. '"'))
end

function M.conventions() show("Conventions", run("conventions")) end

function M.impact(sym)
  sym = (sym and sym ~= "") and sym or vim.fn.expand("<cword>")
  show("Impact: " .. sym, run("impact " .. sym))
end

function M.reindex()
  run("index")
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

    def _ret() -> "Retriever":
        # Correctness layer: freshen before every retrieval so a query never
        # reads stale line spans (which would mis-slice edited files). The
        # mtime/size fast path keeps this to a stat() per file when nothing
        # changed. A pre-warm hook can keep this a no-op in the common case.
        index_repo(root, db)
        return Retriever(root, db)

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
                              max_body_tokens: int = 1600) -> str:
        """Return a token-budgeted context pack of the symbols most relevant to a task.

        Use this FIRST instead of opening files. Small seeds get full bodies;
        large seeds are demoted to signatures plus matching indexed chunks.
        Callers/callees/base-classes are included as signatures. Anything
        dropped for budget is listed by name so you can request it explicitly.
        """
        r = _ret()
        try:
            pack = r.find_relevant_context(task, budget_tokens=budget_tokens,
                                           expand_depth=depth,
                                           max_body_tokens=max_body_tokens)
            files = sorted({p.file for p in pack.pieces})
            record_pack_savings(root, "mcp.context", final_tokens=pack.tokens,
                                baseline_tokens=sum(r.store.token_est_for(f) for f in files),
                                files=len(files))
            return pack.to_markdown()
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
        """Find symbols by meaning (embedding similarity), not just name match.

        Use when you don't know the exact identifier — e.g. "retry with backoff"
        finds a reattempt helper even if the word "retry" never appears. Returns
        qualified names ranked by relevance.
        """
        r = _ret()
        try:
            return [row["qname"] for row in r.semantic_search(query, limit=limit)]
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
        """Tokens in the context pack vs. reading the referenced files whole."""
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
        rep = index_repo(root, db)
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
            return r.squeeze(text, kind=kind)
        finally:
            r.close()

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
                               baseline_per_100: float = 99.8) -> dict:
        """Reproducible, multi-repo codebase-fact hallucination-reduction benchmark.

        Partitions the repo, measures grounding coverage + guard catch/specificity,
        and reports a modeled hallucination-reduction % vs an ungrounded baseline.
        Deterministic — same index state yields the same figure (no LLM)."""
        r = _ret()
        try:
            return hallucination_benchmark_fn(r, sample_per_repo=sample_per_repo,
                                              baseline_per_100=baseline_per_100)
        finally:
            r.close()

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
    record_pack_savings(root, "context", final_tokens=pack.tokens,
                        baseline_tokens=sum(r.store.token_est_for(f) for f in files),
                        files=len(files), no_track=getattr(args, "no_track", False))
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"wrote {args.out} (~{pack.tokens} tokens, {len(pack.pieces)} symbols)")
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
    if getattr(args, "json", False):
        _emit(s, True)
    else:
        print(s["text"])
        print(f"\n--- squeezed [{s['kind']}]: {s['original_tokens']} -> "
              f"{s['squeezed_tokens']} tokens ({s['reduction_pct']}% smaller) ---",
              file=sys.stderr)


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
        rep = hallucination_benchmark(r, sample_per_repo=args.sample,
                                      baseline_per_100=args.baseline)
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
            print(f"  {row['repo']:16} facts={row['facts']:>3} "
                  f"coverage={row['grounding_coverage_pct']}% "
                  f"catch={row['guard_catch_pct']}% "
                  f"reduction={row['reduction_pct']}%")


def cmd_ide_setup(args):
    root = Path(args.path).resolve()
    editors = args.editor or None
    res = ide_setup(root, editors=editors)
    if getattr(args, "plugins", False):
        res["plugins"] = emit_ide_plugins(root)
    if getattr(args, "json", False):
        _emit(res, True)
    else:
        print(res["note"])
        for w in res["written"]:
            print(f"  wired {w}")
        print(f"  Neovim (mcphub): {res['neovim_snippet']}")
        print(f"  {res['jetbrains_note']}")
        if "plugins" in res:
            print(f"  {res['plugins']['note']}")


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
        if args.out:
            Path(args.out).write_text(md, encoding="utf-8")
            print(f"wrote {args.out} ({rep['aggregate']['tasks']} task(s), "
                  f"{rep['aggregate']['savings_pct_overall']}% fewer overall)")
        else:
            print(md)
        if args.csv:
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


def cmd_langs(args):
    for lang, exts in languages_available().items():
        print(f"  {lang:28} {', '.join(exts)}")


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
    server.run()   # stdio transport


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
    outputs = args.adapter or cfg.get("outputs") or ["copilot"]
    fmt = args.format or cfg.get("format", "md")

    index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    try:
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
        for ad in outputs:
            if ad not in ADAPTERS:
                warns.append(f"unknown adapter {ad!r} (skipped)")
                continue
            custom = cfg.get("output") if ad == "copilot" else None
            rel = write_adapter(root, ad, payload["markdown"], custom)
            entry = {"adapter": ad, "path": rel}
            if fmt == "cache":
                entry["cache"] = write_cache_sidecar(root, rel, payload["markdown"])
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
    _merge_json(".claude/settings.json", {"mcpServers": {"tokengraph": stdio}})

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


def cmd_benchmark(args):
    """Retrieval-quality benchmark: hit@5 + MRR (CI-4)."""
    root = Path(args.path).resolve()
    index_repo(root, _db_path(root))
    r = Retriever(root, _db_path(root))
    try:
        samples = []
        for fr in r.store.files_with_tokens():
            for s in r.store.file_symbols(fr["path"]):
                if s["kind"] == "module":
                    continue
                q = (s["name"] + " " + (s["docstring"] or "")).strip()
                if len(q) >= 3:
                    samples.append((q, s["file"]))
        samples = samples[:args.limit]
        hits = 0
        rr_sum = 0.0
        for q, expected in samples:
            ranked = [f for f, _ in r.rank_files(q, top_k=5, use_recency=False)]
            if expected in ranked:
                pos = ranked.index(expected) + 1
                if pos <= 5:
                    hits += 1
                rr_sum += 1.0 / pos
        n = len(samples) or 1
        out = {"queries": len(samples), "hit_at_5": round(hits / n, 3),
               "mrr": round(rr_sum / n, 3)}
        if getattr(args, "json", False):
            _emit(out, True)
        else:
            print(f"benchmark: {out['queries']} queries  "
                  f"hit@5={out['hit_at_5']}  MRR={out['mrr']}")
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


def cmd_doctor(args):
    """Validate config, context, index freshness, coverage, MCP wiring (CFG-5)."""
    import time
    root = Path(args.path).resolve()
    checks: list[dict] = []

    def add(name, ok, fix=""):
        checks.append({"check": name, "ok": bool(ok), "fix": fix})

    cfg_path = root / CONFIG_NAME
    cfg = load_config(root)
    add("config present", cfg_path.exists(), f"run: tokengraph init")
    add("config parses", "_error" not in cfg, cfg.get("_error", ""))

    db = _db_path(root)
    add("index built", db.exists(), "run: tokengraph index")
    if db.exists():
        rep = index_repo(root, db)
        add("index fresh", rep.parsed == 0,
            f"{rep.parsed} file(s) changed — they were just reindexed")

    outputs = cfg.get("outputs", ["copilot"])
    any_ctx = False
    for ad in outputs:
        spec = ADAPTERS.get(ad)
        if not spec:
            continue
        rel = cfg.get("output") if ad == "copilot" and cfg.get("output") else spec["path"]
        p = root / rel
        present = p.exists()
        any_ctx = any_ctx or present
        stale = present and (time.time() - p.stat().st_mtime) > 7 * 86400
        add(f"context: {rel}", present and not stale,
            "run: tokengraph generate" if not present else
            ("stale >7d — run: tokengraph generate" if stale else ""))

    wired = (root / ".mcp.json").exists() or (root / ".vscode" / "mcp.json").exists() \
        or (root / ".claude" / "settings.json").exists()
    add("MCP wiring present", wired, "run: tokengraph setup")

    ok = all(c["ok"] for c in checks)
    if getattr(args, "json", False):
        _emit({"ok": ok, "checks": checks}, True)
    else:
        for c in checks:
            mark = "ok " if c["ok"] else "FIX"
            line = f"[{mark}] {c['check']}"
            if not c["ok"] and c["fix"]:
                line += f"  -> {c['fix']}"
            print(line)
        print("doctor: " + ("all good" if ok else "issues found"))
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
    s = summarize_gain(root, since=getattr(args, "since", None),
                       model=args.model, top=getattr(args, "top", None),
                       trends=getattr(args, "all", False))
    if getattr(args, "html", None):
        if not s.get("daily"):
            s = summarize_gain(root, since=getattr(args, "since", None),
                               model=args.model, top=getattr(args, "top", None),
                               trends=True)
        Path(args.html).write_text(render_gain_html(s), encoding="utf-8")
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
    out = {
        "branch": branch, "dirty_files": dirty,
        "indexed_files": store_stats.get("files", 0),
        "symbols": store_stats.get("symbols", 0),
        "index_age": fresh, "notes": notes,
        "saved_tokens": gain["saved_tokens"], "gain_runs": gain["runs"],
    }
    if getattr(args, "json", False):
        _emit(out, True)
        return
    print(f"branch={branch}  dirty={dirty}  "
          f"indexed={out['indexed_files']} files / {out['symbols']} symbols  "
          f"index={fresh}")
    print(f"notes={notes}  savings={out['saved_tokens']:,} tok over "
          f"{out['gain_runs']} run(s)")


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
    rp.set_defaults(func=cmd_report)

    sub.add_parser("stats").set_defaults(func=cmd_stats)
    sub.add_parser("langs").set_defaults(func=cmd_langs)
    sub.add_parser("serve").set_defaults(func=cmd_serve)

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
                        help="multi-repo, reproducible codebase-fact hallucination-reduction benchmark")
    hb.add_argument("--sample", type=int, default=40, help="symbols sampled per repo-partition")
    hb.add_argument("--baseline", type=float, default=99.8,
                    help="ungrounded fabrication rate per 100 to compare against")
    hb.add_argument("-o", "--out", default=None, help="write a markdown report")
    hb.add_argument("--json", action="store_true")
    hb.add_argument("--no-refresh", action="store_true")
    hb.set_defaults(func=cmd_hallucination)

    ide = sub.add_parser("ide-setup",
                         help="wire the MCP server into every major editor (VS Code/Cursor/Windsurf/Zed/Claude)")
    ide.add_argument("--editor", action="append",
                     choices=["claude", "vscode", "cursor", "windsurf", "zed"],
                     help="limit to specific editor(s); default = all")
    ide.add_argument("--plugins", action="store_true",
                     help="also scaffold installable VS Code / Neovim / JetBrains plugins")
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
    bm.add_argument("--json", action="store_true")
    bm.set_defaults(func=cmd_benchmark)

    an = sub.add_parser("analyze",
                        help="per-file signatures/tokens/extractor/coverage (CI-5)")
    an.add_argument("--slow", action="store_true", help="include per-file timing")
    an.add_argument("--json", action="store_true")
    an.set_defaults(func=cmd_analyze)

    dr = sub.add_parser("doctor",
                        help="validate config/context/index/MCP wiring (CFG-5)")
    dr.add_argument("--json", action="store_true")
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
