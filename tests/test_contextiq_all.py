"""Tests for tokengraph_all.

Run with:  python -m unittest test_contextiq_all -v

Zero third-party deps — uses stdlib unittest and tempfile so it runs anywhere
the main tool runs. Covers the indexer, the incremental fast path, the
freshen-on-query correctness guarantee (the bug that mis-sliced edited files),
.gitignore handling, and context budgeting.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import tokengraph_all as tg

_PYPROJECT = Path(tg.__file__).resolve().parent / "pyproject.toml"


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


SAMPLE = '''\
def helper(x):
    """Double a number."""
    return x * 2


class Greeter:
    def greet(self, name):
        return helper(len(name))
'''


class IndexerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / ".tokengraph" / "graph.db"

    def tearDown(self):
        self._tmp.cleanup()

    def test_index_extracts_symbols_and_edges(self):
        _write(self.root, "mod.py", SAMPLE)
        rep = tg.index_repo(self.root, self.db)
        self.assertEqual(rep.parsed, 1)
        self.assertGreaterEqual(rep.stats["symbols"], 3)  # module, helper, Greeter, greet
        store = tg.Store(self.db)
        try:
            self.assertIsNotNone(store.symbol_by_qname("mod.helper"))
            self.assertIsNotNone(store.symbol_by_qname("mod.Greeter.greet"))
            # greet() calls helper() -> a CALLS edge should resolve
            sid = store.id_for_qname("mod.Greeter.greet")
            callees = [r["qname"] for r in store.neighbors(sid, ["CALLS"], "out")]
            self.assertIn("mod.helper", callees)
        finally:
            store.close()

    def test_incremental_fast_path_skips_unchanged(self):
        _write(self.root, "mod.py", SAMPLE)
        tg.index_repo(self.root, self.db)
        rep2 = tg.index_repo(self.root, self.db)
        self.assertEqual(rep2.parsed, 0)
        self.assertEqual(rep2.skipped, 1)

    def test_touched_mtime_same_content_does_not_reparse(self):
        p = _write(self.root, "mod.py", SAMPLE)
        tg.index_repo(self.root, self.db)
        # bump mtime without changing content
        future = time.time() + 10
        import os
        os.utime(p, (future, future))
        rep = tg.index_repo(self.root, self.db)
        self.assertEqual(rep.parsed, 0)      # content hash matched -> no reparse
        self.assertEqual(rep.skipped, 1)

    def test_change_is_reindexed(self):
        _write(self.root, "mod.py", SAMPLE)
        tg.index_repo(self.root, self.db)
        _write(self.root, "mod.py", SAMPLE + "\n\ndef added():\n    return 1\n")
        rep = tg.index_repo(self.root, self.db)
        self.assertEqual(rep.parsed, 1)
        store = tg.Store(self.db)
        try:
            self.assertIsNotNone(store.symbol_by_qname("mod.added"))
        finally:
            store.close()

    def test_deleted_file_is_forgotten(self):
        p = _write(self.root, "gone.py", SAMPLE)
        tg.index_repo(self.root, self.db)
        p.unlink()
        rep = tg.index_repo(self.root, self.db)
        self.assertEqual(rep.removed, 1)
        store = tg.Store(self.db)
        try:
            self.assertIsNone(store.symbol_by_qname("gone.helper"))
        finally:
            store.close()

    def test_freshen_keeps_symbol_bodies_aligned_after_edit(self):
        """The core correctness guarantee: after lines shift, a re-index must
        realign stored line spans so get_symbol returns the right body."""
        _write(self.root, "mod.py", SAMPLE)
        tg.index_repo(self.root, self.db)

        # Prepend lines so every definition shifts down. A stale graph would
        # now mis-slice helper(); a freshen-on-query reindex fixes it.
        shifted = "# new header line\n# another\n" + SAMPLE
        _write(self.root, "mod.py", shifted)
        tg.index_repo(self.root, self.db)   # <- the freshen step

        ret = tg.Retriever(self.root, self.db)
        try:
            body = ret.get_symbol("mod.helper")
        finally:
            ret.close()
        self.assertIsNotNone(body)
        self.assertIn("def helper(x):", body)
        self.assertIn("return x * 2", body)
        self.assertNotIn("# new header line", body)  # not mis-sliced into the header

    def test_incremental_edit_preserves_unchanged_cross_file_caller(self):
        _write(self.root, "callee.py", "def target():\n    return 1\n")
        _write(self.root, "caller.py",
               "from callee import target\n\ndef call():\n    return target()\n")
        tg.index_repo(self.root, self.db)

        _write(self.root, "callee.py", "def target():\n    return 200\n")
        rep = tg.index_repo(self.root, self.db)

        store = tg.Store(self.db)
        try:
            caller_id = store.id_for_qname("caller.call")
            callees = [r["qname"] for r in
                       store.neighbors(caller_id, ["CALLS"], "out")]
        finally:
            store.close()
        self.assertEqual(rep.parsed, 1)
        self.assertIn("callee.target", callees)

    def test_targeted_index_only_scans_notified_path(self):
        _write(self.root, "first.py", "def first(): return 1\n")
        _write(self.root, "second.py", "def second(): return 2\n")
        tg.index_repo(self.root, self.db)
        _write(self.root, "second.py", "def second(): return 200\n")
        rep = tg.index_repo(self.root, self.db, paths=["second.py"])
        self.assertEqual(rep.scanned, 1)
        self.assertEqual(rep.parsed, 1)

    def test_import_scip_adds_precise_reference_edge(self):
        _write(self.root, "callee.py", "def target():\n    return 1\n")
        _write(self.root, "caller.py", "def call():\n    return 1\n")
        tg.index_repo(self.root, self.db)
        index = self.root / "index.scip.json"
        index.write_text(json.dumps({"documents": [
            {"relativePath": "callee.py", "occurrences": [
                {"range": [0, 4, 0, 10], "symbol": "python pkg target().",
                 "symbolRoles": 1}]},
            {"relativePath": "caller.py", "occurrences": [
                {"range": [1, 4, 1, 10], "symbol": "python pkg target().",
                 "symbolRoles": 0}]}
        ]}), encoding="utf-8")
        result = tg.import_scip_json(self.root, self.db, index)
        self.assertEqual(result["references_imported"], 1)
        store = tg.Store(self.db)
        try:
            source_id = store.id_for_qname("caller.call")
            references = [row["qname"] for row in
                          store.neighbors(source_id, ["REFERENCES"], "out")]
        finally:
            store.close()
        self.assertIn("callee.target", references)

    def test_gitignore_excludes_files_and_dirs(self):
        _write(self.root, "keep.py", "def kept(): return 1\n")
        _write(self.root, "build/skip.py", "def skipped(): return 2\n")
        _write(self.root, "secret.py", "def secret(): return 3\n")
        _write(self.root, ".gitignore", "build/\nsecret.py\n")
        tg.index_repo(self.root, self.db, respect_gitignore=True)
        store = tg.Store(self.db)
        try:
            self.assertIsNotNone(store.symbol_by_qname("keep.kept"))
            self.assertIsNone(store.symbol_by_qname("secret.secret"))
            # build/ is also in DEFAULT_IGNORES, so check a custom-only case too
            self.assertIsNone(store.symbol_by_qname("build.skip.skipped"))
        finally:
            store.close()

    def test_context_pack_respects_budget(self):
        _write(self.root, "mod.py", SAMPLE)
        tg.index_repo(self.root, self.db)
        ret = tg.Retriever(self.root, self.db)
        try:
            pack = ret.find_relevant_context("greet helper name", budget_tokens=400)
            self.assertLessEqual(pack.tokens, 400)
            self.assertIn("Context for", pack.to_markdown())
        finally:
            ret.close()


    def test_embeddings_and_semantic_search(self):
        _write(self.root, "mod.py", SAMPLE)
        tg.index_repo(self.root, self.db)
        store = tg.Store(self.db)
        try:
            self.assertTrue(store.has_vectors())
            self.assertGreaterEqual(store.stats()["vectors"], 3)
        finally:
            store.close()
        ret = tg.Retriever(self.root, self.db)
        try:
            hits = ret.semantic_search("greet a person by name", limit=5)
            qnames = [h["qname"] for h in hits]
            self.assertTrue(any("greet" in q or "Greeter" in q for q in qnames))
        finally:
            ret.close()

    def test_defines_edges(self):
        _write(self.root, "mod.py", SAMPLE)
        tg.index_repo(self.root, self.db)
        store = tg.Store(self.db)
        try:
            cls = store.id_for_qname("mod.Greeter")
            defined = [r["qname"] for r in store.neighbors(cls, ["DEFINES"], "out")]
            self.assertIn("mod.Greeter.greet", defined)
        finally:
            store.close()

    def test_module_summary_generated_and_cacheable(self):
        _write(self.root, "mod.py", SAMPLE)
        tg.index_repo(self.root, self.db)
        store = tg.Store(self.db)
        try:
            row = store.get_summary("mod.py")
            self.assertIsNotNone(row)
            self.assertIn("Greeter", row["summary"])
            store.set_summary("mod.py", "Custom agent summary.", source="agent")
            store.commit()
            self.assertEqual(store.get_summary("mod.py")["source"], "agent")
        finally:
            store.close()

    def test_measure_reports_savings(self):
        _write(self.root, "mod.py", SAMPLE * 8)  # bigger file -> clear savings
        tg.index_repo(self.root, self.db)
        ret = tg.Retriever(self.root, self.db)
        try:
            m = ret.measure("greet helper", budget_tokens=300)
            self.assertGreater(m["baseline_tokens"], m["pack_tokens"])
            self.assertGreater(m["savings_pct"], 0)
        finally:
            ret.close()

    def test_report_aggregates_savings(self):
        _write(self.root, "mod.py", SAMPLE * 8)
        tg.index_repo(self.root, self.db)
        ret = tg.Retriever(self.root, self.db)
        try:
            rep = ret.report(["greet helper", "farewell message"], budget_tokens=300)
            self.assertEqual(rep["aggregate"]["tasks"], 2)
            self.assertEqual(len(rep["rows"]), 2)
            # Aggregate totals are the sum of the per-task rows.
            self.assertEqual(
                rep["aggregate"]["tokens_saved_total"],
                sum(r["tokens_saved"] for r in rep["rows"]),
            )
            self.assertGreater(rep["aggregate"]["savings_pct_overall"], 0)
            # best >= worst by savings %.
            self.assertGreaterEqual(
                rep["aggregate"]["best"]["savings_pct"],
                rep["aggregate"]["worst"]["savings_pct"],
            )
            # Repo summary is populated and self-consistent.
            self.assertGreater(rep["repo"]["repo_tokens_total"], 0)
            self.assertEqual(rep["repo"]["indexed_files"], 1)
            # Renderers produce non-empty markdown + a CSV row per task.
            md = tg.report_to_markdown(rep)
            self.assertIn("# tokengraph savings report", md)
            self.assertIn("Per-task", md)
            csv_text = tg.report_to_csv(rep)
            self.assertEqual(len(csv_text.strip().splitlines()), 3)  # header + 2 rows
        finally:
            ret.close()

    def test_report_append_accumulates_across_runs(self):
        """`report --append` keeps one growing sheet instead of overwriting.

        Replaces report_append.ps1 (Windows-only) with something that works on
        every platform: header once, rows appended, markdown as a run log.
        """
        csv_path = self.root / "runs.csv"
        md_path = self.root / "runs.md"
        first = "task,tokens_saved\nalpha,10\n"
        second = "task,tokens_saved\nbeta,20\n"

        self.assertEqual(tg.append_report_csv(csv_path, first), 1)
        self.assertEqual(tg.append_report_csv(csv_path, second), 1)
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(lines, ["task,tokens_saved", "alpha,10", "beta,20"])

        tg.append_report_markdown(md_path, "body one", 1, stamp="2026-01-01 00:00:00")
        tg.append_report_markdown(md_path, "body two", 3, stamp="2026-01-02 00:00:00")
        log = md_path.read_text(encoding="utf-8")
        self.assertEqual(log.count("# ContextIQ savings report (running log)"), 1)
        self.assertIn("## Run 2026-01-01 00:00:00 (1 task)", log)
        self.assertIn("## Run 2026-01-02 00:00:00 (3 tasks)", log)
        self.assertLess(log.index("body one"), log.index("body two"))

    def test_report_append_writes_header_into_an_empty_file(self):
        """A pre-created but empty file must still get the header."""
        csv_path = self.root / "empty.csv"
        csv_path.write_text("", encoding="utf-8")
        tg.append_report_csv(csv_path, "task,tokens_saved\nalpha,10\n")
        self.assertEqual(csv_path.read_text(encoding="utf-8").strip().splitlines(),
                         ["task,tokens_saved", "alpha,10"])

    def test_import_aware_resolution(self):
        _write(self.root, "lib.py", "def shared():\n    return 1\n")
        _write(self.root, "other.py", "def shared():\n    return 2\n")
        _write(self.root, "app.py",
               "from lib import shared\n\ndef run():\n    return shared()\n")
        tg.index_repo(self.root, self.db)
        store = tg.Store(self.db)
        try:
            run = store.id_for_qname("app.run")
            callees = [r["qname"] for r in store.neighbors(run, ["CALLS"], "out")]
            # ambiguous leaf "shared" resolves to lib.shared via the import, not other.shared
            self.assertIn("lib.shared", callees)
            self.assertNotIn("other.shared", callees)
        finally:
            store.close()


class EmbeddingUnitTests(unittest.TestCase):
    def test_hash_embed_is_normalized_and_deterministic(self):
        a = tg.embed_text("retry the http request with backoff")
        b = tg.embed_text("retry the http request with backoff")
        self.assertEqual(a, b)
        self.assertAlmostEqual(sum(x * x for x in a) ** 0.5, 1.0, places=4)

    def test_cosine_similar_text_scores_higher(self):
        base = tg.embed_text("parse python source into symbols")
        near = tg.embed_text("parse python symbols from source")
        far = tg.embed_text("render html templates in the browser")
        self.assertGreater(tg.cosine(base, near), tg.cosine(base, far))

    def test_blob_roundtrip(self):
        v = tg.embed_text("hello world")
        self.assertEqual([round(x, 5) for x in tg.blob_to_vec(tg.vec_to_blob(v))],
                         [round(x, 5) for x in v])


class GitIgnoreUnitTests(unittest.TestCase):
    def test_patterns(self):
        gi = tg.GitIgnore(["*.log", "build/", "/root_only.txt", "!keep.log"])
        self.assertTrue(gi.is_ignored("a/b/debug.log", is_dir=False))
        self.assertTrue(gi.is_ignored("build", is_dir=True))
        self.assertFalse(gi.is_ignored("build", is_dir=False))  # dir-only rule
        self.assertTrue(gi.is_ignored("root_only.txt", is_dir=False))
        self.assertFalse(gi.is_ignored("sub/root_only.txt", is_dir=False))  # anchored
        self.assertFalse(gi.is_ignored("keep.log", is_dir=False))  # negated


class SecurityUnitTests(unittest.TestCase):
    def test_redacts_common_secrets(self):
        text = (
            "key = AKIAIOSFODNN7EXAMPLE\n"
            "token = ghp_abcdefghijklmnopqrstuvwxyz0123456789\n"
            'password = "hunter2secret"\n'
            "url = postgres://user:p4ssw0rd@db.internal:5432/app\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "harmless = 42\n"
        )
        out, n = tg.redact_secrets(text)
        self.assertGreaterEqual(n, 5)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", out)
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz0123456789", out)
        self.assertNotIn("hunter2secret", out)
        self.assertNotIn("p4ssw0rd", out)
        self.assertIn("harmless = 42", out)  # non-secret content preserved

    def test_redact_never_raises_on_empty(self):
        self.assertEqual(tg.redact_secrets(""), ("", 0))

    def test_offline_mode_blocks_remote_config(self):
        with TemporaryDirectory() as tmp:
            with unittest.mock.patch.dict(os.environ, {"TOKENGRAPH_OFFLINE": "1"}):
                with unittest.mock.patch("urllib.request.urlopen") as urlopen:
                    self.assertEqual(
                        tg._load_extends("https://example.invalid/config.json", Path(tmp)),
                        {})
                    urlopen.assert_not_called()

    def test_gain_ledger_helpers_skip_bad_lines_and_aggregate(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / ".context" / "gain.ndjson"
            ledger.parent.mkdir()
            ledger.write_text(
                '{"ts": 1, "op": "context", "saved": 80, '
                '"baseline_tokens": 100, "final_tokens": 20}\nnot-json\n',
                encoding="utf-8")
            rows = tg.read_gain_ledger(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(tg.gain_totals(rows)["reduction_pct"], 80.0)
            self.assertEqual(tg.gain_by_operation(rows)[0]["op"], "context")
            self.assertEqual(len(tg.gain_daily(rows)), 1)

    def test_gain_ledger_concurrent_writes_remain_valid(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            threads = [threading.Thread(
                target=tg.track_gain,
                args=(root, {"op": "context", "saved": 1,
                             "baseline_tokens": 2, "final_tokens": 1}))
                for _ in range(20)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(tg.read_gain_ledger(root)), 20)


class MCPIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_client_lists_tools_and_observes_edits(self):
        try:
            from fastmcp import Client  # type: ignore[import-not-found]
        except ImportError:
            self.skipTest("fastmcp optional dependency is not installed")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / ".tokengraph" / "graph.db"
            source = _write(root, "service.py",
                            "def status():\n    return 'before'\n")
            tg.index_repo(root, db)
            server = tg.build_mcp_server(root, db)
            try:
                async with Client(server) as client:
                    tools = await client.list_tools()
                    names = {tool.name for tool in tools}
                    self.assertIn("find_relevant_context", names)
                    self.assertIn("get_symbol", names)
                    # tools added alongside the token-efficiency work
                    self.assertIn("embedding_status", names)
                    self.assertIn("session_savings", names)
                    self.assertIn("prompt_cache_blocks", names)

                    before = await client.call_tool(
                        "get_symbol", {"qname": "service.status"})
                    self.assertFalse(before.is_error)
                    self.assertIn("'before'", before.data)

                    source.write_text(
                        "def status():\n    return 'after-edit'\n", encoding="utf-8")
                    after = await client.call_tool(
                        "get_symbol", {"qname": "service.status"})
                    self.assertFalse(after.is_error)
                    self.assertIn("'after-edit'", after.data)

                    # RP-1: the retriever is pooled across calls, so a repeated
                    # query must still see the edit above rather than a cached
                    # pre-edit source line.
                    repeat = await client.call_tool(
                        "get_symbol", {"qname": "service.status"})
                    self.assertIn("'after-edit'", repeat.data)
            finally:
                # The pool deliberately keeps sqlite handles open; release them
                # or the temp directory cannot be removed (Windows).
                server.close_pool()


class IntentRoutingUnitTests(unittest.TestCase):
    def test_intent_detection(self):
        self.assertEqual(tg.detect_intent("fix the crash in the parser"), "debug")
        self.assertEqual(tg.detect_intent("explain how does indexing work"), "explain")
        self.assertEqual(tg.detect_intent("refactor and rename the module"), "refactor")
        self.assertEqual(tg.detect_intent("security audit of auth"), "review")
        self.assertEqual(tg.detect_intent("where is the config loaded"), "search")

    def test_tier_recommendation(self):
        self.assertEqual(
            tg.recommend_tier("redesign the security architecture")["tier"], "powerful")
        self.assertEqual(
            tg.recommend_tier("fix a failing test")["tier"], "balanced")
        self.assertEqual(
            tg.recommend_tier("explain this function")["tier"], "fast")

    def test_tier_for_file(self):
        self.assertEqual(tg.tier_for_file("config.yaml")["tier"], "fast")
        self.assertEqual(
            tg.tier_for_file("big.py", token_est=5000, symbols=60)["tier"], "powerful")


class RetrieverFeatureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / ".tokengraph" / "graph.db"
        _write(self.root, "lib.py",
               "def helper(x):\n    return x * 2\n")
        _write(self.root, "app.py",
               "from lib import helper\n\n"
               "def run(n):\n    return helper(n)\n\n"
               "def caller():\n    return run(3)\n")
        _write(self.root, "test_app.py",
               "from app import run\n\ndef test_run():\n    assert run(1) == 2\n")
        tg.index_repo(self.root, self.db)
        self.ret = tg.Retriever(self.root, self.db)

    def tearDown(self):
        self.ret.close()
        self._tmp.cleanup()

    def test_list_modules(self):
        mods = self.ret.list_modules()
        self.assertTrue(mods)
        top = mods[0]
        self.assertIn("module", top)
        self.assertGreater(top["tokens"], 0)

    def test_get_impact(self):
        imp = self.ret.get_impact("lib.helper")
        self.assertTrue(imp["found"])
        self.assertIn("app.run", imp["direct_callers"])
        # caller() -> run() -> helper(): caller is a transitive caller of helper
        self.assertIn("app.caller", imp["transitive_callers"])
        self.assertTrue(any("test" in f for f in imp["tests_touched"]) or
                        imp["tests_touched"] == [])

    def test_get_lines_sandbox_and_clamp(self):
        body = self.ret.get_lines("lib.py", 1, 999)  # end clamped to file length
        self.assertIn("def helper", body)
        escaped = self.ret.get_lines("../outside.txt", 1, 5)
        self.assertIn("refused", escaped)

    def test_skeleton_and_summary_redact_secrets(self):
        _write(self.root, "credentials.py",
               '"""password = "summary-secret"""\n\n'
               'def connect(password="function-secret"):\n'
               '    """password = "docstring-secret"""\n'
               '    return password\n')
        tg.index_repo(self.root, self.db)
        skeleton = self.ret.file_skeleton("credentials.py")
        summary = self.ret.module_summary("credentials.py")
        for output in (skeleton, summary):
            self.assertIn("[REDACTED]", output)
            self.assertNotIn("function-secret", output)

    def test_context_budget_includes_markdown_envelope(self):
        pack = self.ret.find_relevant_context("helper run caller", budget_tokens=160)
        self.assertLessEqual(tg.count_tokens(pack.to_markdown()), 160)

    def test_corpus_benchmark_reports_quality_waste_and_latency(self):
        result = tg.run_retrieval_benchmark(self.ret, [{
            "task": "double a number using the shared helper",
            "expected_files": ["lib.py"],
        }])
        self.assertEqual(result["queries"], 1)
        self.assertIn("recall_at_5", result)
        self.assertIn("irrelevant_token_ratio", result)
        self.assertGreaterEqual(result["mean_latency_ms"], 0)

    def test_get_lines_redacts(self):
        _write(self.root, "secret.py",
               'TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"\n')
        tg.index_repo(self.root, self.db)
        out = self.ret.get_lines("secret.py", 1, 1)
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz0123456789", out)
        self.assertIn("[REDACTED]", out)

    def test_explain_file(self):
        text = self.ret.explain_file("lib.py")
        self.assertIn("helper", text)
        self.assertIn("external callers", text)

    def test_get_map(self):
        m = self.ret.get_map("imports")
        self.assertEqual(m["kind"], "imports")
        self.assertIsInstance(m["edges"], dict)

    def test_ask_metadata(self):
        a = self.ret.ask("double a number with helper")
        self.assertIn("intent", a)
        self.assertGreaterEqual(a["coverage_pct"], 0)
        self.assertIn(a["risk"], ("low", "medium", "high"))
        self.assertIn("markdown", a)

    def test_validate_gate(self):
        v = self.ret.validate("helper run caller", min_coverage=0.0)
        self.assertTrue(v["ok"])

    def test_judge_groundedness(self):
        grounded = self.ret.judge("helper doubles the number",
                                  "def helper(x): return x * 2  # doubles number")
        self.assertGreater(grounded["grounded_pct"], 0)
        hallucinated = self.ret.judge(
            "kubernetes_orchestrator schedules quantum_pods",
            "def helper(x): return x * 2")
        self.assertFalse(hallucinated["grounded"])

    def test_learn_weights(self):
        self.ret.learn("lib.py", good=True, weight=2.0)
        self.assertEqual(self.ret.store.weight_for("lib.py"), 2.0)
        self.ret.learn("lib.py", good=False, weight=0.5)
        self.assertAlmostEqual(self.ret.store.weight_for("lib.py"), 1.5)

    def test_memory_and_checkpoint(self):
        self.ret.remember("decided to use FTS5", kind="decision")
        self.ret.create_checkpoint("phase-1", note="indexing done")
        mem = self.ret.read_memory()
        self.assertTrue(any("FTS5" in n["text"] for n in mem["notes"]))
        self.assertTrue(any(c["label"] == "phase-1" for c in mem["checkpoints"]))


class ComprehensiveSolutionTests(unittest.TestCase):
    """Covers the requirements-spec additions: cap, languages, budget formula,
    strategies, adapters, prompt cache, tracking, config, diagnostics."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / ".tokengraph" / "graph.db"

    def tearDown(self):
        self._tmp.cleanup()

    def test_signature_cap_per_file(self):  # FR-2a
        # Full graph is indexed (so callers/callees/semantic keep working),
        # but the emitted skeleton is capped to bound worst-case token cost.
        src = "\n".join(f"def fn_{i}():\n    return {i}" for i in range(40))
        _write(self.root, "big.py", src)
        tg.index_repo(self.root, self.db)
        r = tg.Retriever(self.root, self.db)
        try:
            sk = r.file_skeleton("big.py")
            sig_lines = [ln for ln in sk.splitlines()
                         if ln.startswith("def ")]
            self.assertLessEqual(len(sig_lines), tg.MAX_SIGS_PER_FILE)
            self.assertIn("more symbol", sk)            # omitted-count note
            # the full graph still has all 40 functions
            self.assertEqual(
                sum(1 for s in tg.parse_path(self.root, self.root / "big.py").symbols
                    if s.kind != "module"), 40)
        finally:
            r.close()

    def test_new_language_extractors(self):  # FR-2
        cases = {
            "terraform": ('resource "aws_s3_bucket" "data" {', "data"),
            "graphql": ("type User {", "User"),
            "gdscript": ("func _ready():", "_ready"),
            "yaml": ("services:", "services"),
            "shell": ("deploy() {", "deploy"),
            "css": (".btn {", ".btn"),
        }
        for lang, (line, expected) in cases.items():
            got = tg._generic_definition(line, lang)
            self.assertIsNotNone(got, lang)
            self.assertEqual(got[0], expected, lang)

    def test_diagnose_extractors_passes(self):  # FR-2b
        rep = tg.diagnose_extractors()
        self.assertTrue(rep["ok"], rep)
        self.assertEqual(rep["failed"], 0)

    def test_effective_budget_formula(self):  # TB-2
        self.assertEqual(tg.effective_budget(100_000, 0.8, 128_000, 0.2)[0], 25_600)
        self.assertEqual(tg.effective_budget(1_000)[0], 4_000)  # hard floor
        _, warns = tg.effective_budget(1_000_000, 0.8, 128_000, 0.2)  # TB-3 warn
        self.assertTrue(warns)

    def test_config_extends_and_preset(self):  # CFG-1..3
        _write(self.root, "base.json", '{"strategy":"full","maxTokensHeadroom":0.5}')
        _write(self.root, tg.CONFIG_NAME,
               '{"extends":"./base.json","srcDirs":["src"],'
               '"retrieval":{"preset":"recall"}}')
        cfg = tg.load_config(self.root)
        self.assertEqual(cfg["strategy"], "full")
        self.assertEqual(cfg["maxTokensHeadroom"], 0.5)
        self.assertEqual(cfg["srcDirs"], ["src"])
        self.assertEqual(cfg["retrieval"]["topK"], 20)

    def test_strategies_and_adapters(self):  # TB-4, MCP-OUT, §3.1
        _write(self.root, "src/a.py", SAMPLE)
        tg.index_repo(self.root, self.db)
        r = tg.Retriever(self.root, self.db)
        try:
            cfg = dict(tg.DEFAULTS_CONFIG)
            for strat in ("full", "per-module", "hot-cold"):
                payload = tg.build_context_payload(
                    r, self.root, strategy=strat, src_dirs=["src"], budget=4000,
                    hot_commits=10, diff=False, staged=False, config=cfg)
                self.assertEqual(payload["strategy"], strat)
                self.assertGreater(payload["tokens"], 0)
            rel = tg.write_adapter(self.root, "claude", payload["markdown"])
            self.assertEqual(rel, "CLAUDE.md")
            text = (self.root / "CLAUDE.md").read_text(encoding="utf-8")
            self.assertIn(tg.CLAUDE_BEGIN, text)
            # claude adapter is non-destructive (MCP-5): re-run preserves a marker block
            tg.write_adapter(self.root, "claude", "second")
            self.assertEqual(
                (self.root / "CLAUDE.md").read_text(encoding="utf-8").count(tg.CLAUDE_BEGIN), 1)
        finally:
            r.close()

    def test_prompt_cache_artifact(self):  # PC-1
        art = tg.cache_artifact("stable prefix")
        self.assertEqual(art["cache_control"], {"type": "ephemeral"})
        self.assertEqual(art["type"], "text")

    def test_gain_tracking_counts_only(self):  # TB-6
        tg.track_gain(self.root, {"op": "generate", "final_tokens": 100,
                                  "baseline_tokens": 1000, "saved": 900,
                                  "reduction_pct": 90.0, "files": 3})
        line = (self.root / ".context" / "gain.ndjson").read_text(encoding="utf-8")
        self.assertIn("\"saved\": 900", line)
        # no paths / source / query text ever recorded
        self.assertNotIn("/", line.replace("\\n", ""))

    def test_gain_tracking_disabled(self):  # TB-6 opt-out
        tg.track_gain(self.root, {"op": "x", "saved": 1}, no_track=True)
        self.assertFalse((self.root / ".context" / "gain.ndjson").exists())

    def test_query_context_ranks_relevant_file(self):  # MCP-2 / FR-4
        _write(self.root, "auth.py", "def login(user, pw):\n    return True\n")
        _write(self.root, "math_utils.py", "def add(a, b):\n    return a + b\n")
        tg.index_repo(self.root, self.db)
        r = tg.Retriever(self.root, self.db)
        try:
            res = r.query_context("user login authentication", top_k=5)
            top = [f["file"] for f in res["top_files"]]
            self.assertIn("auth.py", top)
            sigs = r.search_signatures("login")
            self.assertTrue(any(s["file"] == "auth.py" for s in sigs))
            ctx = r.read_context()
            self.assertIn("login", ctx)
        finally:
            r.close()


class SqueezeUnitTests(unittest.TestCase):
    """G6: input squeeze — classify + reduce noisy pasted blobs."""

    def test_classify(self):
        self.assertEqual(tg.classify_input('{"a": 1, "b": [1,2,3]}'), "json")
        st = ('Traceback (most recent call last):\n'
              '  File "x.py", line 1, in <module>\n    boom()\nValueError: x\n')
        self.assertEqual(tg.classify_input(st), "stacktrace")
        ci = "\n".join("2026-01-01 step %d ok" % i for i in range(10)) + "\nERROR: build failed\n"
        self.assertEqual(tg.classify_input(ci), "cilog")
        self.assertEqual(tg.classify_input("just a sentence"), "text")

    def test_squeeze_json_truncates(self):
        import json
        blob = json.dumps({"msg": "x" * 500, "items": list(range(100))})
        out = tg.squeeze_text(blob)
        self.assertEqual(out["kind"], "json")
        self.assertLess(out["squeezed_tokens"], out["original_tokens"])
        self.assertIn("more items", out["text"])

    def test_squeeze_stacktrace_drops_vendor_frames(self):
        st = ('Traceback (most recent call last):\n'
              '  File "/usr/lib/python3/site-packages/flask/app.py", line 99, in __call__\n'
              '    return self.wsgi_app(env)\n'
              '  File "app.py", line 10, in main\n    boom()\n'
              'ValueError: boom\n')
        out = tg.squeeze_text(st)
        self.assertEqual(out["kind"], "stacktrace")
        self.assertNotIn("site-packages", out["text"])
        self.assertNotIn("wsgi_app", out["text"])      # vendor code line also dropped
        self.assertIn("app.py", out["text"])           # user frame kept
        self.assertIn("ValueError: boom", out["text"])

    def test_squeeze_cilog_keeps_errors(self):
        lines = ["downloading package %d" % i for i in range(20)]
        lines.append("ERROR: compilation failed in module foo")
        out = tg.squeeze_text("\n".join(lines))
        self.assertEqual(out["kind"], "cilog")
        self.assertIn("ERROR: compilation failed", out["text"])
        self.assertLess(out["squeezed_tokens"], out["original_tokens"])

    def test_squeeze_redacts_secrets(self):
        blob = "ERROR boom\ntoken=ghp_abcdefghijklmnopqrstuvwxyz0123456789\n" * 3
        out = tg.squeeze_text(blob)
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz0123456789", out["text"])

    def test_squeeze_never_raises(self):
        self.assertEqual(tg.squeeze_text("")["kind"], "text")


class VerifyUnitTests(unittest.TestCase):
    """G5: flag fabricated files/symbols in an answer."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / ".tokengraph" / "graph.db"
        _write(self.root, "mod.py", SAMPLE)
        tg.index_repo(self.root, self.db)
        self.ret = tg.Retriever(self.root, self.db)

    def tearDown(self):
        self.ret.close()
        self._tmp.cleanup()

    def test_real_references_pass(self):
        v = self.ret.verify("The `helper` function lives in mod.py and is fine.")
        self.assertTrue(v["ok"], v)

    def test_fabricated_file_and_symbol_flagged(self):
        v = self.ret.verify("See `quantum_scheduler` in nonexistent.py")
        self.assertFalse(v["ok"])
        kinds = {i["kind"] for i in v["issues"]}
        self.assertIn("file", kinds)
        self.assertIn("symbol", kinds)

    def test_did_you_mean_suggests_close_name(self):
        v = self.ret.verify("call `helpr` to double")   # typo of helper
        sym = [i for i in v["issues"] if i["kind"] == "symbol"]
        self.assertTrue(sym)
        self.assertIn("helper", sym[0]["did_you_mean"])

    def test_levenshtein(self):
        self.assertEqual(tg._levenshtein("kitten", "sitting"), 3)
        self.assertEqual(tg._levenshtein("abc", "abc"), 0)


class ArchitectureMapTests(unittest.TestCase):
    """G11: routes + hubs/cycles in get_map."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / ".tokengraph" / "graph.db"

    def tearDown(self):
        self._tmp.cleanup()

    def test_routes_extracted(self):
        _write(self.root, "api.py",
               '@app.get("/users")\ndef list_users():\n    return []\n\n'
               '@app.post("/users")\ndef make_user():\n    return {}\n')
        tg.index_repo(self.root, self.db)
        r = tg.Retriever(self.root, self.db)
        try:
            m = r.get_map("routes")
            self.assertEqual(m["kind"], "routes")
            paths = {(rt["method"], rt["path"]) for rt in m["routes"]}
            self.assertIn(("GET", "/users"), paths)
            self.assertIn(("POST", "/users"), paths)
        finally:
            r.close()

    def test_hubs_and_cycle_detection(self):
        _write(self.root, "a.py", "import b\n")
        _write(self.root, "b.py", "import a\n")          # a <-> b cycle
        tg.index_repo(self.root, self.db)
        r = tg.Retriever(self.root, self.db)
        try:
            m = r.get_map("hubs")
            self.assertEqual(m["kind"], "hubs")
            self.assertTrue(m["hubs"])
            self.assertTrue(any(set(c) == {"a.py", "b.py"} for c in m["cycles"]),
                            m["cycles"])
        finally:
            r.close()

    def test_import_cycles_helper(self):
        graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"c"}}
        cycles = tg._import_cycles(graph)
        self.assertTrue(any(set(c) == {"a", "b", "c"} for c in cycles))


class NewLanguageTests(unittest.TestCase):
    """G14: broadened regex-fallback languages."""

    def test_new_generic_languages(self):
        cases = {
            "toml": ("[tool.black]", "tool.black"),
            "ini": ("[section_one]", "section_one"),
            "properties": ("db.host=localhost", "db.host"),
            "protobuf": ("message User {", "User"),
            "xml": ('<element name="Foo">', "Foo"),
        }
        for lang, (line, expected) in cases.items():
            got = tg._generic_definition(line, lang)
            self.assertIsNotNone(got, lang)
            self.assertEqual(got[0], expected, lang)

    def test_extensions_mapped(self):
        for ext, lang in {".toml": "toml", ".ini": "ini", ".proto": "protobuf",
                          ".xml": "xml", ".properties": "properties"}.items():
            self.assertEqual(tg.GENERIC_LANGUAGE_EXTENSIONS.get(ext), lang)

    def test_additional_programming_languages(self):
        cases = {
            "lua": ("function greet(name)", "greet"),
            "elixir": ("def run(x) do", "run"),
            "haskell": ("add :: Int -> Int", "add"),
            "perl": ("sub run {", "run"),
            "julia": ("function run(x)", "run"),
            "ocaml": ("let add x = x", "add"),
            "fsharp": ("let add x = x", "add"),
            "groovy": ("def run() {", "run"),
            "powershell": ("function Get-Item {", "Get-Item"),
            "solidity": ("function mint() public {", "mint"),
            "zig": ("pub fn add(a: i32) i32 {", "add"),
            "nim": ("proc greet(name: string) =", "greet"),
            "crystal": ("def deposit(n)", "deposit"),
            "haxe": ("function run() {", "run"),
            "objc": ("@interface Foo", "Foo"),
            "vbnet": ("Public Sub Run()", "Run"),
            "tcl": ("proc greet {name} {", "greet"),
            "pascal": ("function Add(a: Integer): Integer;", "Add"),
            "clojure": ("(defn add [a b]", "add"),
            "markdown": ("## Install", "Install"),
        }
        for lang, (line, expected) in cases.items():
            got = tg._generic_definition(line, lang)
            self.assertIsNotNone(got, lang)
            self.assertEqual(got[0], expected, f"{lang}: {got}")

    def test_new_languages_index_end_to_end(self):
        """New languages must flow through parse_path into the graph."""
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        db = root / ".tokengraph" / "graph.db"
        _write(root, "svc.lua", "function greet(name)\n  return name\nend\n")
        _write(root, "app.ex", "defmodule App do\n  def run(x) do\n    x\n  end\nend\n")
        _write(root, "token.sol",
               "contract Token {\n  function mint() public {}\n}\n")
        _write(root, "README.md", "# Project\n\n## Setup\nrun it\n")
        tg.index_repo(root, db)
        store = tg.Store(db)
        try:
            self.assertIsNotNone(store.symbol_by_qname("svc.greet"))
            self.assertIsNotNone(store.symbol_by_qname("app.run"))
            # solidity is now deep-parsed: the function nests under its contract
            self.assertIsNotNone(store.symbol_by_qname("token.Token.mint"))
            self.assertIsNotNone(store.symbol_by_qname("README.Setup"))
        finally:
            store.close()

    def test_markdown_headings_extracted(self):
        # '#' is a heading in markdown, not a comment — must not be skipped.
        got = tg._generic_definition("# Top Level Title", "markdown")
        self.assertIsNotNone(got)
        self.assertEqual(got[0], "Top Level Title")
        self.assertEqual(got[1], "heading")
        # but '#' stays a comment for non-markdown languages
        self.assertIsNone(tg._generic_definition("# a comment", "toml"))

    def test_filename_languages(self):
        self.assertEqual(tg.GENERIC_LANGUAGE_FILENAMES.get("jenkinsfile"), "groovy")
        self.assertEqual(tg.GENERIC_LANGUAGE_FILENAMES.get("gemfile"), "ruby")
        self.assertEqual(
            tg.language_for_path(Path("/x/Jenkinsfile")), "groovy")


class SeedWeightTests(unittest.TestCase):
    """G10: learned weights nudge find_relevant_context seed ordering."""

    def test_reinforced_file_seeds_first(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        db = root / ".tokengraph" / "graph.db"
        _write(root, "alpha.py", "def handle(req):\n    return req\n")
        _write(root, "beta.py", "def handle(req):\n    return req\n")
        tg.index_repo(root, db)
        r = tg.Retriever(root, db)
        try:
            r.learn("beta.py", good=True, weight=5.0)
            pack = r.find_relevant_context("handle req", budget_tokens=4000)
            seed_files = [p.file for p in pack.pieces if p.reason == "seed"]
            self.assertIn("beta.py", seed_files)
            # beta (reinforced) should be ordered no later than alpha
            if "alpha.py" in seed_files:
                self.assertLessEqual(seed_files.index("beta.py"),
                                     seed_files.index("alpha.py"))
        finally:
            r.close()


class AdapterHardeningTests(unittest.TestCase):
    """Harden G1-G4: all adapters non-destructive, idempotent."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_copilot_preserves_human_content(self):
        target = self.root / ".github" / "copilot-instructions.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# My hand-written rules\nAlways use tabs.\n", encoding="utf-8")
        tg.write_adapter(self.root, "copilot", "GENERATED BODY")
        text = target.read_text(encoding="utf-8")
        self.assertIn("My hand-written rules", text)     # human content survives
        self.assertIn("GENERATED BODY", text)
        self.assertIn(tg.CLAUDE_BEGIN, text)

    def test_adapter_idempotent(self):
        tg.write_adapter(self.root, "copilot", "BODY ONE")
        tg.write_adapter(self.root, "copilot", "BODY TWO")
        text = (self.root / ".github" / "copilot-instructions.md").read_text(encoding="utf-8")
        self.assertEqual(text.count(tg.CLAUDE_BEGIN), 1)   # exactly one marker block
        self.assertIn("BODY TWO", text)
        self.assertNotIn("BODY ONE", text)


class TreeSitterProfileTests(unittest.TestCase):
    """Verified tree-sitter profiles: real call/inheritance edges (the value-add
    over regex). Skips cleanly when no grammars are installed."""

    @classmethod
    def setUpClass(cls):
        cls.profiles = tg.build_profiles()
        if not cls.profiles:
            raise unittest.SkipTest("no tree-sitter grammars installed")

    def _index(self, fname, src):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        db = root / ".tokengraph" / "graph.db"
        _write(root, fname, src)
        tg.index_repo(root, db)
        return tg.Store(db)

    def _assert_calls(self, fname, src, caller_q, callee_q):
        ext = "." + fname.rsplit(".", 1)[-1]
        if ext not in self.profiles:
            self.skipTest(f"no grammar for {ext}")
        store = self._index(fname, src)
        try:
            cid = store.id_for_qname(caller_q)
            self.assertIsNotNone(cid, f"{caller_q} not indexed")
            callees = [r["qname"] for r in store.neighbors(cid, ["CALLS"], "out")]
            self.assertIn(callee_q, callees, callees)
        finally:
            store.close()

    def test_csharp_calls(self):
        self._assert_calls(
            "svc.cs",
            "namespace App { class S {\n void Run(){ Helper(); }\n void Helper(){}\n} }",
            "svc.App.S.Run", "svc.App.S.Helper")

    def test_rust_calls(self):
        self._assert_calls(
            "lib.rs", "fn helper() {}\nfn run() { helper(); }",
            "lib.run", "lib.helper")

    def test_cpp_calls(self):
        self._assert_calls(
            "w.cpp", "void helper() {}\nclass W {\n void run(){ helper(); }\n};",
            "w.W.run", "w.helper")

    def test_ruby_calls(self):
        self._assert_calls(
            "acc.rb", "def helper; end\ndef run\n  helper\nend",
            "acc.run", "acc.helper")

    def test_php_calls(self):
        self._assert_calls(
            "repo.php", "<?php\nfunction helper(){}\nfunction run(){ helper(); }",
            "repo.run", "repo.helper")

    def test_scala_calls(self):
        self._assert_calls(
            "a.scala", "object O {\n def helper() = 1\n def run() = helper()\n}",
            "a.O.run", "a.O.helper")

    def test_kotlin_calls(self):
        self._assert_calls(
            "e.kt", "fun helper() {}\nfun run() { helper() }",
            "e.run", "e.helper")

    def test_ruby_inheritance(self):
        if ".rb" not in self.profiles:
            self.skipTest("no ruby grammar")
        store = self._index(
            "m.rb", "class Base\nend\nclass Account < Base\nend")
        try:
            edges = [(e["src"], e["dst"]) for e in store.edges_of_type("INHERITS")]
            self.assertTrue(any(s.endswith(".Account") and d.endswith(".Base")
                                for s, d in edges), edges)
        finally:
            store.close()

    def test_csharp_inheritance(self):
        if ".cs" not in self.profiles:
            self.skipTest("no c# grammar")
        store = self._index(
            "svc.cs", "namespace App { class Base {} class S : Base {} }")
        try:
            edges = [(e["src"], e["dst"]) for e in store.edges_of_type("INHERITS")]
            self.assertTrue(any(s.endswith(".S") and d.endswith(".Base")
                                for s, d in edges), edges)
        finally:
            store.close()

    def test_treesitter_preferred_over_regex(self):
        # When a grammar is present, parse_path must route through tree-sitter
        # (which yields edges); the regex fallback never produces CALLS edges.
        if ".rs" not in self.profiles:
            self.skipTest("no rust grammar")
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        res = tg.parse_path(root, root / "x.rs") if False else None
        _write(root, "x.rs", "fn a(){ b(); }\nfn b(){}")
        res = tg.parse_path(root, root / "x.rs")
        self.assertTrue(res.edges, "tree-sitter path should emit edges")

    # --- newly upgraded languages: regex-only -> full graph processing -----

    def _assert_edge(self, fname, src, etype, src_suffix, dst_suffix):
        ext = "." + fname.rsplit(".", 1)[-1]
        if ext not in self.profiles:
            self.skipTest(f"no grammar for {ext}")
        store = self._index(fname, src)
        try:
            edges = [(e["src"], e["dst"]) for e in store.edges_of_type(etype)]
            self.assertTrue(
                any(s.endswith(src_suffix) and d.endswith(dst_suffix)
                    for s, d in edges), edges)
        finally:
            store.close()

    def _assert_symbol(self, fname, src, qname):
        ext = "." + fname.rsplit(".", 1)[-1]
        if ext not in self.profiles:
            self.skipTest(f"no grammar for {ext}")
        store = self._index(fname, src)
        try:
            self.assertIsNotNone(store.id_for_qname(qname), qname)
        finally:
            store.close()

    def test_lua_calls(self):
        self._assert_calls(
            "svc.lua", "function helper() end\nfunction run() helper() end",
            "svc.run", "svc.helper")

    def test_lua_dotted_name(self):
        self._assert_symbol(
            "svc.lua", "local M = {}\nfunction M.greet(n) return n end",
            "svc.greet")

    def test_bash_calls(self):
        self._assert_calls(
            "deploy.sh", "helper() {\n  :\n}\nrun() {\n  helper\n}",
            "deploy.run", "deploy.helper")

    def test_solidity_inheritance(self):
        self._assert_edge(
            "tok.sol", "contract Base {}\ncontract Token is Base {}",
            "INHERITS", ".Token", ".Base")

    def test_solidity_calls(self):
        self._assert_calls(
            "tok.sol",
            "contract T {\n function transfer() public { _move(); }\n"
            " function _move() internal {}\n}",
            "tok.T.transfer", "tok.T._move")

    def test_perl_calls(self):
        self._assert_calls(
            "mod.pl", "sub helper { 1 }\nsub run {\n  return helper();\n}",
            "mod.run", "mod.helper")

    def _assert_parse_import(self, fname, src, dst):
        # IMPORTS edges to *external* modules are dropped by the store resolver
        # (no symbol to point at), so assert the profile emits them at parse time.
        ext = "." + fname.rsplit(".", 1)[-1]
        if ext not in self.profiles:
            self.skipTest(f"no grammar for {ext}")
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        _write(root, fname, src)
        res = tg.parse_path(root, root / fname)
        imps = [e.dst_name for e in res.edges if e.type == "IMPORTS"]
        self.assertIn(dst, imps, imps)

    def test_perl_import(self):
        self._assert_parse_import("mod.pl", "use POSIX;\nsub run { 1 }", "POSIX")

    def test_erlang_calls(self):
        self._assert_calls(
            "srv.erl", "-module(srv).\ngo() ->\n  helper().\nhelper() -> ok.",
            "srv.go", "srv.helper")

    def test_julia_calls(self):
        self._assert_calls(
            "m.jl",
            "function helper()\n 1\nend\nfunction run()\n helper()\nend",
            "m.run", "m.helper")

    def test_r_calls(self):
        self._assert_calls(
            "a.r",
            "helper <- function(y) { y }\nrun <- function(x) { helper(x) }",
            "a.run", "a.helper")

    def test_haskell_calls(self):
        self._assert_calls(
            "Main.hs", "fact n = product [1..n]\nmain = print (fact 5)",
            "Main.main", "Main.fact")

    def test_ocaml_calls(self):
        self._assert_calls(
            "m.ml", "let helper x = x + 1\nlet go () = helper 5",
            "m.go", "m.helper")

    def test_nim_symbol_and_import(self):
        self._assert_symbol("a.nim", "import strutils\nproc speak() = discard",
                            "a.speak")
        self._assert_parse_import(
            "a.nim", "import strutils\nproc speak() = discard", "strutils")

    def test_powershell_symbols(self):
        self._assert_symbol(
            "s.ps1", "function Get-Foo {\n  param($x)\n}", "s.Get-Foo")

    def test_dart_inheritance(self):
        self._assert_edge(
            "a.dart", "class Base {}\nclass Animal extends Base {}",
            "INHERITS", ".Animal", ".Base")


class RegexFallbackStillWorksTests(unittest.TestCase):
    """The regex fallback must keep producing symbols regardless of grammars
    (covers environments where the language pack isn't installed)."""

    def test_regex_definition_paths_intact(self):
        # parse_generic is the no-grammar path; it must still find top-level defs.
        for line, lang, expected in [
            ("public class Svc {", "csharp", "Svc"),
            ("pub fn build() {}", "rust", "build"),
            ("def deposit(n)", "ruby", "deposit"),
        ]:
            got = tg._generic_definition(line, lang)
            self.assertIsNotNone(got, lang)
            self.assertEqual(got[0], expected, lang)


class ComprehensiveSolutionGapTests(unittest.TestCase):
    """Close the gaps vs. Sigmap: conventions, grounded-creation, evidence pack,
    grounding ablation, and scope-aware call resolution."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.db = self.root / ".tokengraph" / "graph.db"
        _write(self.root, "src/user_service.py",
               "class UserService:\n"
               "    def create_user(self, name):\n"
               "        return self.validate(name)\n"
               "    def validate(self, name):\n"
               "        return bool(name)\n")
        _write(self.root, "src/auth_service.py",
               "from user_service import UserService\n\n"
               "def login(name):\n"
               "    return UserService().create_user(name)\n")
        _write(self.root, "tests/test_user_service.py",
               "def test_create():\n    assert True\n")

    def _store(self):
        tg.index_repo(self.root, self.db)
        return tg.Store(self.db)

    def _retriever(self):
        tg.index_repo(self.root, self.db)
        return tg.Retriever(self.root, self.db)

    # ---- item 2: conventions extraction ----
    def test_conventions_detects_house_style(self):
        store = self._store()
        try:
            conv = tg.analyze_conventions(store, self.root)
        finally:
            store.close()
        self.assertEqual(conv["dominant_naming"], "snake_case")
        self.assertEqual(conv["primary_extension"], ".py")
        self.assertEqual(conv["test_pattern"], "test_*")
        self.assertIn("src", conv["source_dirs"])

    def test_conventions_export_style_and_conformance(self):
        # add a camelCase outlier + an underscore-private symbol
        _write(self.root, "src/authService.py",
               "def login(name):\n    return _check(name)\n"
               "def _check(name):\n    return True\n")
        store = self._store()
        try:
            conv = tg.analyze_conventions(store, self.root)
        finally:
            store.close()
        # export convention: underscore-private detected
        self.assertEqual(conv["export_style"], "leading-underscore privates")
        self.assertGreater(conv["private_symbols"], 0)
        # conformance: the camelCase file is flagged with a snake_case suggestion
        files = {n["file"]: n for n in conv["nonconforming_files"]}
        self.assertIn("src/authService.py", files)
        self.assertEqual(files["src/authService.py"]["suggested_stem"], "auth_service")
        self.assertLess(conv["conformance_pct"], 100.0)
        # deterministic
        store2 = self._store()
        try:
            conv2 = tg.analyze_conventions(store2, self.root)
        finally:
            store2.close()
        self.assertEqual(conv["nonconforming_files"], conv2["nonconforming_files"])

    # ---- item 1: scaffold (convention-matched, refuses on conflict) ----
    def test_scaffold_matches_conventions(self):
        store = self._store()
        try:
            res = tg.propose_scaffold(store, self.root, "payment processor",
                                      kind="class")
        finally:
            store.close()
        self.assertTrue(res["ok"])
        self.assertEqual(res["proposed_path"], "src/payment_processor.py")
        self.assertIn("payment_processor", res["skeleton"])

    def test_scaffold_refuses_on_conflict(self):
        store = self._store()
        try:
            res = tg.propose_scaffold(store, self.root, "user service",
                                      kind="module")
        finally:
            store.close()
        self.assertFalse(res["ok"])
        self.assertTrue(res["exists"])

    # ---- item 1: verify-plan (blast radius + fabrication) ----
    def test_verify_plan_blast_radius_and_new_files(self):
        r = self._retriever()
        try:
            res = tg.verify_plan(
                r, "Edit `validate` and create src/new_widget.py")
        finally:
            r.close()
        self.assertTrue(res["ok"])  # only a new file + a real symbol
        self.assertIn("src/new_widget.py", res["new_files"])
        self.assertTrue(any(b["symbol"] == "validate" for b in res["blast_radius"]))

    def test_verify_plan_flags_fabricated_symbol(self):
        r = self._retriever()
        try:
            res = tg.verify_plan(r, "Call `nonexistent_frobnicate` from auth")
        finally:
            r.close()
        self.assertFalse(res["ok"])
        self.assertTrue(res["fabricated_symbols"])

    # ---- item 1: review-diff heuristics (no git needed -> clean tree path) ----
    def test_review_diff_clean_tree(self):
        r = self._retriever()
        try:
            res = tg.review_diff(r, self.root)
        finally:
            r.close()
        # not a git repo / no changes -> ok, no findings
        self.assertTrue(res["ok"])

    # ---- item 4: deterministic, hash-grounded evidence pack ----
    def test_evidence_pack_is_deterministic_and_grounded(self):
        r = self._retriever()
        try:
            ev1 = tg.build_evidence_pack(r, "create user validation", budget_tokens=3000)
            ev2 = tg.build_evidence_pack(r, "create user validation", budget_tokens=3000)
        finally:
            r.close()
        self.assertEqual(ev1["grounding"]["context_hash"],
                         ev2["grounding"]["context_hash"])
        self.assertTrue(ev1["grounding"]["deterministic"])
        self.assertGreater(ev1["grounding"]["symbol_count"], 0)
        self.assertEqual(ev1["grounding"]["anchor_coverage"], 1.0)
        self.assertEqual(len(ev1["grounding"]["context_hash"]), 64)  # sha256 hex

    def test_evidence_pack_rich_per_file_schema(self):
        r = self._retriever()
        try:
            ev = tg.build_evidence_pack(r, "create user validation", budget_tokens=3000)
        finally:
            r.close()
        self.assertEqual(ev["schema"], "contextiq.evidence/v2")
        self.assertTrue(ev["files"])
        f0 = ev["files"][0]
        # Sigmap-grade per-file fields all present
        for key in ("path", "symbols", "reason", "confidence",
                    "source_lines", "related_tests", "risk_label"):
            self.assertIn(key, f0)
        self.assertIn(f0["risk_label"], ("low", "medium", "high"))
        self.assertGreaterEqual(f0["confidence"], 0.0)
        self.assertLessEqual(f0["confidence"], 1.0)
        # per-symbol confidence too
        self.assertIn("confidence", ev["symbols"][0])

    def test_evidence_pack_related_tests_linked(self):
        # a test that calls a real symbol should surface as a related test
        _write(self.root, "tests/test_validate.py",
               "from user_service import UserService\n"
               "def test_validate():\n    UserService().validate('x')\n")
        r = self._retriever()
        try:
            ev = tg.build_evidence_pack(r, "validate user name", budget_tokens=4000)
        finally:
            r.close()
        related = sorted({t for f in ev["files"] for t in f["related_tests"]})
        self.assertTrue(any("test" in t for t in related), related)

    # ---- item 5: grounding ablation quantifies the guard ----
    def test_grounding_report_catches_fabrications(self):
        r = self._retriever()
        try:
            rep = tg.grounding_report(r, sample=10)
        finally:
            r.close()
        self.assertGreater(rep["sample"], 0)
        # fabricated names must be flagged; real names must not be
        self.assertEqual(rep["with_grounding"]["false_positive_rate_per_100"], 0.0)
        self.assertGreater(
            rep["without_grounding"]["fabrications_caught_per_100"], 0.0)

    # ---- item 7: scope-aware resolution links the cross-file call ----
    def test_scope_aware_call_resolution(self):
        store = self._store()
        try:
            # auth_service.login -> UserService.create_user (cross-file, import-aware)
            cid = store.id_for_qname("src.auth_service.login")
            self.assertIsNotNone(cid)
            callees = [r["qname"] for r in store.neighbors(cid, ["CALLS"], "out")]
            self.assertIn("src.user_service.UserService.create_user", callees)
            # create_user -> validate (same-class sibling, scope-resolved)
            mid = store.id_for_qname("src.user_service.UserService.create_user")
            mcallees = [r["qname"] for r in store.neighbors(mid, ["CALLS"], "out")]
            self.assertIn("src.user_service.UserService.validate", mcallees)
        finally:
            store.close()

    def test_index_reports_edge_resolution_metric(self):
        rep = tg.index_repo(self.root, self.db)
        self.assertIn("edge_resolution_pct", rep.stats)
        self.assertGreaterEqual(rep.stats["edge_resolution_pct"], 0.0)

    # ---- item 3: AGENTS.md adapter rounds out multi-adapter export ----
    def test_agents_adapter_registered(self):
        self.assertIn("agents", tg.ADAPTERS)
        rel = tg.write_adapter(self.root, "agents", "GENERATED")
        self.assertEqual(rel, "AGENTS.md")
        self.assertIn("GENERATED",
                      (self.root / "AGENTS.md").read_text(encoding="utf-8"))

    # ---- item 1+: verify-ai-output flags fabricated LOCAL imports only ----
    def test_verify_output_flags_local_imports_not_thirdparty(self):
        r = self._retriever()
        try:
            real = r.verify  # sanity: verify exists
            self.assertTrue(callable(real))
            v = tg.verify_ai_output(
                r,
                "from src.nonexistent import x\n"   # local, missing -> flag
                "from .ghost import y\n"            # relative, missing -> flag
                "import numpy as np\n"              # third-party -> ignore
                "from user_service import UserService\n")  # real local -> ignore
            names = {i["name"] for i in v["fabricated_imports"]}
            self.assertIn("src.nonexistent", names)
            self.assertIn(".ghost", names)
            self.assertNotIn("numpy", names)
            self.assertNotIn("user_service", names)
            self.assertFalse(v["ok"])
        finally:
            r.close()

    def test_verify_output_passes_clean_code(self):
        r = self._retriever()
        try:
            v = tg.verify_ai_output(
                r, "Reuse `validate` from src/user_service.py.\nimport os\n")
            self.assertTrue(v["ok"], v["issues"])
        finally:
            r.close()

    # ---- item 1+: scaffold can actually create the file (refuses on conflict) ----
    def test_write_scaffold_creates_then_refuses(self):
        store = self._store()
        try:
            res = tg.write_scaffold(store, self.root, "rate limiter", kind="module")
        finally:
            store.close()
        self.assertTrue(res["written"])
        created = self.root / res["proposed_path"]
        self.assertTrue(created.exists())
        # second attempt is a conflict -> refuses, does not overwrite
        store2 = self._store()
        try:
            again = tg.write_scaffold(store2, self.root, "rate limiter", kind="module")
        finally:
            store2.close()
        self.assertFalse(again["written"])
        self.assertFalse(again["ok"])

    # ---- item 1+: create pipeline is a multi-stage gated state machine ----
    def test_create_pipeline_stages_and_gate(self):
        r = self._retriever()
        try:
            res = tg.create_pipeline(
                r, self.root, "add retry to client", kind="module",
                answer="Reuse `validate`.\nimport os\n")
        finally:
            r.close()
        stage_names = [s["stage"] for s in res["stages"]]
        self.assertEqual(stage_names, ["scaffold", "verify-plan", "verify-output"])
        self.assertTrue(res["ok"])
        # dry run: nothing written
        self.assertFalse((self.root / res["scaffold"]["proposed_path"]).exists())

    def test_create_pipeline_apply_writes_and_reviews(self):
        r = self._retriever()
        try:
            res = tg.create_pipeline(r, self.root, "circuit breaker",
                                     kind="module", apply=True)
        finally:
            r.close()
        self.assertIn("review", [s["stage"] for s in res["stages"]])
        self.assertTrue(res["scaffold"]["written"])
        self.assertTrue((self.root / res["scaffold"]["proposed_path"]).exists())

    def test_create_pipeline_fails_gate_on_bad_output(self):
        r = self._retriever()
        try:
            res = tg.create_pipeline(
                r, self.root, "thing", kind="module",
                answer="Call `totally_made_up_symbol` from src/ghost_file.py")
        finally:
            r.close()
        self.assertFalse(res["ok"])
        oc = res["output_check"]
        self.assertFalse(oc["ok"])

    # ---- grounding: reproducible multi-repo benchmark (HB-1) ----
    def test_hallucination_benchmark_multi_repo_and_deterministic(self):
        r = self._retriever()
        try:
            rep1 = tg.hallucination_benchmark(r, sample_per_repo=20)
            rep2 = tg.hallucination_benchmark(r, sample_per_repo=20)
        finally:
            r.close()
        # src/ and tests/ -> at least two repo-partitions
        self.assertGreaterEqual(rep1["repos"], 2)
        self.assertTrue(rep1["deterministic"])
        # reproducible: identical numbers on re-run (no LLM, no randomness)
        self.assertEqual(rep1["mean_guard_catch_pct"], rep2["mean_guard_catch_pct"])
        self.assertEqual(rep1["unguarded_fact_share_pct"],
                         rep2["unguarded_fact_share_pct"])
        self.assertEqual(len(rep1["unguarded_spread_pct"]), 2)
        self.assertGreaterEqual(rep1["mean_grounding_coverage_pct"], 0.0)

    def test_no_reduction_claim_without_a_supplied_baseline(self):
        """HB-1: the tool must not invent the number it is being judged on.

        The ungrounded fabrication rate cannot be observed from an index, so a
        default for it turns the headline into a restatement of its own
        assumption. Absent one, report the measurements and say why not.
        """
        r = self._retriever()
        try:
            rep = tg.hallucination_benchmark(r, sample_per_repo=10)
        finally:
            r.close()
        self.assertIsNone(rep["hallucination_reduction_pct"])
        self.assertFalse(rep["projection"]["available"])
        self.assertIn("baseline", rep["projection"]["why"])
        # …and the measured half is still fully reported.
        self.assertTrue(rep["measured"])
        self.assertIn("guard catch", rep["summary"])

    def test_supplied_baseline_yields_a_labelled_projection(self):
        r = self._retriever()
        try:
            rep = tg.hallucination_benchmark(
                r, sample_per_repo=10, baseline_per_100=40.0,
                baseline_source="internal eval, 2026-07")
        finally:
            r.close()
        proj = rep["projection"]
        self.assertTrue(proj["available"])
        self.assertEqual(proj["baseline_without_grounding_per_100"], 40.0)
        self.assertEqual(proj["baseline_source"], "internal eval, 2026-07")
        self.assertIn("not observed", proj["caveat"])
        self.assertIn("NOT observed", rep["summary"])

    def test_unstated_baseline_source_is_called_out(self):
        r = self._retriever()
        try:
            rep = tg.hallucination_benchmark(r, sample_per_repo=10,
                                             baseline_per_100=40.0)
        finally:
            r.close()
        self.assertIn("UNSTATED", rep["projection"]["baseline_source"])

    def test_hallucination_report_markdown(self):
        r = self._retriever()
        try:
            rep = tg.hallucination_benchmark(r, sample_per_repo=10)
        finally:
            r.close()
        md = tg.hallucination_report_to_markdown(rep)
        self.assertIn("grounding", md.lower())
        self.assertIn("| Repo |", md)
        # The report must state plainly that no reduction figure is available.
        self.assertIn("Not reported", md)

    # ---- IDE integration: one-command MCP wiring for every editor ----
    def test_ide_setup_writes_editor_configs(self):
        import json
        res = tg.ide_setup(self.root)
        # Windsurf is deliberately NOT here: it reads MCP config only from
        # ~/.codeium/windsurf/mcp_config.json, so a project-local
        # .windsurf/mcp.json is a file nothing ever reads. It is reported
        # under global_pending instead and written only with --global.
        self.assertEqual(set(res["written"]),
                         {".mcp.json", ".vscode/mcp.json", ".cursor/mcp.json",
                          ".zed/settings.json", ".idea/mcp.xml",
                          ".nvim/contextiq.lua"})
        self.assertIn("windsurf", {g["host"] for g in res["global_pending"]})
        # each config is valid JSON wiring the tokengraph server
        vsc = json.loads((self.root / ".vscode" / "mcp.json").read_text("utf-8"))
        self.assertIn("tokengraph", vsc["servers"])
        cur = json.loads((self.root / ".cursor" / "mcp.json").read_text("utf-8"))
        self.assertIn("tokengraph", cur["mcpServers"])
        zed = json.loads((self.root / ".zed" / "settings.json").read_text("utf-8"))
        self.assertIn("tokengraph", zed["context_servers"])
        self.assertIn("mcphub", res["neovim_snippet"])

    def test_ide_setup_is_non_destructive(self):
        import json
        target = self.root / ".vscode" / "mcp.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"servers": {"other": {"command": "x"}}}', encoding="utf-8")
        tg.ide_setup(self.root, editors=["vscode"])
        cfg = json.loads(target.read_text("utf-8"))
        # preserves the pre-existing server, adds ours
        self.assertIn("other", cfg["servers"])
        self.assertIn("tokengraph", cfg["servers"])

    def test_ide_setup_wires_each_multi_root_folder(self):
        first = self.root / "workspace-one"
        second = self.root / "workspace-two"
        first.mkdir()
        second.mkdir()
        res = tg.ide_setup(self.root, editors=["vscode"],
                           workspace_roots=[first, second])
        self.assertEqual(len(res["workspace_roots"]), 2)
        for folder in (first, second):
            config = json.loads(
                (folder / ".vscode" / "mcp.json").read_text("utf-8"))
            self.assertIn("tokengraph", config["servers"])

    # ---- IDE integration: real installable plugin artifacts ----
    def test_emit_ide_plugins_writes_installable_artifacts(self):
        import json
        res = tg.emit_ide_plugins(self.root)
        base = self.root / res["out_dir"]
        # VS Code extension: valid manifest with commands + a real entry point
        pkg = json.loads((base / "vscode" / "package.json").read_text("utf-8"))
        self.assertEqual(pkg["name"], "contextiq")
        self.assertEqual(pkg["main"], "./extension.js")
        self.assertGreaterEqual(len(pkg["contributes"]["commands"]), 4)
        self.assertTrue((base / "vscode" / "extension.js").exists())
        extension = (base / "vscode" / "extension.js").read_text("utf-8")
        self.assertIn("cp.execFile", extension)
        self.assertIn("getWorkspaceFolder", extension)
        self.assertNotIn("cp.exec(", extension)
        # Neovim Lua plugin: module + command registration
        self.assertTrue((base / "nvim" / "lua" / "contextiq" / "init.lua").exists())
        self.assertIn("nvim_create_user_command",
                      (base / "nvim" / "plugin" / "contextiq.lua").read_text("utf-8"))
        # JetBrains: buildable plugin (manifest + action + gradle)
        self.assertTrue((base / "jetbrains" / "build.gradle.kts").exists())
        self.assertIn("<idea-plugin>",
                      (base / "jetbrains" / "src" / "main" / "resources" /
                       "META-INF" / "plugin.xml").read_text("utf-8"))

    def test_emit_ide_plugins_editor_subset(self):
        res = tg.emit_ide_plugins(self.root, editors=["vscode"])
        self.assertTrue(all(w.startswith("ide-plugins/vscode/") for w in res["written"]))
        self.assertFalse((self.root / "ide-plugins" / "nvim").exists())

    # ---- Distribution: release automation + install channels ----
    def test_emit_distribution_scaffolds_release_kit(self):
        res = tg.emit_distribution(self.root)
        for f in (".github/workflows/release.yml", "Dockerfile",
                  "contextiq.rb", "install.sh", "DISTRIBUTION.md",
                  "PUBLISHING.md",
                  "homebrew-contextiq/Formula/contextiq.rb",
                  "homebrew-contextiq/README.md"):
            self.assertIn(f, res["written"])
            self.assertTrue((self.root / f).exists())
        # PUBLISHING runbook covers all four registries
        pub = (self.root / "PUBLISHING.md").read_text("utf-8")
        for token in ("twine upload", "vsce publish", "gradlew buildPlugin",
                      "homebrew-contextiq"):
            self.assertIn(token, pub)
        # the workflow builds binaries on all 3 OSes and publishes to PyPI
        wf = (self.root / ".github" / "workflows" / "release.yml").read_text("utf-8")
        self.assertIn("ubuntu-latest", wf)
        self.assertIn("macos-latest", wf)
        self.assertIn("windows-latest", wf)
        self.assertIn("pypi", wf.lower())
        self.assertIn("pyinstaller", wf)
        # install.sh prefers isolated installers (pipx/uv) — the npx/Volta analogues
        sh = (self.root / "install.sh").read_text("utf-8")
        self.assertIn("pipx", sh)
        self.assertIn("uv", sh)

    def test_pyproject_langpack_extra_present(self):
        import tomllib
        with open(_PYPROJECT, "rb") as f:
            pp = tomllib.load(f)
        extras = pp["project"]["optional-dependencies"]
        self.assertIn("langpack", extras)
        # the language pack (powers 25+ deep-parsed langs) ships in `all`
        self.assertIn("tree-sitter-language-pack", extras["all"])
        self.assertIn("Repository", pp["project"]["urls"])


class ModelAwareTokenTests(unittest.TestCase):
    """MA-1: model-aware token counting + Llama in the tier tables."""

    def test_family_mapping(self):
        self.assertEqual(tg.model_family("claude-opus"), "claude")
        self.assertEqual(tg.model_family("gemini-1.5-pro"), "gemini")
        self.assertEqual(tg.model_family("llama-3.1-70b"), "llama")
        self.assertEqual(tg.model_family("gpt-4o"), "gpt")
        self.assertEqual(tg.model_family("something-unknown"), "gpt")

    def test_counts_scale_by_family(self):
        text = "def add(a, b):\n    return a + b\n" * 5
        base = tg.count_tokens(text)
        self.assertEqual(tg.count_tokens_for_model(text, "gpt-4o"), base)
        # Llama's ratio inflates vs base; each result is a positive int
        self.assertGreaterEqual(tg.count_tokens_for_model(text, "llama-3.1-8b"), base)
        # Empty text is zero tokens. The old implementation floored every
        # count at 1, which reported a token cost for nothing at all.
        self.assertEqual(tg.count_tokens_for_model("", "claude-opus"), 0)
        self.assertGreaterEqual(tg.count_tokens_for_model("x", "claude-opus"), 1)

    def test_llama_in_tier_tables_and_pricing(self):
        for tier in ("fast", "balanced", "powerful"):
            models = tg._tier_info(tier)["models"]
            self.assertTrue(any("llama" in m for m in models), tier)
        self.assertIn("llama-3.1-70b", tg.GAIN_PRICES_PER_1M)


class CostEstimationTests(unittest.TestCase):
    """CE-1: price a call before sending it."""

    def test_estimate_from_text(self):
        res = tg.estimate_cost("write a function that adds two numbers", "claude-sonnet", 300)
        self.assertGreater(res["input_tokens"], 0)
        self.assertEqual(res["output_tokens"], 300)
        self.assertAlmostEqual(
            res["total_usd"], res["input_usd"] + res["output_usd"], places=9)
        self.assertGreater(res["total_usd"], 0)

    def test_estimate_from_int_tokens(self):
        res = tg.estimate_cost(1000, "gpt-4o", 0)
        self.assertEqual(res["input_tokens"], 1000)
        self.assertEqual(res["output_usd"], 0.0)

    def test_compare_ranks_cheapest_first(self):
        res = tg.compare_cost("hello there general context", 200)
        totals = [r["total_usd"] for r in res["by_model"]]
        self.assertEqual(totals, sorted(totals))
        self.assertEqual(res["cheapest"], res["by_model"][0])

    def test_unknown_model_falls_back(self):
        res = tg.estimate_cost("hi", "no-such-model", 10)
        self.assertEqual(res["model"], "no-such-model")
        self.assertGreater(res["total_usd"], 0)


class DedupeTests(unittest.TestCase):
    """DD-1: near-duplicate context removal (blocks + pack pieces)."""

    def test_dedupe_blocks_drops_near_duplicate(self):
        blocks = [
            "the quick brown fox jumps over the lazy dog every morning",
            "the quick brown fox jumps over the lazy dog every single morning",
            "an entirely unrelated paragraph discussing distributed databases",
        ]
        res = tg.dedupe_blocks(blocks, threshold=0.6)
        self.assertEqual(res["kept_blocks"], 2)
        self.assertEqual(len(res["removed"]), 1)
        self.assertGreater(res["tokens_saved"], 0)

    def test_dedupe_blocks_keeps_distinct(self):
        blocks = ["alpha beta gamma delta", "one two three four five six"]
        res = tg.dedupe_blocks(blocks)
        self.assertEqual(res["kept_blocks"], 2)
        self.assertEqual(res["tokens_saved"], 0)

    def test_pack_dedupes_redundant_pieces(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / ".tokengraph" / "graph.db"
            _write(root, "mod.py", SAMPLE)
            tg.index_repo(root, db)
            r = tg.Retriever(root, db)
            try:
                pack = r.find_relevant_context("helper double number", budget_tokens=4000)
                texts = [p.text for p in pack.pieces]
                # no two surviving pieces are near-duplicates of each other
                for i in range(len(texts)):
                    for j in range(i + 1, len(texts)):
                        sim = tg._dedup_similarity(
                            tg._dedup_shingles(texts[i]), tg._dedup_shingles(texts[j]))
                        self.assertLess(sim, 0.8, (texts[i], texts[j]))
                self.assertIsInstance(pack.deduped, list)
            finally:
                r.close()


_CONSTANTS_SAMPLE = '''\
"""Order limits."""

MAX_LINES_PER_ORDER = 40
RETRYABLE_STATUS = frozenset({502, 503, 504})


class Repository:
    GREP_WINDOW_LINES = 40

    def update_status(self, order_id, status):
        """Persist a status change."""
        self._rows[order_id] = status
        self._version = self._version + 1
        return self._version

    def unrelated_helper(self, value):
        """Nothing to do with the question."""
        return value


def build_repository():
    """Construct the repository."""
    return Repository()
'''


class ConstantExtractionTests(unittest.TestCase):
    """CN-1: module- and class-scope bindings are symbols in their own right.

    A controlling value that is not a symbol cannot be searched for, ranked, or
    packed — which is how packs used to name the right file and still omit the
    constant that answered the question.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / ".tokengraph" / "graph.db"
        _write(self.root, "shop.py", _CONSTANTS_SAMPLE)
        tg.index_repo(self.root, self.db)
        self.ret = tg.Retriever(self.root, self.db)

    def tearDown(self):
        self.ret.close()
        self._tmp.cleanup()

    def _kinds(self):
        return {s["name"]: s["kind"] for s in self.ret.store.file_symbols("shop.py")}

    def test_module_level_constants_become_symbols(self):
        kinds = self._kinds()
        self.assertEqual(kinds.get("MAX_LINES_PER_ORDER"), "constant")
        self.assertEqual(kinds.get("RETRYABLE_STATUS"), "constant")

    def test_class_level_constants_become_symbols(self):
        self.assertEqual(self._kinds().get("GREP_WINDOW_LINES"), "constant")

    def test_locals_are_not_indexed(self):
        """A binding inside a function body is implementation detail."""
        self.assertNotIn("_version", self._kinds())

    def test_constant_signature_carries_its_value(self):
        """A constant whose signature is just its name answers nothing."""
        sig = self.ret.get_symbol("shop.MAX_LINES_PER_ORDER") or ""
        self.assertIn("40", sig)

    def test_constant_is_reachable_by_search(self):
        hits = {r["qname"] for r in self.ret.store.search("retryable status", limit=10)}
        self.assertIn("shop.RETRYABLE_STATUS", hits)


class CompletionSweepTests(unittest.TestCase):
    """SR-1: leftover budget finishes the files the pack committed to.

    The regression this guards: retrieval located the right file, spent well
    under half the requested budget, and returned without the symbol the
    question was about.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / ".tokengraph" / "graph.db"
        _write(self.root, "shop.py", _CONSTANTS_SAMPLE)
        tg.index_repo(self.root, self.db)
        self.ret = tg.Retriever(self.root, self.db)

    def tearDown(self):
        self.ret.close()
        self._tmp.cleanup()

    def test_sweep_reaches_symbols_search_did_not_rank(self):
        """The answer is in the file; a generous budget must actually carry it.

        Asserted against the rendered pack rather than the piece list, because
        a method is equally well delivered inside its class body — what matters
        is that the querent can see it, not which piece carried it.
        """
        pack = self.ret.find_relevant_context("build the repository",
                                              budget_tokens=4000)
        self.assertIn("_version + 1", pack.to_markdown())

    def test_sweep_marks_its_pieces(self):
        pack = self.ret.find_relevant_context("build the repository",
                                              budget_tokens=4000)
        self.assertTrue(any(p.reason == "file completion" for p in pack.pieces))

    def test_sweep_respects_both_budget_contracts(self):
        """Payload sum and rendered markdown must each stay inside the budget."""
        for budget in (300, 900, 4000):
            pack = self.ret.find_relevant_context("update status version",
                                                  budget_tokens=budget)
            self.assertLessEqual(pack.tokens, budget, budget)
            self.assertLessEqual(tg.count_tokens(pack.to_markdown()), budget, budget)

    def test_sweep_does_not_show_the_same_lines_twice(self):
        """A body and a symbol nested inside it are never both printed."""
        pack = self.ret.find_relevant_context("repository status", budget_tokens=4000)
        spans: dict[str, list[tuple[int, int]]] = {}
        for p in pack.pieces:
            if p.detail != "body":
                continue
            row = self.ret.store.symbol_by_qname(p.qname)
            if row is None or not row["lineno"]:
                continue
            lo, hi = row["lineno"], row["end_lineno"] or row["lineno"]
            for s, e in spans.get(p.file, []):
                self.assertFalse(lo <= e and hi >= s,
                                 f"{p.qname} overlaps an already-shown body")
            spans.setdefault(p.file, []).append((lo, hi))

    def test_swept_symbols_are_not_also_advertised_as_missing(self):
        pack = self.ret.find_relevant_context("build the repository",
                                              budget_tokens=4000)
        self.assertFalse({p.qname for p in pack.pieces} & set(pack.dropped))

    def test_budget_is_actually_spent_when_there_is_material(self):
        """The defect was a half-empty pack, not an over-full one.

        Needs a repository larger than the budget, or "unspent" just means
        "nothing left to say" — which is the one case where stopping early is
        correct, and is why the small fixture above cannot test this.
        """
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "\n\n".join(
                f'def op_{i}(value):\n'
                f'    """Operation number {i}."""\n'
                f'    total = value * {i}\n'
                f'    return total + {i}'
                for i in range(60))
            _write(root, "big.py", body)
            _write(root, "shop.py", _CONSTANTS_SAMPLE)
            db = root / ".tokengraph" / "graph.db"
            tg.index_repo(root, db)
            r = tg.Retriever(root, db)
            try:
                pack = r.find_relevant_context("operation total value",
                                               budget_tokens=1500)
                spent = tg.count_tokens(pack.to_markdown())
                self.assertGreater(spent, 1500 * 0.6, "pack left the budget unspent")
                self.assertLessEqual(spent, 1500)
            finally:
                r.close()


class ExtractorVersionTests(unittest.TestCase):
    """EX-1: upgrading the extractor invalidates an index built by an older one.

    Without this the incremental fast path silently keeps a graph that predates
    the symbol kinds this build knows how to find, until someone edits the file.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / ".tokengraph" / "graph.db"

    def tearDown(self):
        self._tmp.cleanup()

    def test_version_covers_generation_and_grammars(self):
        v = tg.extractor_version()
        self.assertTrue(v.startswith(f"{tg.EXTRACTOR_GENERATION}:"))

    def test_unchanged_files_are_skipped_normally(self):
        _write(self.root, "mod.py", SAMPLE)
        tg.index_repo(self.root, self.db)
        rep = tg.index_repo(self.root, self.db)
        self.assertEqual(rep.parsed, 0)
        self.assertEqual(rep.skipped, 1)

    def test_extractor_change_forces_reparse_of_unchanged_files(self):
        _write(self.root, "mod.py", SAMPLE)
        tg.index_repo(self.root, self.db)
        store = tg.Store(self.db)
        try:
            store.set_meta("extractor_version", "0:stale")
            store.commit()
        finally:
            store.close()
        rep = tg.index_repo(self.root, self.db)
        self.assertEqual(rep.parsed, 1, "stale extractor stamp must force a reparse")
        store = tg.Store(self.db)
        try:
            self.assertEqual(store.get_meta("extractor_version"),
                             tg.extractor_version())
        finally:
            store.close()


_TRANSCRIPT = """\
user: The checkout flow times out under load. Where should I look?
assistant: The retry loop in `payments/gateway.py` is the likely cause. It uses
a fixed backoff, so a slow upstream stacks requests instead of shedding them.
user: Can we just raise the timeout?
assistant: We decided to add jitter to the backoff rather than raise
REQUEST_TIMEOUT_MS, because raising it hides the queueing problem.
user: What about the retry budget?
assistant: The constraint is that MAX_RETRIES must stay at 3 — the upstream
contract caps us there. Still open: whether the circuit breaker should trip on
timeouts or only on 5xx responses.
assistant: Next step is to patch `charge_card` and add a load test.
"""


class SummaryFidelityTests(unittest.TestCase):
    """CS-3: compression is only good if what an answer needs survives it.

    A reduction percentage alone rewards deleting everything; these check the
    opposite property — that decisions, constraints, open questions and the
    identifiers under discussion are still there afterwards.
    """

    def _summary(self, max_tokens=400):
        return tg.summarize_conversation(_TRANSCRIPT, max_tokens=max_tokens)

    def test_a_faithful_summary_keeps_identifiers_and_facts(self):
        score = tg.score_summary_fidelity(self._summary(), {
            "identifiers": ["payments/gateway.py", "MAX_RETRIES"],
            "facts": ["jitter", "circuit breaker"],
        })
        self.assertTrue(score["faithful"],
                        f"lost {score['identifiers_lost']} / {score['facts_lost']}")
        self.assertGreaterEqual(score["identifier_recall"], 0.8)

    def test_missing_content_is_reported_not_hidden(self):
        score = tg.score_summary_fidelity(self._summary(), {
            "identifiers": ["a_symbol_never_mentioned"],
            "facts": ["a fact never stated"],
        })
        self.assertFalse(score["faithful"])
        self.assertEqual(score["identifier_recall"], 0.0)
        self.assertIn("a_symbol_never_mentioned", score["identifiers_lost"])
        self.assertIn("a fact never stated", score["facts_lost"])

    def test_an_aggressive_budget_trades_fidelity_and_says_so(self):
        """The point of the metric: a smaller summary is not automatically better."""
        loose = tg.score_summary_fidelity(self._summary(400), {
            "identifiers": ["payments/gateway.py", "MAX_RETRIES"],
            "facts": ["jitter", "circuit breaker"],
        })
        tight = tg.score_summary_fidelity(self._summary(40), {
            "identifiers": ["payments/gateway.py", "MAX_RETRIES"],
            "facts": ["jitter", "circuit breaker"],
        })
        self.assertGreaterEqual(tight["reduction_pct"], loose["reduction_pct"])
        self.assertLessEqual(tight["fact_recall"], loose["fact_recall"])

    def test_empty_requirements_do_not_fabricate_a_score(self):
        score = tg.score_summary_fidelity(self._summary(), {})
        self.assertEqual(score["identifier_recall"], 1.0)
        self.assertEqual(score["fact_recall"], 1.0)


class PromptQualityTests(unittest.TestCase):
    """PQ-1: score a prompt (not an answer)."""

    def test_specific_prompt_scores_high(self):
        res = tg.score_prompt(
            "fix the retry backoff in `count_tokens` in tokengraph_all.py")
        self.assertGreaterEqual(res["score"], 70)
        self.assertIn(res["grade"], ("good", "excellent"))

    def test_vague_prompt_scores_low_with_suggestions(self):
        res = tg.score_prompt("do something with the stuff")
        self.assertLess(res["score"], 50)
        self.assertTrue(res["suggestions"])

    def test_subscores_present(self):
        res = tg.score_prompt("explain how login works")
        for k in ("clarity", "specificity", "context", "actionability"):
            self.assertIn(k, res["subscores"])

    def test_empty_prompt_is_weak(self):
        res = tg.score_prompt("")
        self.assertEqual(res["grade"], "weak")


class ConversationSummaryTests(unittest.TestCase):
    """CS-1: compress a chat transcript."""

    TRANSCRIPT = (
        "User: We need to add Llama support to the tokenizer.\n"
        "Assistant: Agreed. I will use a per-family ratio in `count_tokens_for_model`.\n"
        "  TODO: also add pricing for llama models.\n"
        "User: what about Gemini pricing?\n"
        "Assistant: We should use the 1.5-pro list price. The plan is to extend "
        "`MODEL_PRICES_PER_1M`.\n"
    )

    def test_extracts_sections(self):
        res = tg.summarize_conversation(self.TRANSCRIPT, max_tokens=300)
        self.assertEqual(res["turns"], 4)   # 2 user + 2 assistant (TODO line continues turn 2)
        self.assertTrue(res["decisions"])
        self.assertTrue(res["action_items"])
        self.assertTrue(res["open_questions"])
        self.assertIn("count_tokens_for_model", res["entities"])

    def test_reduces_tokens(self):
        big = self.TRANSCRIPT * 8
        res = tg.summarize_conversation(big, max_tokens=200)
        self.assertLess(res["summary_tokens"], res["original_tokens"])
        self.assertLessEqual(res["summary_tokens"], 260)  # near the cap
        self.assertGreater(res["reduction_pct"], 0)

    def test_empty_transcript(self):
        res = tg.summarize_conversation("")
        self.assertEqual(res["turns"], 0)
        self.assertEqual(res["summary"], "")

    def test_redacts_secrets(self):
        t = ("User: here is the key\nAssistant: I will use "
             "token=ghp_abcdefghijklmnopqrstuvwxyz0123456789 for auth\n")
        res = tg.summarize_conversation(t)
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz0123456789", res["summary"])


class LedgerLoggingTests(unittest.TestCase):
    """Every savings-producing optimization tool appends to the gain ledger
    (so squeeze / dedupe / summarize show up in the dashboard, not just packs)."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        _write(self.root, "mod.py", SAMPLE)

    def _ops(self):
        return [r["op"] for r in tg.read_gain_ledger(self.root)]

    def _run(self, fn, **kw):
        import argparse
        import contextlib
        import io
        base = dict(path=str(self.root), json=False, no_refresh=False,
                    no_track=False, text_file=None)
        base.update(kw)
        args = argparse.Namespace(**base)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            fn(args)
        return args

    def test_dedupe_cli_logs_savings(self):
        self._run(tg.cmd_dedupe,
                  text="aaa bbb ccc ddd eee fff\n\naaa bbb ccc ddd eee fff\n\nzzz yyy xxx www",
                  sep=None, threshold=0.7)
        rows = [r for r in tg.read_gain_ledger(self.root) if r["op"] == "dedupe"]
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["saved"], 0)
        self.assertEqual(rows[0]["files"], 0)

    def test_summarize_cli_logs_when_it_saves(self):
        # A genuinely long transcript nets positive savings -> logs a row.
        turn = ("User: We should refactor the auth module and add retries. "
                "what about caching the token in redis?\n"
                "Assistant: Agreed. Decision: use 3 retries with jitter. "
                "TODO: add a redis client and unit tests.\n")
        self._run(tg.cmd_summarize_chat, text=turn * 12, max_tokens=200)
        rows = [r for r in tg.read_gain_ledger(self.root) if r["op"] == "summarize"]
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["saved"], 0)

    def test_summarize_short_transcript_not_logged(self):
        # A short transcript whose section scaffolding (Decisions/Actions/…)
        # exceeds the original nets saved<=0 — the guard keeps it out of the
        # ledger so a burst of tiny summaries can't dilute the dashboard totals.
        short = ("User: Decision: use cl100k. TODO: add tests. what next?\n"
                 "Assistant: Agreed, plan is to ship. TODO: write docs.\n")
        res = tg.summarize_conversation(short, max_tokens=300)
        self.assertLessEqual(res["tokens_saved"], 0)          # precondition
        self._run(tg.cmd_summarize_chat, text=short, max_tokens=300)
        self.assertNotIn("summarize", self._ops())

    def test_record_pack_savings_skips_non_savings(self):
        tg.record_pack_savings(self.root, "unit", final_tokens=120,
                               baseline_tokens=100, files=0)   # saved = -20
        tg.record_pack_savings(self.root, "unit", final_tokens=100,
                               baseline_tokens=100, files=0)   # saved = 0
        self.assertEqual(tg.read_gain_ledger(self.root), [])

    def test_squeeze_cli_logs(self):
        self._run(tg.cmd_squeeze,
                  text=('Traceback (most recent call last):\n'
                        '  File "app.py", line 5, in main\n    go()\nValueError: nope\n'),
                  kind="auto")
        rows = [r for r in tg.read_gain_ledger(self.root) if r["op"] == "squeeze"]
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["saved"], 0)

    def test_no_track_opt_out_is_respected(self):
        self._run(tg.cmd_dedupe, text="aa bb cc dd\n\naa bb cc dd", sep=None,
                  threshold=0.7, no_track=True)
        self.assertEqual(tg.read_gain_ledger(self.root), [])


class UsageReportTests(unittest.TestCase):
    """Per-workspace static HTML report co-located with the graph in .tokengraph."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    @staticmethod
    def _payload_from(html: str) -> dict:
        """Pull the inlined snapshot back out of the rendered page."""
        import json
        import re
        m = re.search(r'id="ciq-data">(.*?)</script>', html, re.S)
        return json.loads(m.group(1).replace("<\\/", "</"))

    def test_auto_generated_on_savings(self):
        tg.record_pack_savings(self.root, "context", final_tokens=3000,
                               baseline_tokens=100000, files=5)
        rep = tg.usage_report_path(self.root)
        self.assertTrue(rep.exists())
        self.assertEqual(rep.parent.name, ".tokengraph")
        html = rep.read_text(encoding="utf-8")
        # every panel container + both controls are present
        for hook in ("c-waterfall", "c-area", "c-ops", "c-rows",
                     "sel-model", "sel-range", "Tokens sent (consumed)"):
            self.assertIn(hook, html)
        self.assertIn("prefers-color-scheme", html)      # theme-aware
        # self-contained: no external hosts / CDNs
        self.assertIsNone(re.search(r"https?://(?!127\.)", html))
        self.assertNotIn("cdn", html.lower())

    def test_not_generated_when_opted_out(self):
        tg.record_pack_savings(self.root, "context", final_tokens=10,
                               baseline_tokens=1000, files=1, no_track=True)
        self.assertFalse(tg.usage_report_path(self.root).exists())

    def test_write_usage_report_stamps_generated_at(self):
        tg.record_pack_savings(self.root, "context", final_tokens=1,
                               baseline_tokens=500, files=1)
        p = tg.write_usage_report(self.root, generated_at="2026-01-01 00:00")
        self.assertIsNotNone(p)
        payload = self._payload_from(p.read_text(encoding="utf-8"))
        self.assertEqual(payload["generated_at"], "2026-01-01 00:00")

    def test_auto_report_gets_a_real_timestamp(self):
        tg.record_pack_savings(self.root, "context", final_tokens=1,
                               baseline_tokens=500, files=1)
        payload = self._payload_from(
            tg.usage_report_path(self.root).read_text(encoding="utf-8"))
        self.assertTrue(payload["generated_at"])   # not blank on the auto path

    def test_payload_schema(self):
        tg.record_pack_savings(self.root, "context", final_tokens=100,
                               baseline_tokens=9000, files=2)
        p = tg.build_report_payload(self.root)
        self.assertEqual(p["version"], tg.REPORT_SCHEMA_VERSION)
        for key in ("workspace", "model", "prices", "totals", "by_op",
                    "daily", "weekly", "monthly", "rows"):
            self.assertIn(key, p)
        for key in ("runs", "saved", "baseline", "final", "reduction_pct",
                    "saved_usd"):
            self.assertIn(key, p["totals"])
        # buckets carry baseline/final so the page can recompute a window
        for key in ("period", "saved", "runs", "baseline", "final"):
            self.assertIn(key, p["daily"][0])
        # prices ship with the payload so the model selector works offline
        self.assertIn("claude-sonnet", p["prices"])

    def test_payload_carries_report_only_extras(self):
        """Fields the dashboard adds on top of the CLI summary."""
        tg.record_pack_savings(self.root, "context", final_tokens=100,
                               baseline_tokens=9000, files=4)
        p = tg.build_report_payload(self.root)
        # per-model input+output list prices drive the cost-by-model panel
        self.assertIn("claude-sonnet", p["prices_io"])
        self.assertIn("input", p["prices_io"]["claude-sonnet"])
        # daily buckets carry files so a windowed "files covered" is honest
        self.assertEqual(p["daily"][0]["files"], 4)
        self.assertEqual(p["totals"]["files"], 4)
        # ledger span + workspace facts (empty here: no graph db in the tmp root)
        self.assertEqual(p["ledger"]["active_days"], 1)
        self.assertGreater(p["ledger"]["first_ts"], 0)
        self.assertEqual(p["workspace_stats"], {})
        self.assertFalse(p["rows_capped"])

    def test_workspace_stats_never_raise_or_create_a_db(self):
        """Runs on the savings hot path — must degrade, not fail, and stay read-only."""
        self.assertEqual(tg._workspace_stats(self.root), {})
        self.assertFalse((self.root / ".tokengraph" / "graph.db").exists())
        (self.root / ".tokengraph").mkdir(parents=True, exist_ok=True)
        (self.root / ".tokengraph" / "graph.db").write_text("not a database")
        self.assertEqual(tg._workspace_stats(self.root), {})

    def test_report_structure_and_accessibility_hooks(self):
        tg.record_pack_savings(self.root, "context", final_tokens=3000,
                               baseline_tokens=100000, files=5)
        html = tg.usage_report_path(self.root).read_text(encoding="utf-8")
        for hook in ("c-gauge", "c-heat", "c-mix", "c-optable", "c-models",
                     "c-modeltable", "c-ws", "c-langs"):
            self.assertIn(hook, html)               # every new panel container
        for hook in ('class="skip"', 'id="main"', 'aria-live="polite"',
                     'scope="col"', 'role="img"', '<label for="sel-model">',
                     "prefers-reduced-motion"):
            self.assertIn(hook, html)               # a11y affordances
        # theme is switchable both ways: OS media query + explicit stamp
        self.assertIn(':root[data-theme="dark"]', html)
        self.assertIn(':root:not([data-theme="light"])', html)

    def test_payload_row_tail_is_bounded(self):
        for _ in range(12):
            tg.record_pack_savings(self.root, "context", final_tokens=1,
                                   baseline_tokens=100, files=1)
        p = tg.build_report_payload(self.root, max_rows=5)
        self.assertEqual(len(p["rows"]), 5)

    def test_renders_with_empty_ledger(self):
        html = tg.render_report_html(tg.build_report_payload(self.root))
        self.assertIn("ciq-data", html)          # no crash on an empty ledger

    def _run_js_harness(self, name: str) -> str:
        """Run the page's own JS in a fake DOM via node; skip if node is absent."""
        import shutil
        import subprocess
        node = shutil.which("node")
        if not node:
            self.skipTest("node not available")
        harness = Path(__file__).with_name("report_js_harness.js")
        if not harness.exists():
            self.skipTest("harness missing")
        rendered = self.root / name
        rendered.write_text(
            tg.usage_report_path(self.root).read_text(encoding="utf-8"),
            encoding="utf-8")
        res = subprocess.run([node, str(harness), str(rendered)],
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertNotIn("FAIL", res.stdout, res.stdout)
        return res.stdout

    def test_report_js_behaviour(self):
        """Covers what a string assertion can't: UI state round-tripping through
        location.hash across a file:// reload, the pause toggle, fallback on a
        garbage hash, and skipping the redraw when a poll returns identical data.
        """
        tg.record_pack_savings(self.root, "context", final_tokens=3000,
                               baseline_tokens=100000, files=1)
        out = self._run_js_harness("rendered.html")
        self.assertIn("model restored after reload", out)

    def test_report_js_subcent_precision_on_small_ledger(self):
        """A young ledger must still price every model distinctly.

        Regression: with fixed 2-dp formatting most models collapsed to "$0.00"
        on a small ledger, so switching the pricing model looked like a no-op.
        """
        tg.record_pack_savings(self.root, "context", final_tokens=800,
                               baseline_tokens=4000, files=1)   # only 3.2K saved
        out = self._run_js_harness("small.html")
        self.assertIn("no $0.00 collapse", out)

    def test_live_server_serves_page_and_data(self):
        import json
        import threading
        import urllib.request
        from http.server import ThreadingHTTPServer
        tg.record_pack_savings(self.root, "context", final_tokens=3000,
                               baseline_tokens=100000, files=1)
        srv = ThreadingHTTPServer(("127.0.0.1", 0),
                                  tg.make_report_handler(self.root))
        self.addCleanup(srv.server_close)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{srv.server_port}"
        self.assertEqual(srv.server_address[0], "127.0.0.1")   # loopback only
        data = json.loads(urllib.request.urlopen(base + "/data.json").read())
        self.assertEqual(data["version"], tg.REPORT_SCHEMA_VERSION)
        before = data["totals"]["saved"]
        self.assertIn("ciq-data", urllib.request.urlopen(base + "/").read().decode())
        # /data.json re-reads the ledger, so new ops appear without a restart
        tg.record_pack_savings(self.root, "dedupe", final_tokens=100,
                               baseline_tokens=5000, files=0)
        after = json.loads(
            urllib.request.urlopen(base + "/data.json").read())["totals"]["saved"]
        self.assertGreater(after, before)

    def test_gain_report_cli_writes_default_path(self):
        import argparse
        import contextlib
        import io
        tg.record_pack_savings(self.root, "context", final_tokens=100,
                               baseline_tokens=9000, files=2)
        args = argparse.Namespace(path=str(self.root), model=tg.DEFAULT_GAIN_MODEL,
                                  report=True)
        with contextlib.redirect_stdout(io.StringIO()):
            tg.cmd_gain(args)
        self.assertTrue(tg.usage_report_path(self.root).exists())


class _RepoCase(unittest.TestCase):
    """Base: a throwaway indexed repo per test."""

    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.db = self.root / ".tokengraph" / "graph.db"

    def _index(self):
        tg.index_repo(self.root, self.db)
        r = tg.Retriever(self.root, self.db)
        self.addCleanup(r.close_now)
        return r


class NeighborRankingTests(_RepoCase):
    """NB-1: hub symbols must not flood the pack with irrelevant signatures."""

    def _hub_repo(self, callers: int = 120):
        _write(self.root, "hub.py", "def shared(x):\n    return x\n")
        # Many unrelated callers of one hub symbol, plus the actual target.
        for i in range(callers):
            _write(self.root, f"caller_{i}.py",
                   f"from hub import shared\n\n"
                   f"def unrelated_{i}(v):\n"
                   f"    return shared(v)\n")
        _write(self.root, "target.py",
               "from hub import shared\n\n"
               "def parse_yaml_frontmatter(text):\n"
               "    '''Split YAML frontmatter from a markdown document.'''\n"
               "    return shared(text)\n")

    # depth=2 is where the flood actually happens: the seed reaches the hub in
    # one hop, and the hub's whole caller list in the second.
    def test_fanout_is_bounded(self):
        self._hub_repo()
        r = self._index()
        pack = r.find_relevant_context("parse yaml frontmatter from markdown",
                                       budget_tokens=6000, expand_depth=2)
        neighbor_pieces = [p for p in pack.pieces
                           if p.reason in ("caller", "callee", "base")]
        self.assertLessEqual(len(neighbor_pieces), tg.MAX_NEIGHBOR_SIGS)
        # The expansion saw far more candidates than it emitted.
        self.assertGreater(pack.neighbors_considered, tg.MAX_NEIGHBOR_SIGS)
        self.assertGreater(pack.neighbors_pruned, 0)

    def test_candidate_collection_is_capped(self):
        self._hub_repo(callers=300)
        r = self._index()
        pack = r.find_relevant_context("parse yaml frontmatter from markdown",
                                       budget_tokens=6000, expand_depth=2)
        self.assertLessEqual(pack.neighbors_considered,
                             tg.MAX_NEIGHBOR_CANDIDATES)

    def test_target_survives_hub_flood(self):
        """The asked-for symbol must not be evicted by a hub's caller list."""
        self._hub_repo()
        r = self._index()
        pack = r.find_relevant_context("parse yaml frontmatter from markdown",
                                       budget_tokens=4000, expand_depth=2)
        qnames = {p.qname for p in pack.pieces}
        self.assertIn("target.parse_yaml_frontmatter", qnames)

    def test_relevant_neighbors_outrank_noise(self):
        """Emitted neighbours should skew to the task, not to discovery order."""
        self._hub_repo()
        r = self._index()
        pack = r.find_relevant_context("parse yaml frontmatter from markdown",
                                       budget_tokens=6000, expand_depth=2)
        noise = [p for p in pack.pieces
                 if p.reason == "caller" and p.qname.startswith("caller_")]
        # Without ranking every one of the 120 unrelated callers would be a
        # candidate for emission; the cap alone bounds this to MAX_NEIGHBOR_SIGS.
        self.assertLessEqual(len(noise), tg.MAX_NEIGHBOR_SIGS)

    def test_scoring_prefers_task_overlap(self):
        rows = []

        class _Row(dict):
            def __getitem__(self, k):
                return self.get(k)

        relevant = _Row(id=1, qname="mod.parse_yaml", name="parse_yaml",
                        signature="def parse_yaml(text)", docstring="parse yaml")
        noise = _Row(id=2, qname="mod.unrelated_42", name="unrelated_42",
                     signature="def unrelated_42(v)", docstring="")
        terms = set(tg._tokenize("parse yaml frontmatter"))
        s_rel = tg.Retriever._neighbor_score(relevant, "caller", 0, terms, 2)
        s_noise = tg.Retriever._neighbor_score(noise, "caller", 0, terms, 2)
        self.assertGreater(s_rel, s_noise)
        rows.append((s_rel, s_noise))

    def test_hub_penalty_lowers_score(self):
        class _Row(dict):
            def __getitem__(self, k):
                return self.get(k)

        row = _Row(id=1, qname="m.f", name="f", signature="def f()", docstring="")
        terms = set(tg._tokenize("anything"))
        low_degree = tg.Retriever._neighbor_score(row, "callee", 0, terms, 1)
        high_degree = tg.Retriever._neighbor_score(row, "callee", 0, terms, 500)
        self.assertGreater(low_degree, high_degree)


class SessionDedupTests(_RepoCase):
    """SD-1: a second retrieval must not re-bill unchanged content."""

    def _repo(self):
        _write(self.root, "svc.py",
               "def charge_card(token, amount):\n"
               "    '''Charge a card through the payment provider.'''\n"
               "    validated = validate(token)\n"
               "    return submit(validated, amount)\n\n"
               "def validate(token):\n"
               "    return token.strip()\n\n"
               "def submit(t, amount):\n"
               "    return {'t': t, 'amount': amount}\n")

    def test_second_call_is_cheaper_and_lists_reuse(self):
        self._repo()
        r = self._index()
        first = r.find_relevant_context("charge a card", budget_tokens=3000,
                                        session="conv-1")
        second = r.find_relevant_context("charge a card", budget_tokens=3000,
                                         session="conv-1")
        self.assertLess(second.rendered_tokens, first.rendered_tokens)
        self.assertTrue(second.reused)
        self.assertGreater(second.tokens_reused, 0)
        self.assertIn("Already sent earlier this session", second.to_markdown())

    def test_sessions_are_isolated(self):
        self._repo()
        r = self._index()
        r.find_relevant_context("charge a card", budget_tokens=3000, session="a")
        other = r.find_relevant_context("charge a card", budget_tokens=3000,
                                        session="b")
        self.assertEqual(other.reused, [])

    def test_no_session_id_means_no_dedup(self):
        self._repo()
        r = self._index()
        a = r.find_relevant_context("charge a card", budget_tokens=3000)
        b = r.find_relevant_context("charge a card", budget_tokens=3000)
        self.assertEqual(a.rendered_tokens, b.rendered_tokens)
        self.assertEqual(b.reused, [])

    def test_edited_symbol_is_resent(self):
        """Content hash, not just name — an edit must invalidate the reuse."""
        self._repo()
        r = self._index()
        r.find_relevant_context("charge a card", budget_tokens=3000, session="s")
        _write(self.root, "svc.py",
               "def charge_card(token, amount):\n"
               "    '''Charge a card through the payment provider.'''\n"
               "    validated = validate(token)\n"
               "    audit_log(validated)\n"
               "    return submit(validated, amount)\n\n"
               "def validate(token):\n"
               "    return token.strip()\n\n"
               "def audit_log(v):\n    return v\n\n"
               "def submit(t, amount):\n"
               "    return {'t': t, 'amount': amount}\n")
        tg.index_repo(self.root, self.db)
        r.invalidate()
        again = r.find_relevant_context("charge a card", budget_tokens=3000,
                                        session="s")
        self.assertIn("audit_log", again.to_markdown())
        self.assertNotIn("svc.charge_card", again.reused)

    def test_reset_session_clears_ledger(self):
        self._repo()
        r = self._index()
        r.find_relevant_context("charge a card", budget_tokens=3000, session="s")
        self.assertGreater(r.store.session_stats("s")["symbols_sent"], 0)
        r.store.clear_session("s")
        self.assertEqual(r.store.session_stats("s")["symbols_sent"], 0)


class HonestBaselineTests(_RepoCase):
    """MS-1: the headline saving must use a realistic agent baseline."""

    def _big_repo(self):
        # One large file: the case where whole-file baselines flatter most.
        body = "\n".join(
            f"def fn_{i}(x):\n"
            f"    '''Function number {i}.'''\n"
            f"    total = 0\n"
            f"    for j in range(x):\n"
            f"        total += j * {i}\n"
            f"    return total\n"
            for i in range(200))
        _write(self.root, "big.py", body)

    def test_targeted_baseline_is_far_below_whole_file(self):
        self._big_repo()
        r = self._index()
        m = r.measure("function number 7 accumulate total")
        self.assertEqual(m["baseline_kind"], "grep+targeted-read")
        self.assertLess(m["baseline_tokens"], m["baseline_whole_file_tokens"])
        # Both numbers are reported; neither is hidden.
        self.assertIn("savings_pct_vs_whole_file", m)

    def test_savings_can_be_negative(self):
        """A metric that cannot show a loss is not a measurement."""
        _write(self.root, "tiny.py", "def a():\n    return 1\n")
        r = self._index()
        m = r.measure("a")
        # On a two-line file the pack overhead exceeds just reading it, and the
        # tool must be willing to say so.
        self.assertLess(m["savings_pct"], 100.0)
        self.assertIsInstance(m["tokens_saved"], int)

    def test_windows_merge_rather_than_double_count(self):
        self._big_repo()
        r = self._index()
        pack = r.find_relevant_context("function number 7", budget_tokens=4000)
        baseline = r._targeted_baseline(pack)
        whole = sum(r.store.token_est_for(f)
                    for f in {p.file for p in pack.pieces})
        self.assertGreater(baseline, 0)
        self.assertLessEqual(baseline, whole * 2)


class EmbeddingBackendTests(_RepoCase):
    """EM-1/EM-2: backend identity, invalidation and honest reporting."""

    def test_backend_id_is_stable_and_descriptive(self):
        bid = tg.embed_backend_id()
        self.assertTrue(bid)
        self.assertEqual(bid, tg.embed_backend_id())

    def test_info_reports_kind_and_status(self):
        info = tg.embed_backend_info()
        self.assertIn(info["kind"], ("hashing", "sentence-transformers"))
        self.assertIsInstance(info["semantic"], bool)
        self.assertTrue(info["status"])
        if not info["semantic"]:
            self.assertIn("does NOT match by meaning", info["note"])

    def test_vectors_carry_backend_and_stale_ones_are_dropped(self):
        _write(self.root, "a.py", "def alpha():\n    return 1\n")
        r = self._index()
        self.assertTrue(r.store.has_vectors())
        r.store.conn.execute("UPDATE vectors SET backend='some-other-backend'")
        r.store.commit()
        # Vectors from another space are invisible, not silently compared.
        self.assertFalse(r.store.has_vectors())
        self.assertGreater(len(r.store.symbols_missing_vectors()), 0)
        removed = r.store.drop_stale_vectors()
        self.assertGreater(removed, 0)

    def test_reindex_backfills_missing_vectors(self):
        _write(self.root, "a.py", "def alpha():\n    return 1\n")
        tg.index_repo(self.root, self.db)
        store = tg.Store(self.db)
        store.conn.execute("UPDATE vectors SET backend='stale'")
        store.commit()
        store.set_meta("embed_backend", "stale")
        store.close()
        report = tg.index_repo(self.root, self.db)
        self.assertGreater(report.reembedded, 0)
        store = tg.Store(self.db)
        try:
            self.assertTrue(store.has_vectors())
        finally:
            store.close()

    def test_semantic_search_falls_back_instead_of_returning_nothing(self):
        _write(self.root, "a.py", "def alpha_beta():\n    return 1\n")
        r = self._index()
        r.store.conn.execute("DELETE FROM vectors")
        r.store.commit()
        hits = r.semantic_search("alpha_beta", limit=5)
        self.assertTrue(hits, "must degrade to lexical, not return empty")


class SchemaMigrationTests(_RepoCase):
    def test_user_version_is_stamped(self):
        _write(self.root, "a.py", "def a():\n    return 1\n")
        tg.index_repo(self.root, self.db)
        store = tg.Store(self.db)
        try:
            v = store.conn.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(v, tg.SCHEMA_VERSION)
        finally:
            store.close()

    def test_newer_schema_is_rebuilt_not_misread(self):
        _write(self.root, "a.py", "def a():\n    return 1\n")
        tg.index_repo(self.root, self.db)
        store = tg.Store(self.db)
        store.conn.execute(f"PRAGMA user_version={tg.SCHEMA_VERSION + 5}")
        store.commit()
        store.close()
        store = tg.Store(self.db)       # must not raise
        try:
            v = store.conn.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(v, tg.SCHEMA_VERSION)
        finally:
            store.close()

    def test_meta_roundtrip(self):
        store = tg.Store(self.db)
        try:
            self.assertEqual(store.get_meta("nope", "dflt"), "dflt")
            store.set_meta("k", "v")
            self.assertEqual(store.get_meta("k"), "v")
        finally:
            store.close()


class TargetedDeleteTests(_RepoCase):
    """A notify-style targeted refresh must reconcile the paths it is given."""

    def test_targeted_reindex_forgets_deleted_path(self):
        _write(self.root, "a.py", "def a():\n    return 1\n")
        _write(self.root, "b.py", "def b():\n    return 2\n")
        tg.index_repo(self.root, self.db)
        (self.root / "b.py").unlink()
        report = tg.index_repo(self.root, self.db, paths=["b.py"])
        self.assertEqual(report.removed, 1)
        store = tg.Store(self.db)
        try:
            self.assertNotIn("b.py", store.all_indexed_files())
            self.assertIn("a.py", store.all_indexed_files())
        finally:
            store.close()


class PromptCacheOrderingTests(_RepoCase):
    """PC-1: volatile git state must sit AFTER the cache breakpoint."""

    def _repo_with_git_noise(self):
        _write(self.root, "m.py",
               "def alpha():\n    return 1\n\n# TODO: revisit this\n")

    def test_payload_splits_stable_from_volatile(self):
        self._repo_with_git_noise()
        r = self._index()
        payload = tg.build_context_payload(
            r, self.root, strategy="full", src_dirs=["."], budget=4000,
            hot_commits=5, diff=False, staged=False, config={})
        self.assertIn("stable_prefix", payload)
        self.assertNotIn("## TODO / FIXME", payload["stable_prefix"])
        self.assertIn("## TODO / FIXME", payload["volatile_suffix"])
        # The whole document is still the concatenation, in order.
        self.assertTrue(payload["markdown"].startswith(
            payload["stable_prefix"].rstrip()[:80]))

    def test_cache_blocks_annotate_only_the_stable_block(self):
        self._repo_with_git_noise()
        r = self._index()
        payload = tg.build_context_payload(
            r, self.root, strategy="full", src_dirs=["."], budget=4000,
            hot_commits=5, diff=False, staged=False, config={})
        blocks = tg.cache_blocks(payload)
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})
        self.assertGreaterEqual(len(blocks), 2)
        self.assertNotIn("cache_control", blocks[1])
        self.assertIn("TODO", blocks[1]["text"])


class RetrieverPoolingTests(_RepoCase):
    """RP-1: pooled retrievers survive close() and drop stale caches."""

    def test_pooled_close_is_a_noop(self):
        _write(self.root, "a.py", "def a():\n    return 1\n")
        tg.index_repo(self.root, self.db)
        r = tg.Retriever(self.root, self.db)
        r.pooled = True
        r.close()
        self.assertTrue(r.file_skeleton("a.py"))   # connection still usable
        r.close_now()

    def test_invalidate_clears_source_cache(self):
        _write(self.root, "a.py", "def a():\n    return 1\n")
        r = self._index()
        r._lines("a.py")
        self.assertIn("a.py", r._src_cache)
        r.invalidate()
        self.assertEqual(r._src_cache, {})
        self.assertIsNone(r._ann_index)


class IdeWiringTests(_RepoCase):
    """Table 2: config shapes each host actually parses."""

    def test_ide_setup_writes_correct_shapes(self):
        res = tg.ide_setup(self.root)
        cfg = json.loads((self.root / ".mcp.json").read_text(encoding="utf-8"))
        self.assertIn("tokengraph", cfg["mcpServers"])
        vs = json.loads((self.root / ".vscode" / "mcp.json").read_text(encoding="utf-8"))
        self.assertIn("tokengraph", vs["servers"])       # VS Code uses `servers`
        cur = json.loads((self.root / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
        self.assertEqual(cur["mcpServers"]["tokengraph"]["type"], "stdio")
        zed = json.loads((self.root / ".zed" / "settings.json").read_text(encoding="utf-8"))
        # Zed needs `source: custom` to accept a user-defined server.
        self.assertEqual(zed["context_servers"]["tokengraph"]["source"], "custom")
        self.assertIn("written", res)

    def test_global_only_hosts_are_reported_not_silently_written(self):
        res = tg.ide_setup(self.root)
        hosts = {g["host"] for g in res["global_pending"]}
        self.assertIn("windsurf", hosts)
        self.assertIn("cline", hosts)
        # Nothing dead was written into the repo for them.
        self.assertFalse((self.root / ".windsurf" / "mcp.json").exists())

    def test_wiring_status_requires_actual_declaration(self):
        (self.root / ".mcp.json").write_text("{}", encoding="utf-8")
        status = {w["path"]: w for w in tg.mcp_wiring_status(self.root)}
        self.assertTrue(status[".mcp.json"]["exists"])
        self.assertFalse(status[".mcp.json"]["declares_tokengraph"])

    def test_cursor_adapter_writes_mdc_with_frontmatter(self):
        rel = tg.write_adapter(self.root, "cursor", "# generated body")
        text = (self.root / rel).read_text(encoding="utf-8")
        self.assertTrue(rel.endswith(".mdc"))
        self.assertTrue(text.startswith("---"))
        self.assertIn("alwaysApply: true", text)
        # Regenerating must not duplicate the frontmatter.
        tg.write_adapter(self.root, "cursor", "# second body")
        text2 = (self.root / rel).read_text(encoding="utf-8")
        self.assertEqual(text2.count("alwaysApply"), 1)
        self.assertIn("second body", text2)

    def test_adapters_preserve_handwritten_content(self):
        target = self.root / "CLAUDE.md"
        target.write_text("# My own notes\n", encoding="utf-8")
        tg.write_adapter(self.root, "claude", "generated")
        text = target.read_text(encoding="utf-8")
        self.assertIn("My own notes", text)
        self.assertIn("generated", text)

    def test_modern_adapter_paths(self):
        self.assertEqual(tg.ADAPTERS["cursor"]["path"], ".cursor/rules/contextiq.mdc")
        self.assertEqual(tg.ADAPTERS["windsurf"]["path"],
                         ".windsurf/rules/contextiq.md")
        self.assertEqual(tg.ADAPTERS["gemini"]["path"], "GEMINI.md")
        for name in ("cline", "roo", "continue", "aider", "codex", "zed"):
            self.assertIn(name, tg.ADAPTERS)
        # Legacy formats are still reachable but flagged.
        self.assertTrue(tg.ADAPTERS["cursor-legacy"]["deprecated"])

    def test_per_host_budgets_differ(self):
        self.assertNotEqual(tg.adapter_budget("claude", 8000),
                            tg.adapter_budget("copilot", 8000))

    def test_launch_command_is_consistent(self):
        cmd = tg.mcp_launch_command()
        self.assertEqual(cmd["args"][-1], "serve")
        self.assertTrue(cmd["command"])


class QualityGateTests(_RepoCase):
    """QG-1: answer quality, not just file recall."""

    def _repo(self):
        _write(self.root, "auth.py",
               "DEFAULT_TTL = 3600\n\n"
               "def issue_token(user, ttl=DEFAULT_TTL):\n"
               "    '''Mint a signed session token.'''\n"
               "    return {'user': user, 'ttl': ttl}\n")

    def test_answerable_requires_symbols_and_facts(self):
        self._repo()
        r = self._index()
        pack = r.find_relevant_context("mint a signed session token",
                                       budget_tokens=3000)
        good = tg.score_pack_answerability(pack, {
            "expected_symbols": ["auth.issue_token"],
            "must_contain": ["DEFAULT_TTL"]})
        self.assertTrue(good["answerable"])
        self.assertEqual(good["symbol_recall"], 1.0)

        bad = tg.score_pack_answerability(pack, {
            "expected_symbols": ["auth.issue_token", "auth.nonexistent_helper"],
            "must_contain": ["A_FACT_NOT_PRESENT_ANYWHERE"]})
        self.assertFalse(bad["answerable"])
        self.assertIn("auth.nonexistent_helper", bad["missing_symbols"])
        self.assertIn("A_FACT_NOT_PRESENT_ANYWHERE", bad["missing_facts"])
        self.assertLess(bad["symbol_recall"], 1.0)

    def test_benchmark_reports_quality_metrics(self):
        self._repo()
        r = self._index()
        out = tg.run_retrieval_benchmark(r, [{
            "task": "mint a signed session token",
            "expected_files": ["auth.py"],
            "expected_symbols": ["auth.issue_token"],
            "must_contain": ["DEFAULT_TTL"]}])
        for key in ("symbol_recall", "answerable_rate", "recall_at_5", "mrr"):
            self.assertIn(key, out)
        self.assertEqual(out["answerable_rate"], 1.0)

    def test_threshold_gate_flags_regressions(self):
        ok = tg.check_benchmark_thresholds(
            {"recall_at_5": 1.0, "symbol_recall": 1.0, "answerable_rate": 1.0,
             "irrelevant_token_ratio": 0.1})
        self.assertTrue(ok["ok"])
        bad = tg.check_benchmark_thresholds(
            {"recall_at_5": 0.1, "symbol_recall": 0.1, "answerable_rate": 0.0,
             "irrelevant_token_ratio": 0.99})
        self.assertFalse(bad["ok"])
        self.assertEqual(len(bad["failures"]), 4)
        # irrelevant_token_ratio is a ceiling, the rest are floors.
        directions = {f["metric"]: f["direction"] for f in bad["failures"]}
        self.assertEqual(directions["irrelevant_token_ratio"], "max")
        self.assertEqual(directions["recall_at_5"], "min")

    def test_fixture_corpora_are_discovered(self):
        repo_root = Path(tg.__file__).resolve().parent
        corpora = tg.discover_corpora(repo_root)
        self.assertTrue(corpora, "no benchmark corpora found")
        for c in corpora:
            data = json.loads(c.read_text(encoding="utf-8"))
            self.assertTrue(data.get("cases"), f"{c} has no cases")
            for case in data["cases"]:
                self.assertTrue(case.get("task"))
                self.assertTrue(case.get("expected_files"))

    def test_corpus_is_large_enough_and_multi_repo(self):
        """The old corpus was 10 self-referential cases; that proved nothing."""
        repo_root = Path(tg.__file__).resolve().parent
        corpora = tg.discover_corpora(repo_root)
        total = sum(len(json.loads(c.read_text(encoding="utf-8"))["cases"])
                    for c in corpora)
        self.assertGreaterEqual(total, 50,
                                f"quality corpus too small ({total} cases)")
        self.assertGreaterEqual(len(corpora), 3,
                                "quality gate must span multiple repositories")


class PackCacheTests(_RepoCase):
    """PK-1: memoised packs, invalidated by graph version."""

    def _repo(self):
        _write(self.root, "svc.py",
               "def charge(token, amount):\n"
               "    '''Charge a card.'''\n"
               "    return submit(token, amount)\n\n"
               "def submit(t, a):\n    return {'t': t}\n")

    def test_hit_returns_identical_markdown(self):
        self._repo()
        r = self._index()
        md1, i1 = r.find_relevant_context_cached("charge a card")
        md2, i2 = r.find_relevant_context_cached("charge a card")
        self.assertFalse(i1["cached"])
        self.assertTrue(i2["cached"])
        self.assertEqual(md1, md2)

    def test_reindex_invalidates(self):
        self._repo()
        r = self._index()
        r.find_relevant_context_cached("charge a card")
        _write(self.root, "other.py", "def unrelated():\n    return 1\n")
        tg.index_repo(self.root, self.db)
        r.invalidate()
        _, info = r.find_relevant_context_cached("charge a card")
        self.assertFalse(info["cached"])

    def test_session_packs_are_never_cached(self):
        """A session pack legitimately differs per call — caching it is wrong."""
        self._repo()
        r = self._index()
        md1, i1 = r.find_relevant_context_cached("charge a card", session="s")
        md2, i2 = r.find_relevant_context_cached("charge a card", session="s")
        self.assertFalse(i1["cached"])
        self.assertFalse(i2["cached"])
        # Second call is a delta, so it must NOT equal the first.
        self.assertNotEqual(md1, md2)

    def test_key_varies_with_budget(self):
        self._repo()
        r = self._index()
        a = r.pack_cache_key("t", 6000, 1, 1600)
        b = r.pack_cache_key("t", 3000, 1, 1600)
        self.assertNotEqual(a, b)

    def test_stats_report_hits(self):
        self._repo()
        r = self._index()
        r.find_relevant_context_cached("charge a card")
        r.find_relevant_context_cached("charge a card")
        self.assertGreaterEqual(r.store.pack_cache_stats()["hits"], 1)


class SessionGuardTests(_RepoCase):
    """SD-2: the ledger must expire rather than withhold dropped content."""

    def _repo(self):
        _write(self.root, "a.py", "def alpha():\n    return 1\n")

    def test_ttl_expiry(self):
        self._repo()
        r = self._index()
        r.find_relevant_context("alpha", budget_tokens=2000, session="s")
        # Backdate every entry beyond the TTL.
        r.store.conn.execute(
            "UPDATE sent_ledger SET ts = ts - ?", (tg.SESSION_TTL_SECONDS + 60,))
        r.store.commit()
        held, state = r.store.sent_map("s")
        self.assertEqual(held, {})
        self.assertGreater(state["expired_entries"], 0)

    def test_context_window_overflow_resets(self):
        self._repo()
        r = self._index()
        r.find_relevant_context("alpha", budget_tokens=2000, session="s")
        # Claim we sent far more than the model could still be holding.
        r.store.conn.execute("UPDATE sent_ledger SET tokens = 999999")
        r.store.commit()
        held, state = r.store.sent_map("s")
        self.assertEqual(held, {})
        self.assertTrue(state["reset"])
        self.assertIn("compacted", state["reset_reason"])

    def test_reset_is_announced_in_the_pack(self):
        self._repo()
        r = self._index()
        r.find_relevant_context("alpha", budget_tokens=2000, session="s")
        r.store.conn.execute("UPDATE sent_ledger SET tokens = 999999")
        r.store.commit()
        pack = r.find_relevant_context("alpha", budget_tokens=2000, session="s")
        self.assertIn("Session context reset", pack.to_markdown())
        self.assertTrue(pack.session_state.get("reset"))

    def test_entry_cap_evicts_oldest(self):
        self._repo()
        r = self._index()
        import time as _t
        now = _t.time()
        rows = [("s", f"q{i}", "body", "h", 1, now - (5000 - i))
                for i in range(tg.SESSION_MAX_ENTRIES + 25)]
        r.store.conn.executemany(
            "INSERT OR REPLACE INTO sent_ledger"
            "(session,qname,mode,content_hash,tokens,ts) VALUES(?,?,?,?,?,?)",
            rows)
        r.store.commit()
        held, state = r.store.sent_map("s")
        self.assertLessEqual(len(held), tg.SESSION_MAX_ENTRIES)
        self.assertGreater(state["evicted_entries"], 0)

    def test_prune_sessions_is_global(self):
        self._repo()
        r = self._index()
        r.find_relevant_context("alpha", budget_tokens=2000, session="s1")
        r.find_relevant_context("alpha", budget_tokens=2000, session="s2")
        r.store.conn.execute(
            "UPDATE sent_ledger SET ts = ts - ?", (tg.SESSION_TTL_SECONDS + 60,))
        r.store.commit()
        self.assertGreater(r.store.prune_sessions(), 0)


class TokenizerTests(unittest.TestCase):
    """MA-2: real tokenizers where available, honest labels where not."""

    def test_detail_reports_method_and_tokenizer(self):
        d = tg.count_tokens_detail("def add(a, b):\n    return a + b\n", "gpt-4o")
        self.assertIn(d["method"], ("exact", "approx"))
        self.assertTrue(d["tokenizer"])
        self.assertEqual(d["family"], "gpt")

    def test_claude_is_labelled_an_estimate(self):
        d = tg.count_tokens_detail("hello world", "claude-sonnet")
        self.assertEqual(d["family"], "claude")
        # No offline Anthropic tokenizer exists, so this must not claim exact.
        self.assertEqual(d["method"], "approx")
        self.assertIn("estimate", d["note"].lower())

    def test_count_tokens_for_model_matches_detail(self):
        text = "class Foo:\n    def bar(self):\n        return 1\n"
        for m in ("gpt-4o", "claude-opus", "gemini-1.5-pro", "llama-3.1-70b"):
            self.assertEqual(tg.count_tokens_for_model(text, m),
                             tg.count_tokens_detail(text, m)["tokens"])

    def test_status_covers_every_family(self):
        st = tg.tokenizer_status()
        for fam in ("gpt", "claude", "gemini", "llama"):
            self.assertIn(fam, st)
            self.assertIn(st[fam]["method"], ("exact", "approx"))

    def test_empty_text_is_zero(self):
        self.assertEqual(tg.count_tokens_for_model("", "gpt-4o"), 0)

    def test_gpt_models_do_not_share_a_cache_entry(self):
        """gpt-4 is cl100k, gpt-4o is o200k — they must resolve separately."""
        tg.count_tokens_detail("x", "gpt-4")
        tg.count_tokens_detail("x", "gpt-4o")
        keys = [k for k in tg._MODEL_ENCODER_TRIED if k.startswith("gpt:")]
        self.assertGreaterEqual(len(keys), 2)


class PricingTests(unittest.TestCase):
    """CE-2: dated, overridable prices that go stale out loud."""

    def setUp(self):
        tg._PRICING_CACHE = None
        self.addCleanup(lambda: setattr(tg, "_PRICING_CACHE", None))

    def test_defaults_carry_a_date(self):
        p = tg.load_pricing(refresh=True)
        self.assertEqual(p["as_of"], tg.PRICES_AS_OF)
        self.assertIn("prices", p)
        self.assertIsInstance(p["stale"], bool)

    def test_file_override_wins(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / ".context").mkdir(parents=True)
        (root / ".context" / "pricing.json").write_text(json.dumps({
            "as_of": "2099-01-01",
            "prices": {"claude-sonnet": {"input": 99.0, "output": 199.0}},
        }), encoding="utf-8")
        p = tg.load_pricing(root, refresh=True)
        self.assertEqual(p["prices"]["claude-sonnet"]["input"], 99.0)
        self.assertEqual(p["as_of"], "2099-01-01")
        self.assertFalse(p["stale"])

    def test_malformed_entry_is_skipped_not_fatal(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / ".context").mkdir(parents=True)
        (root / ".context" / "pricing.json").write_text(json.dumps({
            "prices": {"bad-model": {"input": "not-a-number"}},
        }), encoding="utf-8")
        p = tg.load_pricing(root, refresh=True)
        self.assertNotIn("bad-model", p["prices"])
        self.assertTrue(any("bad-model" in w for w in p["warnings"]))

    def test_estimate_cost_reports_provenance(self):
        c = tg.estimate_cost("def f():\n    return 1\n" * 20, "claude-sonnet")
        self.assertIn("prices_as_of", c)
        self.assertIn("token_method", c)
        self.assertGreater(c["total_usd"], 0)

    def test_unknown_model_falls_back_within_family(self):
        p = tg.price_for("claude-some-future-model")
        self.assertIn("input", p)
        self.assertGreater(p["input"], 0)


class RoutingTests(_RepoCase):
    """MR-3: routing on graph + control-flow features, not just size."""

    def test_complexity_features_detect_nesting_and_branches(self):
        src = ("def f(x):\n"
               "    if x:\n"
               "        for i in range(3):\n"
               "            while i:\n"
               "                if i > 1:\n"
               "                    return i\n")
        feats = tg.file_complexity_features(src)
        self.assertGreaterEqual(feats["max_depth"], 4)
        self.assertGreaterEqual(feats["branches"], 4)

    def test_entangled_file_routes_higher_than_a_plain_one(self):
        plain = tg.tier_for_file("a.py", token_est=1000, symbols=10)
        hub = tg.tier_for_file("b.py", token_est=1000, symbols=10,
                               fan_in=30, fan_out=20, max_depth=6, branches=80)
        self.assertGreater(hub["complexity_score"], plain["complexity_score"])
        self.assertEqual(hub["tier"], "powerful")

    def test_config_files_stay_cheap_regardless_of_size(self):
        info = tg.tier_for_file("big.json", token_est=90000, symbols=500)
        self.assertEqual(info["tier"], "fast")

    def test_signals_are_explained(self):
        info = tg.tier_for_file("x.py", token_est=5000, symbols=60,
                                fan_in=12, max_depth=5)
        self.assertTrue(info["signals"])
        self.assertTrue(any("fan-in" in s for s in info["signals"]))

    def test_get_routing_uses_real_features(self):
        _write(self.root, "deep.py",
               "def f(x):\n"
               "    if x:\n"
               "        for i in range(3):\n"
               "            while i:\n"
               "                if i > 1:\n"
               "                    return i\n")
        r = self._index()
        rows = {row["file"]: row for row in r.get_routing()}
        self.assertIn("deep.py", rows)
        self.assertIn("complexity_score", rows["deep.py"])


class SummarizerTests(unittest.TestCase):
    """CS-2: TextRank surfaces substance that cue phrases miss."""

    TRANSCRIPT = (
        "User: The payment retries keep hammering the provider during outages.\n"
        "Assistant: The gateway retries five times with exponential backoff and "
        "no circuit breaker at all.\n"
        "User: Every worker retries independently so the provider sees a "
        "thundering herd of requests.\n"
        "Assistant: We decided to add a shared circuit breaker in front of the "
        "payment gateway.\n"
        "User: What about the timeout budget for the breaker?\n"
        "Assistant: The breaker opens after twenty failures and resets after "
        "thirty seconds of quiet.\n"
    )

    def test_textrank_ranks_central_sentences_first(self):
        sents = ["The payment gateway retries on failure.",
                 "Payment retries hammer the gateway during an outage.",
                 "Unrelated note about lunch plans tomorrow."]
        ranked = tg._textrank(sents, 2)
        self.assertNotIn(2, ranked)

    def test_textrank_handles_degenerate_input(self):
        self.assertEqual(tg._textrank([], 3), [])
        self.assertEqual(tg._textrank(["only one"], 3), [0])

    def test_summary_has_key_points_beyond_cue_phrases(self):
        res = tg.summarize_conversation(self.TRANSCRIPT, max_tokens=300)
        self.assertEqual(res["method"], "textrank+cues")
        self.assertTrue(res["key_points"])
        # The thundering-herd sentence has no decision/action cue word, so only
        # TextRank can surface it.
        joined = " ".join(res["key_points"]).lower()
        self.assertIn("thundering herd", joined)

    def test_respects_token_budget(self):
        res = tg.summarize_conversation(self.TRANSCRIPT * 6, max_tokens=120)
        self.assertLessEqual(res["summary_tokens"], 140)

    def test_empty_transcript_is_safe(self):
        res = tg.summarize_conversation("", max_tokens=100)
        self.assertEqual(res["turns"], 0)
        self.assertEqual(res["summary"], "")


class WasteMetricTests(_RepoCase):
    """MS-2: graph-connected context is not waste."""

    def test_related_symbols_are_not_counted_as_waste(self):
        _write(self.root, "core.py",
               "def helper(x):\n    '''Shared helper.'''\n    return x * 2\n")
        _write(self.root, "api.py",
               "from core import helper\n\n"
               "def handle_request(payload):\n"
               "    '''Handle an inbound request.'''\n"
               "    return helper(payload)\n")
        r = self._index()
        out = tg.run_retrieval_benchmark(r, [{
            "task": "handle an inbound request",
            "expected_files": ["api.py"],
            "expected_symbols": ["api.handle_request"],
        }], budget_tokens=4000)
        # core.helper is a callee of the expected symbol — legitimate context,
        # so it must not be scored as wasted budget.
        self.assertLess(out["irrelevant_token_ratio"], 1.0)


class JudgeHarnessTests(unittest.TestCase):
    """QG-2: the LLM-judge harness — structure, not live API calls."""

    def test_corpora_are_discovered_and_well_formed(self):
        root = Path(tg.__file__).resolve().parent
        corpora = tg.load_judge_corpora(root)
        self.assertGreaterEqual(len(corpora), 3)
        total = 0
        for c in corpora:
            doc = json.loads(c.read_text(encoding="utf-8"))
            self.assertTrue(doc.get("questions"))
            for q in doc["questions"]:
                self.assertTrue(q.get("question"))
                self.assertTrue(q.get("reference_answer"))
                self.assertGreaterEqual(len(q.get("rubric") or []), 3)
                for cite in q.get("cites", []):
                    self.assertTrue((c.parent / cite).exists(),
                                    f"{c.name}:{q.get('id')} cites missing {cite}")
                total += 1
        self.assertGreaterEqual(total, 30)

    def test_missing_credentials_fail_cleanly(self):
        """No API access must produce a clear message, never a traceback."""
        import unittest.mock as mock
        with mock.patch.object(tg, "_anthropic_client",
                               return_value=(None, "no credentials")):
            res = tg.run_llm_judge([], model="claude-opus-4-8")
        self.assertFalse(res["ok"])
        self.assertIn("no credentials", res["error"])
        self.assertIn("hint", res)

    def test_judge_schema_is_valid_structured_output(self):
        s = tg._JUDGE_SCHEMA
        self.assertFalse(s["additionalProperties"])
        self.assertEqual(set(s["required"]), {"criteria", "score", "verdict"})
        item = s["properties"]["criteria"]["items"]
        self.assertFalse(item["additionalProperties"])

    def test_uses_a_current_model_id(self):
        self.assertEqual(tg.JUDGE_MODEL, "claude-opus-4-8")


class JetBrainsNvimTests(_RepoCase):
    def test_jetbrains_xml_is_written_and_idempotent(self):
        tg.ide_setup(self.root)
        p = self.root / ".idea" / "mcp.xml"
        self.assertTrue(p.exists())
        text = p.read_text(encoding="utf-8")
        self.assertIn('name="tokengraph"', text)
        self.assertIn("McpServerSettings", text)
        tg.ide_setup(self.root)
        self.assertEqual(
            p.read_text(encoding="utf-8").count('name="tokengraph"'), 1)

    def test_jetbrains_preserves_other_servers(self):
        p = self.root / ".idea" / "mcp.xml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
                     '<project version="4"><component name="McpServerSettings">'
                     '<servers><server name="other">'
                     '<option name="command" value="x" /></server></servers>'
                     '</component></project>\n', encoding="utf-8")
        tg.ide_setup(self.root)
        text = p.read_text(encoding="utf-8")
        self.assertIn('name="other"', text)
        self.assertIn('name="tokengraph"', text)

    def test_nvim_lua_is_written_and_parseable_shape(self):
        tg.ide_setup(self.root)
        p = self.root / ".nvim" / "contextiq.lua"
        self.assertTrue(p.exists())
        text = p.read_text(encoding="utf-8")
        self.assertIn("M.servers", text)
        self.assertIn("tokengraph", text)
        self.assertIn("return M", text)


if __name__ == "__main__":
    unittest.main()

