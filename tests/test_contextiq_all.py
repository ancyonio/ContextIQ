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

            async with Client(server) as client:
                tools = await client.list_tools()
                names = {tool.name for tool in tools}
                self.assertIn("find_relevant_context", names)
                self.assertIn("get_symbol", names)

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

    # ---- hallucination reduction: reproducible multi-repo benchmark ----
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
        self.assertEqual(rep1["hallucination_reduction_pct"],
                         rep2["hallucination_reduction_pct"])
        self.assertEqual(rep1["modeled_with_grounding_per_100"],
                         rep2["modeled_with_grounding_per_100"])
        # a real, quantified, multi-repo figure with a spread
        self.assertGreater(rep1["hallucination_reduction_pct"], 0.0)
        self.assertEqual(len(rep1["reduction_spread_pct"]), 2)
        self.assertGreaterEqual(rep1["mean_grounding_coverage_pct"], 0.0)

    def test_hallucination_report_markdown(self):
        r = self._retriever()
        try:
            rep = tg.hallucination_benchmark(r, sample_per_repo=10)
        finally:
            r.close()
        md = tg.hallucination_report_to_markdown(rep)
        self.assertIn("hallucination", md.lower())
        self.assertIn("Reduction %", md)
        self.assertIn("| Repo |", md)

    # ---- IDE integration: one-command MCP wiring for every editor ----
    def test_ide_setup_writes_editor_configs(self):
        import json
        res = tg.ide_setup(self.root)
        self.assertEqual(set(res["written"]),
                         {".mcp.json", ".vscode/mcp.json", ".cursor/mcp.json",
                          ".windsurf/mcp.json", ".zed/settings.json"})
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
        self.assertGreaterEqual(tg.count_tokens_for_model("", "claude-opus"), 1)

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


if __name__ == "__main__":
    unittest.main()

