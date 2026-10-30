"""
Fast tests that need no API key and no network.
Run with:  pytest -q
"""
import json
from types import SimpleNamespace

import pytest

# the modules import these heavy libraries at the top; skip cleanly if they are not installed
for _lib in ("torch", "faiss", "sentence_transformers", "pypdf"):
    pytest.importorskip(_lib)

import agentic_rag as ar          # noqa: E402
import rag_eval as ev             # noqa: E402
import single_shot_rag as ss      # noqa: E402


# ---------------------------------------------------------------- indexing helpers
def test_chunk_text_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = ar.chunk_text(text, chunk_size=40, overlap=10)
    assert chunks[0].split()[0] == "w0" and len(chunks[0].split()) == 40
    assert chunks[1].split()[0] == "w30"           # step = 40 - 10
    assert chunks[-1].split()[-1] == "w99"         # nothing is lost at the end
    assert all(c.strip() for c in chunks)


def test_chunk_text_short_text_is_one_chunk():
    assert ar.chunk_text("just a few words", 200, 40) == ["just a few words"]


def test_find_pdfs_searches_subfolders(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    (tmp_path / "sub" / "b.PDF").write_bytes(b"")
    (tmp_path / "sub" / "deep" / "c.pdf").write_bytes(b"")
    (tmp_path / "sub" / "notes.txt").write_text("not a pdf")
    found = ar.find_pdfs(str(tmp_path))
    assert sorted(p.name for p in found) == ["a.pdf", "b.PDF", "c.pdf"]


def test_find_pdfs_missing_folder_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        ar.find_pdfs(str(tmp_path / "does_not_exist"))


def test_clean_citations_handles_both_marker_styles():
    assert ar.clean_citations("A【5†L1-L4】 and B【11】.") == "A[5] and B[11]."


def test_system_prompt_mentions_web_only_when_enabled():
    assert "search_web" in ar.build_system_prompt(True)
    assert "search_web" not in ar.build_system_prompt(False)


# ---------------------------------------------------------------- single-shot prompt
def test_single_shot_messages_number_passages_and_end_with_answer_suffix():
    msgs = ss.build_single_shot_messages("Why?", [{"source": "a.pdf", "text": "T1"}, {"source": "b.pdf", "text": "T2"}])
    user = msgs[1]["content"]
    assert "[1] (source: a.pdf)\nT1" in user and "[2] (source: b.pdf)\nT2" in user
    assert user.endswith("Question: Why?\n\nAnswer:")
    assert ss.ABSTAIN_PHRASE in msgs[0]["content"]


# ---------------------------------------------------------------- synthetic benchmark
def test_corpus_and_questions_are_deterministic():
    assert ev.generate_corpus_and_questions(42) == ev.generate_corpus_and_questions(42)
    assert ev.generate_corpus_and_questions(42) != ev.generate_corpus_and_questions(7)


def test_question_set_shape():
    docs, qs = ev.generate_corpus_and_questions(42)
    cats = [q["category"] for q in qs]
    assert (cats.count("simple"), cats.count("multi_hop"), cats.count("unanswerable")) == (10, 10, 5)
    assert len({q["id"] for q in qs}) == len(qs)
    assert len({d["id"] for d in docs}) == len(docs)


@pytest.mark.parametrize("seed", [1, 42, 123])
def test_gold_answers_really_are_in_the_support_documents(seed):
    docs, qs = ev.generate_corpus_and_questions(seed)
    text_by_id = {d["id"]: d["text"] for d in docs}
    for q in qs:
        if q["category"] == "unanswerable":
            assert q["support"] == []
            continue
        assert q["support"] and all(s in text_by_id for s in q["support"])
        joined = " ".join(text_by_id[s] for s in q["support"])
        assert q["gold"] in joined, f"{q['id']}: gold '{q['gold']}' missing from its support documents"
        if q["category"] == "multi_hop":
            assert len(q["support"]) >= 2


def test_unanswerable_topics_never_appear_in_the_corpus():
    docs, _ = ev.generate_corpus_and_questions(42)
    corpus = " ".join(d["text"] for d in docs).lower()
    for word in ["budget", "programming", "director", "thesis", "zephyrine"]:
        assert word not in corpus


# ---------------------------------------------------------------- metrics + judge parsing
def test_wilson_interval():
    assert ev.wilson(0, 0) == (0.0, 0.0)
    lo, hi = ev.wilson(10, 10)
    assert hi == 1.0 and lo > 0.6
    lo, hi = ev.wilson(5, 10)
    assert lo < 0.5 < hi


def test_cost_usd():
    # 1M prompt tokens + 1M completion tokens at 0.15 / 0.60
    assert ev.cost_usd("openai/gpt-oss-120b", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert ev.cost_usd("unknown-model", 1000, 1000) == 0.0


@pytest.mark.parametrize("reply, expected", [
    ('{"verdict": "CORRECT", "reason": "matches"}', True),
    ('{"verdict": "INCORRECT", "reason": "wrong name"}', False),
    ("Verdict: INCORRECT because it differs", False),     # 'INCORRECT' contains 'CORRECT': must not read as True
    ("no usable verdict here", None),
])
def test_judge_parsing(monkeypatch, reply, expected):
    monkeypatch.setattr(ev, "groq_chat",
                        lambda payload, retries=5: ({"content": reply}, {"prompt_tokens": 10, "completion_tokens": 5}, 0.0))
    assert ev.judge("Q?", "gold", "answer")["verdict"] is expected


# ---------------------------------------------------------------- agent loop with a fake LLM
def test_agent_loop_collects_evidence_and_stats(monkeypatch):
    monkeypatch.setattr(ar, "SAVE_PAYLOADS", False)   # do not write request files during tests
    monkeypatch.setattr(ar, "retrieve", lambda kb, q, k: [
        {"text": "The Veldris Institute is in Altamere.", "source": "lab_veldris.txt", "page": 1, "score": 0.9}])

    replies = iter([
        {"content": "", "tool_calls": [{"id": "c1", "type": "function",
                                        "function": {"name": "search_papers", "arguments": json.dumps({"query": "Veldris"})}}]},
        {"content": "It is in Altamere [1]."},
    ])
    monkeypatch.setattr(ar, "groq_chat",
                        lambda payload, retries=5: (next(replies), {"prompt_tokens": 100, "completion_tokens": 20}, 0.5))

    agent = ar.AgenticRAG(SimpleNamespace(chunks=[]), use_web=False)
    answer = agent.ask("Where is the Veldris Institute?", verbose=False)

    assert answer == "It is in Altamere [1]."
    assert agent.stats["llm_calls"] == 2
    assert agent.stats["prompt_tokens"] == 200 and agent.stats["completion_tokens"] == 40
    assert agent.stats["wait_s"] == 1.0
    assert agent.stats["tool_calls"] == ["search_papers"]
    assert [e["source"] for e in agent.evidence] == ["lab_veldris.txt"]


def test_agent_does_not_resend_passages_it_already_showed(monkeypatch):
    hit = {"text": "same passage", "source": "x.txt", "page": 1, "score": 0.5}
    monkeypatch.setattr(ar, "retrieve", lambda kb, q, k: [hit])
    agent = ar.AgenticRAG(SimpleNamespace(chunks=[]), use_web=False)
    first = agent.tool_search_papers("q1")
    second = agent.tool_search_papers("q2")
    assert "same passage" in first
    assert "already shown" in second and "same passage" not in second
    assert len(agent.evidence) == 1
