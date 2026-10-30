# -*- coding: utf-8 -*-
"""
Single-shot RAG baseline (the "normal" RAG that agentic_rag.py is compared against).

    question -> embed -> FAISS top-k -> one prompt with the passages -> one Groq call -> answer

There is no loop and no tool use: the search query is the user's question, exactly as typed,
and retrieval happens once. rag_eval.py imports the prompt and message builder from here, so
the baseline in the evaluation is exactly this one.

Usage
    python single_shot_rag.py --folder "path/to/pdfs"
    python single_shot_rag.py --folder "path/to/pdfs" -q "Question" -k 5
"""

import argparse

import agentic_rag as ar
from agentic_rag import build_knowledge_base, clean_citations, groq_chat, retrieve

ABSTAIN_PHRASE = "I don't know based on the available information."

SINGLE_SYSTEM = (
    "You are a careful research assistant. Answer the user's question using ONLY the numbered context "
    "passages provided. Cite the passages you use like [1], [2]. If the answer is not in the context, "
    f'reply exactly: "{ABSTAIN_PHRASE}"'
)

DEFAULT_K = 5   # passages retrieved per question


def build_single_shot_messages(question: str, retrieved: list[dict]) -> list[dict]:
    """The prompt: system instruction + numbered context passages + question + 'Answer:' suffix."""
    context = "\n\n".join(f"[{i}] (source: {r['source']})\n{r['text']}" for i, r in enumerate(retrieved, 1))
    return [
        {"role": "system", "content": SINGLE_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
    ]


def ask(kb, question: str, k: int = DEFAULT_K, verbose: bool = True) -> str:
    retrieved = retrieve(kb, question, k)
    payload = {"model": ar.GROQ_MODEL, "messages": build_single_shot_messages(question, retrieved),
               "temperature": 0.1, "max_tokens": ar.MAX_OUTPUT_TOKENS}
    if ar.REASONING_EFFORT:
        payload["reasoning_effort"] = ar.REASONING_EFFORT
    msg, usage, _ = groq_chat(payload)
    answer = clean_citations((msg.get("content") or "").strip())

    if verbose:
        print("=" * 70)
        print(f"❓ {question}")
        print("📄 Retrieved:")
        for i, r in enumerate(retrieved, 1):
            print(f"  [{i}] score={r['score']:.3f}  {r['source']} (p.{r['page']})")
        print(f"\n🤖 Answer:\n{answer}")
        print(f"\n📊 1 LLM call, {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion tokens")
        print("=" * 70 + "\n")
    return answer


def main():
    ap = argparse.ArgumentParser(description="Single-shot RAG over a folder of PDFs (Groq)")
    ap.add_argument("--folder", default=ar.PDF_FOLDER, help="folder with PDFs; subfolders are included")
    ap.add_argument("-q", "--question", help="ask one question and exit (otherwise interactive)")
    ap.add_argument("-k", type=int, default=DEFAULT_K, help="passages to retrieve per question")
    args = ap.parse_args()

    kb = build_knowledge_base(args.folder)
    if args.question:
        ask(kb, args.question, args.k)
        return

    print("Type a question (or 'quit' to exit).")
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            ask(kb, q, args.k)


if __name__ == "__main__":
    main()
