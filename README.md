# CPU_rag: agentic RAG over your PDFs (FAISS + Groq)

Ask questions about a folder of PDFs (subfolders included) and get answers that cite the file and passage they came from. Embedding and search run locally on CPU; the answering model runs on [Groq](https://groq.com).

The repo has two parts:

1. **An agentic RAG system**: the LLM gets a search tool and decides what to search for, how many times, and when it has enough evidence, instead of retrieving once.
2. **An automated evaluation** that compares it with a plain single-shot RAG and a no-retrieval baseline on accuracy, latency and token cost. It generates its own synthetic corpus and questions, so it needs no PDFs and runs the same way every time.

![Agentic RAG block diagram](docs/agentic_rag_diagram.svg)

## How it works

**Indexing (once, at startup).** Every `.pdf` under the folder is found, text is extracted page by page, split into overlapping 200-word chunks (file name and page are kept), embedded with `all-MiniLM-L6-v2`, and stored in a FAISS inner-product index (cosine similarity).

**Agent loop (per question).** The model is given tools and a system prompt, and replies either with a tool call or a final answer. Tool results are fed back and the model is called again, up to 6 rounds.

| Tool | What it does |
|---|---|
| `search_papers(query)` | Embeds the query, returns the top-3 nearest chunks from the index as numbered passages with file and page |
| `list_papers()` | Lists the files in the library |
| `search_web(query)` | Optional DuckDuckGo search, **off by default** (`--web` to enable; the model-written query leaves your machine) |

Practical details: passages already sent are not resent (saves tokens), passage numbers stay consistent across searches so `[n]` citations are unambiguous, rate-limit (429) errors are waited out and retried, and every Groq call reports its token usage.

**Single-shot baseline** (`single_shot_rag.py`): embed the question as typed, retrieve the top-k chunks once, build one prompt, make one LLM call.

## Quick start

Python 3.10 or newer.

```bash
git clone https://github.com/Rimas-Codes/CPU_rag.git
cd CPU_rag

python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Get a free API key at [console.groq.com/keys](https://console.groq.com/keys) and provide it in one of these ways (never commit it):

- an environment variable: `GROQ_API_KEY=gsk_...`
- a `.env` file: copy `.env.example` to `.env` and edit it (`.env` is git-ignored)
- or paste it when the script asks

Run it:

```bash
python agentic_rag.py --folder "path/to/your/pdfs"                 # interactive
python agentic_rag.py --folder "path/to/your/pdfs" -q "Your question"
python agentic_rag.py --folder "path/to/your/pdfs" --web           # also allow web search

python single_shot_rag.py --folder "path/to/your/pdfs"             # the plain baseline
```

Without `--folder` it uses `./pdfs` (or the `RAG_PDF_FOLDER` environment variable). The default model is `openai/gpt-oss-120b`; change `GROQ_MODEL` at the top of `agentic_rag.py` if it is not available on your account.

## Evaluation

```bash
python rag_eval.py --per-category 2    # quick smoke test
python rag_eval.py                     # full run: 25 questions x 3 systems
python rag_eval.py --systems single,agentic
```

**What it generates.** A made-up world of 12 fictional institutes with projects, lead researchers and universities, written out as 44 short documents. Names are fictional, so no model can answer from training data. It then writes 25 questions with known answers:

- **10 simple**: the answer is in one document.
- **10 multi-hop**: the answer needs 2-3 documents chained together. The entity in the question appears only in the first document ("Who leads the flagship project of the Brisnar Institute?"), so a single search on the question's words is unlikely to find the second document.
- **5 unanswerable**: about real entities, but the fact was never written down (budget, programming language, current director, thesis title, a non-existent institute). A good system says it does not know.

**Systems compared.** `closed` (LLM alone, a sanity check that the data is not leaked), `single` (top-5 chunks, one call), `agentic` (this repo's agent, web search off).

**Judging.** A different model (`gpt-oss-20b`) compares each answer with the known gold answer. For unanswerable questions an answer counts as correct only if it abstains and states no specific answer. The judge's reason is saved for every question in `results.jsonl`.

**Results** (seed 42, answering model `gpt-oss-120b`; full tables in [`results/seed42/summary.md`](results/seed42/summary.md)):

| System | Overall | Simple | Multi-hop | Unanswerable | Latency (s) | Tokens / question | $ / question |
|---|---|---|---|---|---|---|---|
| closed-book | 4/25 (16%) | 0/10 | 0/10 | 4/5 | 0.4 | 157 | 0.00004 |
| single-shot | 17/25 (68%) | 10/10 | 2/10 | 5/5 | 0.5 | 553 | 0.00010 |
| **agentic** | **25/25 (100%)** | 10/10 | **10/10** | 5/5 | 1.1 | 1,720 | 0.00031 |

95% confidence intervals: overall 48-83% (single-shot) vs 87-100% (agentic); multi-hop 6-51% vs 72-100%. Latency excludes time spent waiting on Groq free-tier rate limits.

What the numbers show:

- Single-shot is as good as the agent on simple questions. It fails multi-hop questions mostly by retrieving the first document but not the second: the needed documents were all retrieved for only 2 of 10 multi-hop questions (the agent: 10 of 10). Its failures are "I don't know", not hallucinations.
- The agent adapts its effort: 1 search for every simple question, 2-3 for multi-hop (matching the length of the chain).
- It costs about 3x the tokens and 2.4x the latency of single-shot, and about twice as much on simple questions where it gains nothing. On multi-hop questions it is cheaper per correct answer ($0.00043 vs $0.00055).
- The closed-book baseline scores 0/20 on answerable questions, confirming the benchmark cannot be solved from model knowledge.

**Read these results with care.**

- The benchmark is synthetic, clean and small (44 documents, 25 questions). Real PDFs are messier, so absolute scores will be lower.
- The multi-hop questions were designed so that one search cannot find the second document, so the 20% vs 100% gap demonstrates the mechanism, not a general real-world gap.
- The single-shot baseline retrieves 5 chunks; in this run the agent saw up to 9 across its searches. A single-shot control with a larger k would show how much of the gain is just "more chunks" rather than iterative search (not run yet).
- With 5-10 questions per category, differences of one or two questions are noise.
- The judge is an LLM and can be wrong; spot-check `results.jsonl`.
- Token prices are set in `PRICES` in `rag_eval.py`; check Groq's pricing page.

Results are saved after every question, so an interrupted run resumes where it stopped (`--fresh` starts over, `--seed N` generates a different corpus and question set).

## Project structure

```
agentic_rag.py        indexing + Groq client + agent loop + CLI
single_shot_rag.py    single-shot baseline (also used by the evaluation)
rag_eval.py           synthetic corpus, questions, LLM judge, metrics, report
tests/                unit tests (no API key or network needed): pytest -q
results/seed42/       corpus, questions, per-question results and summary of the run above
docs/                 block diagram
```

## Limitations and ideas

- Scanned PDFs have no text layer and are skipped (no OCR).
- The index is rebuilt in memory on every run; caching it to disk would remove the start-up wait on large libraries.
- Fixed-size word chunks ignore paper structure (sections, tables, figures). No reranking and no hybrid (keyword + vector) search.
- Retrieval returns a fixed 3 chunks per search call; the free Groq tier (token-per-minute limits) is the main constraint on latency.
- Not done yet: single-shot control with larger k, evaluation on real papers, multiple seeds averaged, a page-restricted search tool.

## Troubleshooting

- **`OSError: [WinError 1114] ... c10.dll`** when importing torch: install the CPU build in a clean environment (command in Quick start) and make sure your IDE uses that environment.
- **The script seems to hang after the embedding step**: it is waiting for the API key. Set `GROQ_API_KEY`, or paste the key into the console.
- **`Groq API error 404 ... model_not_found`**: the model is not available on your account; set another one in `GROQ_MODEL`.
- **`429` rate limit messages**: normal on the free tier; the script waits and retries.

## License

MIT, see [LICENSE](LICENSE).
