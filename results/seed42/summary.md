# RAG evaluation (seed 42)

Answer model: `openai/gpt-oss-120b`  |  Judge model: `openai/gpt-oss-20b`  |  Single-shot top-k: 5  |  Agent: max 6 rounds, 3 chunks/search, web search off

## Accuracy (LLM judge; 95% confidence interval in brackets)

| system | overall | simple | multi-hop | unanswerable |
|---|---|---|---|---|
| closed | 4/25 = 16% [6-35] | 0/10 = 0% [0-28] | 0/10 = 0% [0-28] | 4/5 = 80% [38-96] |
| single | 17/25 = 68% [48-83] | 10/10 = 100% [72-100] | 2/10 = 20% [6-51] | 5/5 = 100% [57-100] |
| agentic | 25/25 = 100% [87-100] | 10/10 = 100% [72-100] | 10/10 = 100% [72-100] | 5/5 = 100% [57-100] |

## Latency and token cost (per question, answering model only)

| system | latency mean (s) | latency median (s) | incl. rate-limit waits (s) | LLM calls | prompt tok | completion tok | total tok | $ / question | $ / 1,000 questions |
|---|---|---|---|---|---|---|---|---|---|
| closed | 0.4 | 0.5 | 0.4 | 1.0 | 114 | 43 | 157 | $0.00004 | $0.04 |
| single | 0.5 | 0.6 | 1.0 | 1.0 | 508 | 45 | 553 | $0.00010 | $0.10 |
| agentic | 1.1 | 1.0 | 8.4 | 2.7 | 1602 | 117 | 1720 | $0.00031 | $0.31 |

Latency excludes time spent sleeping on Groq rate limits (shown separately in the 'incl.' column).

## By question type

| system | type | accuracy | latency (s) | total tok |
|---|---|---|---|---|
| closed | simple | 0/10 = 0% [0-28] | 0.5 | 166 |
| closed | multi-hop | 0/10 = 0% [0-28] | 0.5 | 151 |
| closed | unanswerable | 4/5 = 80% [38-96] | 0.4 | 153 |
| single | simple | 10/10 = 100% [72-100] | 0.5 | 550 |
| single | multi-hop | 2/10 = 20% [6-51] | 0.5 | 574 |
| single | unanswerable | 5/5 = 100% [57-100] | 0.5 | 517 |
| agentic | simple | 10/10 = 100% [72-100] | 0.8 | 1084 |
| agentic | multi-hop | 10/10 = 100% [72-100] | 1.4 | 2374 |
| agentic | unanswerable | 5/5 = 100% [57-100] | 1.2 | 1681 |

## Retrieval: were the documents needed for the answer retrieved?

| system | type | avg support recall | all needed docs retrieved |
|---|---|---|---|
| single | simple | 100% | 10/10 |
| single | multi-hop | 57% | 2/10 |
| agentic | simple | 100% | 10/10 |
| agentic | multi-hop | 100% | 10/10 |

## Notes

- Judge overhead: 19,985 tokens (about $0.0024); judge calls with no usable verdict (counted as incorrect): 0.
- Total time spent waiting on rate limits: 195s.
- Prices used (USD per 1M tokens, input/output): openai/gpt-oss-120b 0.15/0.6, openai/gpt-oss-20b 0.075/0.3. Edit PRICES in rag_eval.py if they change.
- With 5-10 questions per category the confidence intervals are wide: treat differences of one or two questions as noise.
