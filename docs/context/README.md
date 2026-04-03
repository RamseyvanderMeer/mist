# MIST documentation context (topic tree)

**Start here for AI onboarding.**

1. Skim **[OVERVIEW.md](OVERVIEW.md)** (one page).
2. For implementation work, read **[core/SPEC.md](core/SPEC.md)** and **[core/REPO_RULES_FOR_AI.md](core/REPO_RULES_FOR_AI.md)**.
3. Open **only the topic folder** you need from the table below.

---

## Topic → folder

| You are working on… | Open |
|----------------------|------|
| HTTP API, auth, rate limits, env vars, repo layout | [core/SPEC.md](core/SPEC.md), [core/ARCHITECTURE.md](core/ARCHITECTURE.md) |
| Where to put code, tests, safety | [core/REPO_RULES_FOR_AI.md](core/REPO_RULES_FOR_AI.md) |
| DiagDocDb, xmlvalueprimitive, ISTA tables | [data/DATABASE.md](data/DATABASE.md), [data/ISTA_DATABASE_GUIDE.md](data/ISTA_DATABASE_GUIDE.md) |
| Vector search, rerank, similarity vs LLM match | [retrieval/RETRIEVAL_MISMATCH_INVESTIGATION.md](retrieval/RETRIEVAL_MISMATCH_INVESTIGATION.md) |
| LLM / clarification framing | [llm/mechanic_diagnostic_framework.md](llm/mechanic_diagnostic_framework.md) |
| Scrapy, sources, scraping prompt | [scraping/](scraping/) (architecture, discovery, prompt) |
| Training pipeline, scraped data, optimization | [training/](training/) |
| IAP, Cloud Run access, proxy | [deploy/](deploy/) |
| What landed recently on `main` | [changelog/RECENT_PRS_CONTEXT.md](changelog/RECENT_PRS_CONTEXT.md) |
| Repo structure proposals (historical / planning) | [planning/RESTRUCTURING_PROPOSAL.md](planning/RESTRUCTURING_PROPOSAL.md) |

---

## Folder layout

```
docs/context/
├── README.md          ← you are here
├── OVERVIEW.md        ← condensed spine
├── core/              SPEC, ARCHITECTURE, REPO_RULES_FOR_AI
├── data/              DATABASE, ISTA_DATABASE_GUIDE
├── retrieval/         RETRIEVAL_MISMATCH_INVESTIGATION
├── llm/               mechanic_diagnostic_framework
├── scraping/          SCRAPER_*, WEB_SCRAPING_PROMPT
├── training/          TRAINING_PIPELINE_*, DATA_OPTIMIZATION_SUMMARY
├── deploy/            IAP_ACCESS, PROXY_API_SETUP
├── changelog/         RECENT_PRS_CONTEXT
└── planning/          RESTRUCTURING_PROPOSAL
```

---

## Quick links (humans & agents)

| Need | Doc |
|------|-----|
| IDE setup, run, test, lint | [AGENTS.md](../../AGENTS.md) |
| Project overview | [README.md](../../README.md) |
| Matcher / retrieval tests | `tests/test_matcher_accuracy.py`, `tests/test_retrieval_evaluation.py` |

