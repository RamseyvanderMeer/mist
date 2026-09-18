# MIST Minimal-Cost Production Profile

Goal: keep MIST as close as possible to GCP free-tier + Chroma free-tier limits.

## Current biggest cost drivers
- Large Cloud Run image due to training/scraping/browser dependencies in the API container
- 2 GiB Cloud Run memory allocation
- Large Chroma collection (`repair_guides_qwen3`) with 4096-dim vectors and ~337k indexed docs
- Retrieval fanout that is too high for a public low-cost product
- Main-branch deploy churn triggering Cloud Build

## Immediate changes applied
- Added `requirements-prod.txt` for API runtime only
- Dockerfile now installs production-only dependencies
- Reduced retrieval defaults:
  - `initial_k: 20`
  - `rerank_k: 10`
  - `final_k: 5`
  - `max_questions: 2`
  - `max_clarifications_per_session: 2`

## Recommended Cloud Run profile
Use this for the public app unless profiling proves otherwise:
- memory: `512Mi` or `1Gi`
- cpu: `1`
- min instances: `0`
- max instances: `1` to start
- timeout: `120` if safe for query path
- concurrency: `10-20` instead of `80` if latency spikes under load

## Recommended product policy
- Guest mode: 3 requests/day
- Signed-in free mode: 1 request/day
- Keep public traffic on the cheapest retrieval path
- Avoid expensive clarification loops for anonymous usage

## Biggest Chroma cost lever
The current Chroma footprint is likely too large for sustained free-tier use.

### Best next move
Create a smaller public collection:
- index fewer documents
- reduce chunk count per procedure
- prefer procedure-level entries or coarse chunks
- keep only highest-value / highest-frequency repair guides

### Target
Aim for a public collection in the ~25k-75k item range instead of ~337k.

## Better architecture split
Separate these concerns:
1. **Production API runtime**
   - FastAPI
   - auth
   - Redis
   - Postgres
   - Chroma client
   - chosen hosted LLM client(s)
2. **Offline jobs / dev workflows**
   - indexing
   - training
   - scraper
   - browser automation
   - local experiments

## Things to avoid
- Shipping Playwright in the production API image
- Shipping training dependencies in the production API image
- Frequent pushes to `main` for minor config iteration
- Full-size vector collection for public anonymous traffic

## Strong recommendation
If staying on Chroma free-tier is mandatory, reduce the public index size significantly.
If staying on GCP free-tier is mandatory, keep the API image lean and reduce Cloud Run memory.

## Next highest-ROI follow-up changes
1. Lower Cloud Run memory in Terraform
2. Add a separate smaller public Chroma collection
3. Consider switching public guest search to a smaller vector dimension / smaller embedding model
4. Keep full corpus retrieval for internal/admin or batch/offline workflows only
