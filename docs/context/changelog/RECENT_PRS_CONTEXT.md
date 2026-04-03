# Recent integrated changes (PR / main index)

Rolling context for agents: what landed most recently on **`main`**. The GitHub repo currently shows **one** merged pull request ([PR #1](https://github.com/RamseyvanderMeer/mist/pull/1) — development environment setup, 2026-02-25). Most work is integrated as **direct commits** on `main`; this file indexes the **latest seven** of those by commit order (newest first).

Refresh this doc when you want the next snapshot: `git log -7 --format='%h %ad %s' --date=short`.

---

| # | Commit | Date | Summary | Where to look |
|---|--------|------|---------|----------------|
| 1 | `165ad75` | 2026-04-02 | **IAP access guide** — browser, programmatic (gcloud ID token), and service-account patterns for the Cloud Run + IAP URL. | [IAP_ACCESS.md](../deploy/IAP_ACCESS.md) |
| 2 | `6e22485` | 2026-04-02 | **`DEV_MODE`** — when `DEV_MODE=true`, skip IAP JWT verification for local testing (do not use in production behind IAP). | `src/auth/dependencies.py` (`DEV_MODE`, `verify_iap_jwt`) |
| 3 | `e1c399d` | 2026-04-01 | **JWT debug logging** — extra visibility when diagnosing missing/invalid IAP JWT headers. | `src/auth/dependencies.py` |
| 4 | `86321fd` | 2026-04-01 | **IAP JWT verification** — validates `X-Goog-Iap-Jwt-Assertion` (JWKS from Google) so IAP user headers cannot be spoofed without a real IAP-issued token. | `src/auth/dependencies.py`, `get_current_user` path |
| 5 | `00812bd` | 2026-04-01 | **Deploy workflow** — removed PR-trigger path; documents feature-branch workflow for deployments. | `.github/workflows/` (deploy workflow) |
| 6 | `d32fa00` | 2026-04-01 | **Deploy image tag** — use **latest** image tag for deployment to avoid commit-SHA mismatch issues. | `.github/workflows/` |
| 7 | `bc59412` | 2026-04-01 | **Workload Identity** — CI change testing deployment with corrected IAM permissions for WIF. | `.github/workflows/`, GCP IAM / service account setup |

---

## Related docs

- [ARCHITECTURE.md](../core/ARCHITECTURE.md) — auth, API, deployment overview.
- [SPEC.md](../core/SPEC.md) — IAP headers, rate limits, env vars.
- [PROXY_API_SETUP.md](../deploy/PROXY_API_SETUP.md) — proxy / API access patterns if applicable to your environment.
