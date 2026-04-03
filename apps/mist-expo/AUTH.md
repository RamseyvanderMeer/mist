# MIST Expo client — authentication

This app talks to the **FastAPI** service in the repo root. Rules live in `src/auth/dependencies.py`, `src/auth/routes.py`, and `src/auth/google_oauth.py`.

## Recommended: Sign in with Google

1. In [Google Cloud Console](https://console.cloud.google.com/apis/credentials), create OAuth **Web**, **iOS**, and **Android** clients as needed for each platform you ship.
2. Add the **same** client ID strings to the API environment variable **`GOOGLE_OAUTH_CLIENT_IDS`** (comma-separated). Example: `123.apps.googleusercontent.com,456.apps.googleusercontent.com`
3. Copy client IDs into this app’s `.env` (`EXPO_PUBLIC_GOOGLE_*` — see `.env.example`).
4. Install backend deps: `pip install -r requirements.txt` (includes `google-auth`).

**Web:** The client uses Expo’s `useIdTokenAuthRequest` so Google returns an **ID token** (`id_token`). The plain `useAuthRequest` hook defaults to `response_type=token` on web, which yields only an access token and produces “No ID token returned” in this app.

If the browser logs `Cross-Origin-Opener-Policy … would block the window.closed call`, that comes from Expo’s web auth popup polling `popup.closed` under a strict COOP; it is often noisy but harmless once the popup posts the redirect URL back. If sign-in still fails, confirm **Authorized JavaScript origins** include your dev origin (e.g. `http://localhost:8081`).

The app sends:

- `Authorization: Bearer <Google ID token>`
- `X-Goog-Authenticated-User-Email: <same email as in token>` (set automatically from the token payload)

The API verifies the ID token with Google and matches the email header when present.

## Other modes

| Mode | Required on protected routes | Notes |
|------|------------------------------|--------|
| **Google Sign-In** | Valid **Google ID token** in `Authorization: Bearer` | Requires `GOOGLE_OAUTH_CLIENT_IDS` on the server. |
| **IAP** | Valid **IAP JWT** in `X-Goog-Iap-Jwt-Assertion` | If this header is present, it takes precedence over Bearer. |
| **Local dev** (`DEV_MODE=true`) | `X-Goog-Authenticated-User-Email` only | No JWT verification. |
| **Optional** | `X-API-Key` | If `API_KEYS` is set on the server. |

`GET /health` stays unauthenticated. **`/auth/register`** accepts either a verified Google Bearer token or IAP-style email headers (legacy).

## Storage

Tokens and email are stored in **Secure Store** (native) or **sessionStorage** (web). `EXPO_PUBLIC_MIST_API_KEY` is for internal builds only when the API enforces API keys.

## Advanced

The sign-in screen still has an **Advanced** section for pasting an IAP assertion JWT when your deployment requires it.

## Related docs

- `docs/context/deploy/IAP_ACCESS.md` — IAP and `DEV_MODE`.
- OpenAPI: `{API_URL}/docs`.
