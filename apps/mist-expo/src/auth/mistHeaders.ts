/**
 * Header names and builders aligned with FastAPI (`src/auth/dependencies.py`).
 *
 * Production: `get_current_user` verifies `X-Goog-Iap-Jwt-Assertion` (IAP JWT).
 * DEV_MODE=true: only `X-Goog-Authenticated-User-Email` is required (no JWT).
 *
 * Optional: `API_KEYS` on the server adds `X-API-Key` on every path except `/health`.
 */

export const IAP_JWT_HEADER = 'X-Goog-Iap-Jwt-Assertion';
export const IAP_EMAIL_HEADER = 'X-Goog-Authenticated-User-Email';
export const IAP_SUBJECT_HEADER = 'X-Goog-Authenticated-User-Id';
export const API_KEY_HEADER = 'X-API-Key';

export type MistCredentialSnapshot = {
  /** IAP assertion JWT (production behind IAP). */
  iapJwt: string | null;
  /** Google OAuth ID token — sent as Authorization Bearer when backend sets GOOGLE_OAUTH_CLIENT_IDS. */
  googleIdToken: string | null;
  /** Used with DEV_MODE and must match Google / IAP token email in production. */
  iapEmail: string | null;
  iapSubject: string | null;
  /** When server has `API_KEYS` set. */
  apiKey: string | null;
};

export function buildMistAuthHeaders(creds: MistCredentialSnapshot): Record<string, string> {
  const headers: Record<string, string> = {};
  if (creds.iapJwt?.trim()) {
    headers[IAP_JWT_HEADER] = creds.iapJwt.trim();
  }
  if (creds.googleIdToken?.trim()) {
    headers.Authorization = `Bearer ${creds.googleIdToken.trim()}`;
  }
  if (creds.iapEmail?.trim()) {
    headers[IAP_EMAIL_HEADER] = creds.iapEmail.trim().toLowerCase();
  }
  if (creds.iapSubject?.trim()) {
    headers[IAP_SUBJECT_HEADER] = creds.iapSubject.trim();
  }
  if (creds.apiKey?.trim()) {
    headers[API_KEY_HEADER] = creds.apiKey.trim();
  }
  return headers;
}
