/** Decode JWT payload (no signature verification — use only for display / header hints). */
export function decodeJwtPayloadJson(token: string): Record<string, unknown> | null {
  try {
    const parts = token.split('.');
    if (parts.length < 2) return null;
    const segment = parts[1];
    const base64 = segment.replace(/-/g, '+').replace(/_/g, '/');
    const padLen = (4 - (base64.length % 4)) % 4;
    const padded = base64 + '='.repeat(padLen);
    const atobFn = globalThis.atob;
    if (typeof atobFn !== 'function') return null;
    const json = atobFn(padded);
    return JSON.parse(json) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export function emailFromIdToken(idToken: string): string | null {
  const p = decodeJwtPayloadJson(idToken);
  const email = p?.email;
  return typeof email === 'string' && email.includes('@') ? email.toLowerCase() : null;
}
