export function getMistApiBaseUrl(): string {
  const raw = process.env.EXPO_PUBLIC_MIST_API_URL || 'http://127.0.0.1:8000';
  return raw.replace(/\/$/, '');
}

export type MistFetchOptions = RequestInit & {
  getAuthHeaders?: () => Promise<Record<string, string>>;
  /** When false, skip auth headers (e.g. GET /health). */
  withAuth?: boolean;
};

export async function mistFetch(path: string, options: MistFetchOptions = {}): Promise<Response> {
  const base = getMistApiBaseUrl();
  const { getAuthHeaders, withAuth = true, headers: initHeaders, ...init } = options;

  const authHeaders =
    withAuth && getAuthHeaders ? await getAuthHeaders() : ({} as Record<string, string>);

  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...authHeaders,
    ...(initHeaders as Record<string, string>),
  };

  const body = init.body;
  if (body && typeof body === 'string' && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  return fetch(`${base}${path.startsWith('/') ? path : `/${path}`}`, {
    ...init,
    headers,
  });
}
