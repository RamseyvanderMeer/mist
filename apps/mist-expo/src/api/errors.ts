export function parseApiError(body: unknown, fallback = 'Request failed'): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const obj = body as { detail: unknown; code?: unknown; limit?: unknown };
    const d = obj.detail;
    if (typeof d === 'string') {
      if (obj.code === 'RATE_LIMIT_EXCEEDED') {
        const limit = typeof obj.limit === 'string' && obj.limit ? obj.limit : null;
        return limit ? `You have hit your rate limit of ${limit}.` : d;
      }
      return d;
    }
    if (Array.isArray(d)) {
      return d
        .map((item: unknown) => {
          if (item && typeof item === 'object' && 'msg' in item) {
            return String((item as { msg: unknown }).msg);
          }
          return String(item);
        })
        .join('; ');
    }
  }
  return fallback;
}

export async function readErrorBody(res: Response): Promise<string> {
  try {
    const j = await res.json();
    return parseApiError(j, `HTTP ${res.status}`);
  } catch {
    return `HTTP ${res.status}`;
  }
}
