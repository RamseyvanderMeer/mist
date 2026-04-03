export function parseApiError(body: unknown, fallback = 'Request failed'): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === 'string') return d;
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
