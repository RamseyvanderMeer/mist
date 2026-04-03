import { mistFetch } from './mistClient';
import { readErrorBody } from './errors';
import type {
  AuthCheckResponse,
  HealthResponse,
  QueryResponse,
  UserMe,
} from './types';

export type AuthHeaderGetter = () => Promise<Record<string, string>>;

export async function apiHealth(): Promise<HealthResponse> {
  const res = await mistFetch('/health', { withAuth: false });
  if (!res.ok) throw new Error(await readErrorBody(res));
  return res.json() as Promise<HealthResponse>;
}

export async function apiAuthCheck(getAuthHeaders: AuthHeaderGetter): Promise<AuthCheckResponse> {
  const res = await mistFetch('/auth/check', { getAuthHeaders });
  const data = (await res.json()) as AuthCheckResponse;
  if (!res.ok) throw new Error(await readErrorBody(res));
  return data;
}

export async function apiAuthMe(getAuthHeaders: AuthHeaderGetter): Promise<UserMe> {
  const res = await mistFetch('/auth/me', { getAuthHeaders });
  if (!res.ok) throw new Error(await readErrorBody(res));
  return res.json() as Promise<UserMe>;
}

export async function apiAuthRegister(
  getAuthHeaders: AuthHeaderGetter,
  body: { display_name?: string | null },
): Promise<UserMe> {
  const res = await mistFetch('/auth/register', {
    method: 'POST',
    getAuthHeaders,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await readErrorBody(res));
  return res.json() as Promise<UserMe>;
}

export async function apiQuery(
  getAuthHeaders: AuthHeaderGetter,
  body: {
    fault_codes: string[];
    obd_data: Record<string, unknown>;
    description?: string | null;
    vehicle_context?: Record<string, unknown> | null;
    session_id?: string | null;
  },
): Promise<QueryResponse> {
  const res = await mistFetch('/query', {
    method: 'POST',
    getAuthHeaders,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await readErrorBody(res));
  return res.json() as Promise<QueryResponse>;
}

export async function apiClarify(
  getAuthHeaders: AuthHeaderGetter,
  body: { session_id: string; responses: string[] },
): Promise<QueryResponse> {
  const res = await mistFetch('/clarify', {
    method: 'POST',
    getAuthHeaders,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await readErrorBody(res));
  return res.json() as Promise<QueryResponse>;
}

export async function apiFeedbackRating(
  getAuthHeaders: AuthHeaderGetter,
  body: { session_id: string; rating: number; selected_guide?: string | null },
): Promise<void> {
  const res = await mistFetch('/feedback/rating', {
    method: 'POST',
    getAuthHeaders,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await readErrorBody(res));
}

export async function apiFeedbackOutcome(
  getAuthHeaders: AuthHeaderGetter,
  body: { session_id: string; outcome: string; details?: Record<string, unknown> | null },
): Promise<void> {
  const res = await mistFetch('/feedback/outcome', {
    method: 'POST',
    getAuthHeaders,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await readErrorBody(res));
}
