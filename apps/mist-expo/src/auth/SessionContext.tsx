import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import type { AuthCheckResponse } from '../api/types';
import { apiAuthCheck, apiAuthMe } from '../api/mistApi';
import type { MistCredentialSnapshot } from './mistHeaders';
import { useMistAuth } from './AuthContext';

export function hasSignInCredentials(creds: MistCredentialSnapshot): boolean {
  return Boolean(
    creds.iapEmail?.trim() || creds.iapJwt?.trim() || creds.googleIdToken?.trim(),
  );
}

type SessionValue = {
  check: AuthCheckResponse | null;
  me: import('../api/types').UserMe | null;
  status: 'idle' | 'loading' | 'ready';
  lastError: string | null;
  refreshSession: () => Promise<void>;
};

const SessionContext = createContext<SessionValue | null>(null);

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const { ready, creds, getAuthHeaders } = useMistAuth();
  const [check, setCheck] = useState<AuthCheckResponse | null>(null);
  const [me, setMe] = useState<import('../api/types').UserMe | null>(null);
  const [status, setStatus] = useState<SessionValue['status']>('idle');
  const [lastError, setLastError] = useState<string | null>(null);

  const refreshSession = useCallback(async () => {
    if (!hasSignInCredentials(creds)) {
      setCheck(null);
      setMe(null);
      setLastError(null);
      setStatus('ready');
      return;
    }
    setStatus('loading');
    setLastError(null);
    try {
      const c = await apiAuthCheck(getAuthHeaders);
      setCheck(c);
      if (c.authenticated && c.registered) {
        try {
          const profile = await apiAuthMe(getAuthHeaders);
          setMe(profile);
        } catch (e) {
          setMe(null);
          setLastError(e instanceof Error ? e.message : 'Failed to load profile');
        }
      } else {
        setMe(null);
      }
    } catch (e) {
      setCheck(null);
      setMe(null);
      setLastError(e instanceof Error ? e.message : 'Session refresh failed');
    } finally {
      setStatus('ready');
    }
  }, [creds, getAuthHeaders]);

  useEffect(() => {
    if (!ready) return;
    refreshSession();
  }, [ready, creds.iapEmail, creds.iapJwt, creds.googleIdToken, creds.iapSubject, refreshSession]);

  const value = useMemo(
    () => ({
      check,
      me,
      status,
      lastError,
      refreshSession,
    }),
    [check, me, status, lastError, refreshSession],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionValue {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error('useSession must be used within SessionProvider');
  return ctx;
}

export function isRegisteredSession(check: AuthCheckResponse | null): boolean {
  return Boolean(check && check.authenticated && check.registered);
}
