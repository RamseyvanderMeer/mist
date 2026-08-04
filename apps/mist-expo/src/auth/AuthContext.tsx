import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import { buildMistAuthHeaders, MistCredentialSnapshot } from './mistHeaders';
import { getSecurePref, setSecurePref, STORAGE_KEYS } from './securePrefs';

type AuthContextValue = {
  /** Values persisted on device (tokens are sensitive). */
  creds: MistCredentialSnapshot;
  ready: boolean;
  setIapJwt: (v: string | null) => Promise<void>;
  setGoogleIdToken: (v: string | null) => Promise<void>;
  setIapEmail: (v: string | null) => Promise<void>;
  setIapSubject: (v: string | null) => Promise<void>;
  setGuestMode: (v: boolean) => Promise<void>;
  clearCredentials: () => Promise<void>;
  getAuthHeaders: () => Promise<Record<string, string>>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

function readEnvApiKey(): string | null {
  return process.env.EXPO_PUBLIC_MIST_API_KEY?.trim() || null;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [iapJwt, setIapJwtState] = useState<string | null>(null);
  const [googleIdToken, setGoogleIdTokenState] = useState<string | null>(null);
  const [iapEmail, setIapEmailState] = useState<string | null>(null);
  const [iapSubject, setIapSubjectState] = useState<string | null>(null);
  const [guestMode, setGuestModeState] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [jwt, google, email, subject, guest] = await Promise.all([
        getSecurePref(STORAGE_KEYS.iapJwt),
        getSecurePref(STORAGE_KEYS.googleIdToken),
        getSecurePref(STORAGE_KEYS.iapEmail),
        getSecurePref(STORAGE_KEYS.iapSubject),
        getSecurePref(STORAGE_KEYS.guestMode),
      ]);
      if (!cancelled) {
        setIapJwtState(jwt);
        setGoogleIdTokenState(google);
        setIapEmailState(email);
        setIapSubjectState(subject);
        setGuestModeState(guest === 'true');
        setReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const setIapJwt = useCallback(async (v: string | null) => {
    setIapJwtState(v);
    await setSecurePref(STORAGE_KEYS.iapJwt, v);
  }, []);

  const setGoogleIdToken = useCallback(async (v: string | null) => {
    setGoogleIdTokenState(v);
    await setSecurePref(STORAGE_KEYS.googleIdToken, v);
  }, []);

  const setIapEmail = useCallback(async (v: string | null) => {
    const normalized = v?.trim().toLowerCase() || null;
    setIapEmailState(normalized);
    await setSecurePref(STORAGE_KEYS.iapEmail, normalized);
  }, []);

  const setIapSubject = useCallback(async (v: string | null) => {
    const trimmed = v?.trim() || null;
    setIapSubjectState(trimmed);
    await setSecurePref(STORAGE_KEYS.iapSubject, trimmed);
  }, []);

  const setGuestMode = useCallback(async (v: boolean) => {
    setGuestModeState(v);
    await setSecurePref(STORAGE_KEYS.guestMode, v ? 'true' : null);
  }, []);

  const clearCredentials = useCallback(async () => {
    await Promise.all([
      setSecurePref(STORAGE_KEYS.iapJwt, null),
      setSecurePref(STORAGE_KEYS.googleIdToken, null),
      setSecurePref(STORAGE_KEYS.iapEmail, null),
      setSecurePref(STORAGE_KEYS.iapSubject, null),
      setSecurePref(STORAGE_KEYS.guestMode, null),
    ]);
    setIapJwtState(null);
    setGoogleIdTokenState(null);
    setIapEmailState(null);
    setIapSubjectState(null);
    setGuestModeState(false);
  }, []);

  const creds: MistCredentialSnapshot = useMemo(
    () => ({
      iapJwt,
      googleIdToken,
      iapEmail,
      iapSubject,
      guestMode,
      apiKey: readEnvApiKey(),
    }),
    [iapJwt, googleIdToken, iapEmail, iapSubject, guestMode],
  );

  const getAuthHeaders = useCallback(async () => buildMistAuthHeaders(creds), [creds]);

  const value = useMemo(
    () => ({
      creds,
      ready,
      setIapJwt,
      setGoogleIdToken,
      setIapEmail,
      setIapSubject,
      setGuestMode,
      clearCredentials,
      getAuthHeaders,
    }),
    [
      creds,
      ready,
      setIapJwt,
      setGoogleIdToken,
      setIapEmail,
      setIapSubject,
      setGuestMode,
      clearCredentials,
      getAuthHeaders,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useMistAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useMistAuth must be used within AuthProvider');
  }
  return ctx;
}
