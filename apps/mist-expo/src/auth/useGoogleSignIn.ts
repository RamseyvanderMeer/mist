import * as Google from 'expo-auth-session/providers/google';
import * as WebBrowser from 'expo-web-browser';
import { useCallback, useMemo } from 'react';
import { Platform } from 'react-native';

WebBrowser.maybeCompleteAuthSession();

const webClientId = process.env.EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID ?? '';
const iosClientId = process.env.EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID ?? '';
const androidClientId = process.env.EXPO_PUBLIC_GOOGLE_ANDROID_CLIENT_ID ?? '';

export function isGoogleSignInConfigured(): boolean {
  if (Platform.OS === 'web') return Boolean(webClientId);
  if (Platform.OS === 'ios') return Boolean(iosClientId || webClientId);
  if (Platform.OS === 'android') return Boolean(androidClientId || webClientId);
  return Boolean(webClientId || iosClientId || androidClientId);
}

/**
 * Google OAuth for Sign in with Google. Backend must set GOOGLE_OAUTH_CLIENT_IDS to include
 * the same OAuth client ID(s) that mint the ID token (per platform).
 */
export function useGoogleAuthRequest() {
  // Web: default useAuthRequest uses response_type=token (access_token only, no id_token).
  // useIdTokenAuthRequest sets response_type=id_token on web so the backend can verify JWTs.
  const [request, response, promptAsync] = Google.useIdTokenAuthRequest({
    webClientId: webClientId || undefined,
    iosClientId: iosClientId || undefined,
    androidClientId: androidClientId || undefined,
    scopes: ['openid', 'profile', 'email'],
  });

  const ready = useMemo(() => Boolean(request), [request]);

  const signIn = useCallback(async () => {
    if (!request) return { ok: false as const, error: 'Google OAuth not ready' };
    const result = await promptAsync();
    if (result.type !== 'success') {
      return {
        ok: false as const,
        error: result.type === 'dismiss' ? 'Sign-in cancelled' : 'Google sign-in failed',
      };
    }
    const auth = result.authentication;
    const params = result.params as { id_token?: string; access_token?: string };
    const idToken = auth?.idToken ?? params.id_token;
    if (!idToken) {
      return { ok: false as const, error: 'No ID token returned — check Google OAuth client config.' };
    }
    return { ok: true as const, idToken };
  }, [promptAsync, request]);

  return { request, response, signIn, ready, configured: isGoogleSignInConfigured() };
}
