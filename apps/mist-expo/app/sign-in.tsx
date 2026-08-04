import { useRouter } from 'expo-router';
import { useEffect, useMemo, useState } from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { apiAuthRegister, apiStartGuestSession } from '../src/api/mistApi';
import { getMistApiBaseUrl } from '../src/api/mistClient';
import {
  Body,
  Card,
  GhostButton,
  PrimaryButton,
  Screen,
  Subtitle,
  Title,
} from '../src/components/ui';
import { useMistAuth } from '../src/auth/AuthContext';
import { emailFromIdToken } from '../src/auth/jwtPayload';
import { hasSignInCredentials, isGuestSession, isRegisteredSession, useSession } from '../src/auth/SessionContext';
import { useGoogleAuthRequest, isGoogleSignInConfigured } from '../src/auth/useGoogleSignIn';
import { colors, font, radius, space } from '../src/theme/tokens';

export default function SignInScreen() {
  const router = useRouter();
  const {
    setIapJwt,
    setGoogleIdToken,
    setIapEmail,
    setIapSubject,
    setGuestMode,
    clearCredentials,
    getAuthHeaders,
    creds,
  } = useMistAuth();
  const { check, refreshSession, lastError } = useSession();
  const { signIn: googleSignIn, ready: googleReady, configured: googleConfigured } =
    useGoogleAuthRequest();

  const [displayName, setDisplayName] = useState('');
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    if (isRegisteredSession(check) || isGuestSession(check)) {
      router.replace('/home');
    }
  }, [check, router]);

  const subtitle = useMemo(() => {
    return 'Search BMW repair guides with Google sign-in or try guest mode for quick public debugging.';
  }, []);

  const onGoogle = async () => {
    setLocalError(null);
    if (!isGoogleSignInConfigured()) {
      setLocalError('Google sign-in is not configured for this deployment.');
      return;
    }
    setBusy(true);
    try {
      const result = await googleSignIn();
      if (!result.ok) {
        setLocalError(result.error);
        return;
      }
      await setGuestMode(false);
      await setIapJwt(null);
      await setGoogleIdToken(result.idToken);
      const em = emailFromIdToken(result.idToken);
      if (em) await setIapEmail(em);
      await refreshSession();
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : 'Google sign-in failed');
    } finally {
      setBusy(false);
    }
  };

  const onGuest = async () => {
    setLocalError(null);
    setBusy(true);
    try {
      await clearCredentials();
      await apiStartGuestSession();
      await setGuestMode(true);
      await refreshSession();
      router.replace('/home');
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : 'Failed to start guest mode');
    } finally {
      setBusy(false);
    }
  };

  const onRegister = async () => {
    setLocalError(null);
    setBusy(true);
    try {
      await apiAuthRegister(getAuthHeaders, { display_name: displayName.trim() || null });
      await refreshSession();
      router.replace('/home');
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : 'Registration failed');
    } finally {
      setBusy(false);
    }
  };

  const onSignOut = async () => {
    await clearCredentials();
    await refreshSession();
  };

  const needsRegister = check && check.authenticated && !check.registered && 'email' in check;

  return (
    <Screen scroll>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.flex}>
        <View style={styles.hero}>
          <View style={styles.badge}><Text style={styles.badgeText}>MIST</Text></View>
          <Title>Fix the issue faster</Title>
          <Text style={styles.tagline}>{subtitle}</Text>
          <Text style={styles.endpoint}>API · {getMistApiBaseUrl()}</Text>
        </View>

        <Card style={styles.primaryCard}>
          <Subtitle>Get started</Subtitle>
          <Body muted>Choose the fastest way in. Guest mode gives you 3 debugging requests per day in this browser.</Body>
          <View style={styles.ctaStack}>
            <PrimaryButton
              title="Continue with Google"
              onPress={onGoogle}
              loading={busy}
              disabled={busy || !googleConfigured || !googleReady}
            />
            <Pressable onPress={onGuest} disabled={busy} style={({ pressed }) => [styles.guestBtn, pressed && { opacity: 0.9 }]}>
              <Text style={styles.guestBtnTitle}>Continue as guest</Text>
              <Text style={styles.guestBtnBody}>3 debugging requests per day · no account needed</Text>
            </Pressable>
          </View>
          {!googleConfigured ? (
            <Body muted>Google sign-in is unavailable on this deployment, but guest mode still works.</Body>
          ) : null}
        </Card>

        <Card>
          <Subtitle>How it works</Subtitle>
          <View style={styles.featureList}>
            <View style={styles.featureItem}><Text style={styles.featureDot}>1</Text><Text style={styles.featureText}>Paste fault codes or describe the symptom</Text></View>
            <View style={styles.featureItem}><Text style={styles.featureDot}>2</Text><Text style={styles.featureText}>Answer clarifying questions if needed</Text></View>
            <View style={styles.featureItem}><Text style={styles.featureDot}>3</Text><Text style={styles.featureText}>Review ranked repair guides and next steps</Text></View>
          </View>
        </Card>

        {needsRegister ? (
          <Card>
            <Subtitle>Finish setup</Subtitle>
            <Body muted>You are signed in as {(check as { email: string }).email} but do not have a MIST account yet.</Body>
            <Pressable onPress={onRegister} disabled={busy} style={({ pressed }) => [styles.registerBtn, pressed && { opacity: 0.9 }]}>
              <Text style={styles.registerBtnText}>Create my MIST account</Text>
            </Pressable>
          </Card>
        ) : null}

        {(hasSignInCredentials(creds) || creds.guestMode) ? (
          <GhostButton title="Clear session" onPress={onSignOut} danger />
        ) : null}

        {(localError || lastError) && (
          <Card style={{ borderColor: colors.danger }}>
            <Body>{localError || lastError}</Body>
          </Card>
        )}

        <View style={{ height: space.lg }} />
      </KeyboardAvoidingView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  hero: { marginBottom: space.lg, marginTop: space.sm, gap: space.sm },
  badge: {
    alignSelf: 'flex-start',
    backgroundColor: colors.bgElevated,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  badgeText: { color: colors.accent, fontWeight: '800', letterSpacing: 2 },
  tagline: { color: colors.textMuted, fontSize: font.body, lineHeight: 22 },
  endpoint: { color: colors.textMuted, fontSize: font.caption },
  primaryCard: {
    borderColor: '#3a465c',
    backgroundColor: '#131b29',
  },
  ctaStack: { gap: space.sm, marginTop: space.sm, marginBottom: space.sm },
  guestBtn: {
    backgroundColor: colors.bgElevated,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    paddingVertical: 14,
    paddingHorizontal: 16,
  },
  guestBtnTitle: { color: colors.text, fontSize: font.headline, fontWeight: '700' },
  guestBtnBody: { color: colors.textMuted, fontSize: font.caption, marginTop: 4 },
  featureList: { gap: space.sm },
  featureItem: { flexDirection: 'row', gap: space.sm, alignItems: 'flex-start' },
  featureDot: {
    color: '#0a0c10',
    backgroundColor: colors.accent,
    width: 22,
    height: 22,
    textAlign: 'center',
    lineHeight: 22,
    borderRadius: 11,
    fontWeight: '800',
    overflow: 'hidden',
  },
  featureText: { color: colors.text, fontSize: font.body, flex: 1, lineHeight: 22 },
  registerBtn: {
    marginTop: space.sm,
    backgroundColor: colors.accent,
    borderRadius: radius.md,
    paddingVertical: 14,
    alignItems: 'center',
  },
  registerBtnText: { color: '#0a0c10', fontWeight: '800', fontSize: font.body },
});