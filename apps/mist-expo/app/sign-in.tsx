import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { apiAuthRegister } from '../src/api/mistApi';
import { getMistApiBaseUrl } from '../src/api/mistClient';
import {
  Body,
  Card,
  Field,
  GhostButton,
  PrimaryButton,
  Screen,
  Subtitle,
  Title,
} from '../src/components/ui';
import { useMistAuth } from '../src/auth/AuthContext';
import { emailFromIdToken } from '../src/auth/jwtPayload';
import { hasSignInCredentials, isRegisteredSession, useSession } from '../src/auth/SessionContext';
import { useGoogleAuthRequest, isGoogleSignInConfigured } from '../src/auth/useGoogleSignIn';
import { colors, font, space } from '../src/theme/tokens';

export default function SignInScreen() {
  const router = useRouter();
  const {
    setIapJwt,
    setGoogleIdToken,
    setIapEmail,
    setIapSubject,
    clearCredentials,
    getAuthHeaders,
    creds,
  } = useMistAuth();
  const { check, refreshSession, lastError } = useSession();
  const { signIn: googleSignIn, ready: googleReady, configured: googleConfigured } =
    useGoogleAuthRequest();

  const [email, setEmail] = useState(creds.iapEmail || '');
  const [jwt, setJwt] = useState('');
  const [subject, setSubject] = useState(creds.iapSubject || '');
  const [displayName, setDisplayName] = useState('');
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    if (!hasSignInCredentials(creds)) return;
    if (isRegisteredSession(check)) {
      router.replace('/home');
    }
  }, [creds, check, router]);

  const onGoogle = async () => {
    setLocalError(null);
    if (!isGoogleSignInConfigured()) {
      setLocalError('Add Google OAuth client IDs to .env (see .env.example).');
      return;
    }
    setBusy(true);
    try {
      const result = await googleSignIn();
      if (!result.ok) {
        setLocalError(result.error);
        return;
      }
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

  const onContinue = async () => {
    setLocalError(null);
    if (!email.trim() && !jwt.trim()) {
      setLocalError('Use Sign in with Google, or enter email (dev) / IAP JWT (advanced).');
      return;
    }
    setBusy(true);
    try {
      await setGoogleIdToken(null);
      await setIapJwt(jwt.trim() || null);
      await setIapEmail(email.trim() || null);
      await setIapSubject(subject.trim() || null);
      await refreshSession();
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : 'Sign-in failed');
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
    setEmail('');
    setJwt('');
    setSubject('');
  };

  const needsRegister =
    check && check.authenticated && !check.registered && 'email' in check;

  return (
    <Screen scroll>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.flex}
      >
        <View style={styles.header}>
          <Text style={styles.logo}>MIST</Text>
          <Text style={styles.tagline}>BMW diagnostic guide search</Text>
        </View>

        <Card>
          <Title>Sign in</Title>
          <Body muted>
            API: {getMistApiBaseUrl()}
            {'\n'}
            Use your Google work account when the API has{' '}
            <Text style={styles.mono}>GOOGLE_OAUTH_CLIENT_IDS</Text> set. Dev servers can use email
            only with <Text style={styles.mono}>DEV_MODE=true</Text>.
          </Body>
        </Card>

        {googleConfigured ? (
          <Card>
            <Subtitle>Recommended</Subtitle>
            <PrimaryButton
              title="Continue with Google"
              onPress={onGoogle}
              loading={busy}
              disabled={busy || !googleReady}
            />
            {!googleReady ? (
              <Body muted>Preparing Google sign-in…</Body>
            ) : (
              <Body muted>Opens the Google account picker; no JWT paste required.</Body>
            )}
          </Card>
        ) : (
          <Card style={{ borderColor: colors.warning }}>
            <Subtitle>Google sign-in disabled</Subtitle>
            <Body muted>
              Set EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID (and iOS/Android client IDs for native) in{' '}
              <Text style={styles.mono}>.env</Text>, and add the same client ID(s) to the API env{' '}
              <Text style={styles.mono}>GOOGLE_OAUTH_CLIENT_IDS</Text>.
            </Body>
          </Card>
        )}

        <Card>
          <Subtitle>Manual sign-in</Subtitle>
          <Body muted>For local dev (email) or IAP JWT from your ops team.</Body>
          <Field
            label="Work email"
            placeholder="you@shop.com"
            autoCapitalize="none"
            keyboardType="email-address"
            value={email}
            onChangeText={setEmail}
          />
          <PrimaryButton title="Continue with email / dev" onPress={onContinue} loading={busy} disabled={busy} />
        </Card>

        <Pressable onPress={() => setShowAdvanced((s) => !s)} style={styles.advancedToggle}>
          <Text style={styles.advancedToggleText}>{showAdvanced ? '▼' : '▶'} Advanced (IAP JWT)</Text>
        </Pressable>

        {showAdvanced ? (
          <Card>
            <Field
              label="IAP assertion JWT"
              placeholder="Paste only if required by your deployment"
              autoCapitalize="none"
              value={jwt}
              onChangeText={setJwt}
              multiline
              style={styles.multiline}
            />
            <Field
              label="Subject ID (optional)"
              placeholder="accounts.google.com:…"
              autoCapitalize="none"
              value={subject}
              onChangeText={setSubject}
            />
            <PrimaryButton title="Save advanced credentials" onPress={onContinue} loading={busy} disabled={busy} />
          </Card>
        ) : null}

        {hasSignInCredentials(creds) ? (
          <GhostButton title="Clear saved session" onPress={onSignOut} danger />
        ) : null}

        {needsRegister ? (
          <Card>
            <Subtitle>Complete registration</Subtitle>
            <Body muted>
              You are signed in as {(check as { email: string }).email} but have no MIST account yet.
            </Body>
            <Field
              label="Display name (optional)"
              placeholder="Service bay name"
              value={displayName}
              onChangeText={setDisplayName}
            />
            <PrimaryButton title="Register account" onPress={onRegister} loading={busy} disabled={busy} />
          </Card>
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
  header: { marginBottom: space.lg, marginTop: space.sm },
  logo: {
    color: colors.accent,
    fontSize: 36,
    fontWeight: '800',
    letterSpacing: 4,
  },
  tagline: {
    color: colors.textMuted,
    fontSize: font.body,
    marginTop: space.xs,
  },
  mono: { fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }) },
  multiline: { minHeight: 72, textAlignVertical: 'top' },
  advancedToggle: { paddingVertical: space.sm, marginBottom: space.xs },
  advancedToggleText: { color: colors.textMuted, fontSize: font.caption },
});
