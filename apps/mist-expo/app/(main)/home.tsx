import { useRouter } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import { StyleSheet, Text, View } from 'react-native';
import { apiHealth } from '../../src/api/mistApi';
import {
  Body,
  Card,
  GhostButton,
  PrimaryButton,
  Screen,
  Subtitle,
  Title,
} from '../../src/components/ui';
import { useMistAuth } from '../../src/auth/AuthContext';
import { isGuestSession, useSession } from '../../src/auth/SessionContext';
import { colors, font, radius, space } from '../../src/theme/tokens';

export default function HomeScreen() {
  const router = useRouter();
  const { clearCredentials } = useMistAuth();
  const { me, check, refreshSession } = useSession();

  const health = useQuery({
    queryKey: ['health'],
    queryFn: apiHealth,
  });

  const guest = isGuestSession(check);
  const tierLabel = guest ? 'Guest' : me?.tier ?? '—';
  const subtitle = guest
    ? 'Guest mode is active. You can run up to 3 debugging requests per day in this browser.'
    : 'Signed in and ready to search repair guides.';

  return (
    <Screen scroll>
      <View style={styles.hero}>
        <Title>{guest ? 'Welcome, guest' : 'Welcome back'}</Title>
        <Text style={styles.name}>{guest ? 'Public debugging mode' : me?.display_name || me?.email || 'Technician'}</Text>
        <Body muted>{subtitle}</Body>
      </View>

      <Card style={styles.primaryCard}>
        <Subtitle>Start a diagnosis</Subtitle>
        <Body muted>
          Enter fault codes, symptoms, or both. MIST will search repair guidance and ask follow-up questions only if needed.
        </Body>
        <PrimaryButton title="Search guides" onPress={() => router.push('/diagnose')} />
      </Card>

      <Card>
        <Subtitle>Your access</Subtitle>
        <View style={styles.row}>
          <Text style={styles.statLabel}>Mode</Text>
          <Text style={styles.statValue}>{guest ? 'Guest' : 'Signed in'}</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.statLabel}>Limit</Text>
          <Text style={styles.statValue}>{guest ? '3 requests / day' : tierLabel}</Text>
        </View>
        {!guest ? (
          <View style={styles.row}>
            <Text style={styles.statLabel}>Roles</Text>
            <Text style={styles.statValue}>{me?.roles?.join(', ') || '—'}</Text>
          </View>
        ) : null}
      </Card>

      <Card>
        <Subtitle>API status</Subtitle>
        {health.isFetching ? (
          <Body>Checking…</Body>
        ) : health.data ? (
          <Text style={styles.ok}>● {health.data.status}</Text>
        ) : (
          <Text style={styles.bad}>{health.error?.message || 'Unreachable'}</Text>
        )}
        <PrimaryButton title="Refresh status" onPress={() => health.refetch()} />
      </Card>

      <GhostButton
        title={guest ? 'Exit guest mode' : 'Sign out'}
        danger
        onPress={async () => {
          await clearCredentials();
          await refreshSession();
          router.replace('/sign-in');
        }}
      />
      <View style={{ height: space.xl }} />
    </Screen>
  );
}

const styles = StyleSheet.create({
  hero: { marginBottom: space.md, gap: space.xs },
  name: {
    color: colors.text,
    fontSize: font.headline,
    fontWeight: '600',
  },
  primaryCard: {
    borderColor: '#3a465c',
    backgroundColor: '#131b29',
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: space.sm,
    gap: space.md,
  },
  statLabel: { color: colors.textMuted, fontSize: font.caption },
  statValue: { color: colors.text, fontSize: font.body, fontWeight: '600', flex: 1, textAlign: 'right' },
  ok: { color: colors.success, fontSize: font.body, marginBottom: space.sm },
  bad: { color: colors.danger, fontSize: font.body, marginBottom: space.sm },
});