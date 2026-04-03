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
import { useSession } from '../../src/auth/SessionContext';
import { colors, font, space } from '../../src/theme/tokens';

export default function HomeScreen() {
  const router = useRouter();
  const { clearCredentials, getAuthHeaders } = useMistAuth();
  const { me, refreshSession } = useSession();

  const health = useQuery({
    queryKey: ['health'],
    queryFn: apiHealth,
  });

  const tierLabel = me?.tier ?? '—';
  const blocked = tierLabel.toLowerCase() === 'blocked';

  return (
    <Screen scroll>
      <View style={styles.top}>
        <Title>Welcome back</Title>
        <Text style={styles.name}>{me?.display_name || me?.email || 'Technician'}</Text>
        <Body muted>{me?.email}</Body>
      </View>

      <Card>
        <Subtitle>Your access</Subtitle>
        <View style={styles.row}>
          <Text style={styles.statLabel}>Plan tier</Text>
          <Text style={[styles.statValue, blocked && { color: colors.warning }]}>{tierLabel}</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.statLabel}>Roles</Text>
          <Text style={styles.statValue}>{me?.roles?.join(', ') || '—'}</Text>
        </View>
        {blocked ? (
          <Body muted>
            This tier cannot call /query yet. Ask an admin to upgrade your account in Postgres.
          </Body>
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

      <Card>
        <Subtitle>Workflow</Subtitle>
        <Body muted>Run a fault-code or symptom search, answer clarifying questions if needed, then review guides and send feedback.</Body>
        <PrimaryButton title="Start diagnosis" onPress={() => router.push('/diagnose')} />
      </Card>

      <GhostButton
        title="Sign out"
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
  top: { marginBottom: space.md },
  name: {
    color: colors.text,
    fontSize: font.headline,
    fontWeight: '600',
    marginTop: space.xs,
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
