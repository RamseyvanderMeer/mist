import { Redirect } from 'expo-router';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
import { useMistAuth } from '../src/auth/AuthContext';
import {
  hasSignInCredentials,
  isRegisteredSession,
  useSession,
} from '../src/auth/SessionContext';
import { colors } from '../src/theme/tokens';

export default function Index() {
  const { ready: authReady, creds } = useMistAuth();
  const { check, status } = useSession();

  if (!authReady) {
    return (
      <View style={styles.splash}>
        <ActivityIndicator size="large" color={colors.accent} />
      </View>
    );
  }

  if (!hasSignInCredentials(creds)) {
    return <Redirect href="/sign-in" />;
  }

  if (status === 'idle' || status === 'loading') {
    return (
      <View style={styles.splash}>
        <ActivityIndicator size="large" color={colors.accent} />
      </View>
    );
  }

  if (isRegisteredSession(check)) {
    return <Redirect href="/home" />;
  }

  return <Redirect href="/sign-in" />;
}

const styles = StyleSheet.create({
  splash: {
    flex: 1,
    backgroundColor: colors.bg,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
