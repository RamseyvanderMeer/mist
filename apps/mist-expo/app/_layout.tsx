import 'react-native-gesture-handler';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Stack } from 'expo-router';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { TamaguiProvider, Theme } from 'tamagui';
import { AuthProvider } from '../src/auth/AuthContext';
import { SessionProvider } from '../src/auth/SessionContext';
import { colors } from '../src/theme/tokens';
import tamaguiConfig from '../tamagui.config';

const queryClient = new QueryClient();

export default function RootLayout() {
  return (
    <GestureHandlerRootView style={{ flex: 1, backgroundColor: colors.bg }}>
      <TamaguiProvider config={tamaguiConfig} defaultTheme="dark">
        <Theme name="dark">
          <QueryClientProvider client={queryClient}>
            <AuthProvider>
              <SessionProvider>
                <Stack
                  screenOptions={{
                    headerShown: false,
                    contentStyle: { backgroundColor: colors.bg },
                  }}
                >
                  <Stack.Screen name="index" />
                  <Stack.Screen name="sign-in" />
                  <Stack.Screen name="(main)" />
                </Stack>
              </SessionProvider>
            </AuthProvider>
          </QueryClientProvider>
        </Theme>
      </TamaguiProvider>
    </GestureHandlerRootView>
  );
}
