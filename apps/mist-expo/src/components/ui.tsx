import type { ReactNode } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  type TextInputProps,
  type ViewStyle,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { colors, font, radius, space } from '../theme/tokens';

export function Screen({
  children,
  scroll = false,
  bottomInset = true,
}: {
  children: ReactNode;
  scroll?: boolean;
  bottomInset?: boolean;
}) {
  const content = scroll ? (
    <ScrollView
      contentContainerStyle={[styles.scrollInner, bottomInset && { paddingBottom: space.xl * 2 }]}
      keyboardShouldPersistTaps="handled"
      showsVerticalScrollIndicator={false}
    >
      {children}
    </ScrollView>
  ) : (
    <View style={[styles.fill, bottomInset && { paddingBottom: space.lg }]}>{children}</View>
  );
  return (
    <SafeAreaView style={styles.screen} edges={['top', 'left', 'right']}>
      {content}
    </SafeAreaView>
  );
}

export function Card({ children, style }: { children: ReactNode; style?: ViewStyle }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

export function Title({ children }: { children: ReactNode }) {
  return <Text style={styles.title}>{children}</Text>;
}

export function Subtitle({ children }: { children: ReactNode }) {
  return <Text style={styles.subtitle}>{children}</Text>;
}

export function Body({ children, muted }: { children: ReactNode; muted?: boolean }) {
  return <Text style={[styles.body, muted && styles.bodyMuted]}>{children}</Text>;
}

export function Field({
  label,
  style,
  ...props
}: TextInputProps & { label: string }) {
  return (
    <View style={styles.field}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        placeholderTextColor={colors.textMuted}
        style={[styles.input, style]}
        {...props}
      />
    </View>
  );
}

export function PrimaryButton({
  title,
  onPress,
  disabled,
  loading,
}: {
  title: string;
  onPress: () => void;
  disabled?: boolean;
  loading?: boolean;
}) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled || loading}
      style={({ pressed }) => [
        styles.primaryBtn,
        (disabled || loading) && styles.primaryBtnDisabled,
        pressed && styles.primaryBtnPressed,
      ]}
    >
      {loading ? (
        <ActivityIndicator color="#0a0c10" />
      ) : (
        <Text style={styles.primaryBtnText}>{title}</Text>
      )}
    </Pressable>
  );
}

export function GhostButton({
  title,
  onPress,
  danger,
}: {
  title: string;
  onPress: () => void;
  danger?: boolean;
}) {
  return (
    <Pressable onPress={onPress} style={styles.ghostBtn}>
      <Text style={[styles.ghostBtnText, danger && { color: colors.danger }]}>{title}</Text>
    </Pressable>
  );
}

export function StepPill({ n, total }: { n: number; total: number }) {
  return (
    <View style={styles.stepRow}>
      {Array.from({ length: total }, (_, i) => (
        <View
          key={i}
          style={[styles.stepDot, i < n && styles.stepDotActive, i === n - 1 && styles.stepDotCurrent]}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  fill: { flex: 1, paddingHorizontal: space.md },
  scrollInner: {
    paddingHorizontal: space.md,
    paddingTop: space.sm,
  },
  card: {
    backgroundColor: colors.bgCard,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: space.md,
    marginBottom: space.md,
  },
  title: {
    color: colors.text,
    fontSize: font.title,
    fontWeight: '700',
    letterSpacing: -0.3,
    marginBottom: space.xs,
  },
  subtitle: {
    color: colors.accent,
    fontSize: font.headline,
    fontWeight: '600',
    marginBottom: space.sm,
  },
  body: {
    color: colors.text,
    fontSize: font.body,
    lineHeight: 22,
  },
  bodyMuted: {
    color: colors.textMuted,
    fontSize: font.caption,
    lineHeight: 20,
  },
  field: { marginBottom: space.md },
  label: {
    color: colors.textMuted,
    fontSize: font.caption,
    marginBottom: space.xs,
    fontWeight: '500',
  },
  input: {
    backgroundColor: colors.bgElevated,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.sm,
    color: colors.text,
    fontSize: font.body,
    paddingHorizontal: space.md,
    paddingVertical: space.sm,
  },
  primaryBtn: {
    backgroundColor: colors.accent,
    borderRadius: radius.md,
    paddingVertical: space.md,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 48,
  },
  primaryBtnDisabled: { opacity: 0.45 },
  primaryBtnPressed: { opacity: 0.88 },
  primaryBtnText: {
    color: '#0a0c10',
    fontSize: font.headline,
    fontWeight: '700',
  },
  ghostBtn: { paddingVertical: space.sm, alignItems: 'center' },
  ghostBtnText: { color: colors.textMuted, fontSize: font.body },
  stepRow: { flexDirection: 'row', gap: 8, marginBottom: space.md },
  stepDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.border,
  },
  stepDotActive: { backgroundColor: colors.accentMuted },
  stepDotCurrent: {
    width: 22,
    backgroundColor: colors.accent,
  },
});
