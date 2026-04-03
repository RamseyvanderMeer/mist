import { useCallback, useEffect, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import type { Recommendation } from '../../src/api/types';
import {
  apiClarify,
  apiFeedbackOutcome,
  apiFeedbackRating,
  apiQuery,
} from '../../src/api/mistApi';
import {
  Body,
  Card,
  Field,
  GhostButton,
  PrimaryButton,
  Screen,
  StepPill,
  Subtitle,
  Title,
} from '../../src/components/ui';
import { useMistAuth } from '../../src/auth/AuthContext';
import { colors, font, space } from '../../src/theme/tokens';

type Phase = 'input' | 'clarify' | 'results';

function parseFaultCodes(raw: string): string[] {
  return raw
    .split(/[\s,;]+/)
    .map((s) => s.trim().toUpperCase())
    .filter(Boolean);
}

export default function DiagnoseScreen() {
  const { getAuthHeaders } = useMistAuth();

  const [phase, setPhase] = useState<Phase>('input');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [faultRaw, setFaultRaw] = useState('');
  const [description, setDescription] = useState('');
  const [vehicleModel, setVehicleModel] = useState('');
  const [vehicleYear, setVehicleYear] = useState('');
  const [obdRaw, setObdRaw] = useState('{}');

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [queryText, setQueryText] = useState('');
  const [questions, setQuestions] = useState<string[]>([]);
  const [answers, setAnswers] = useState<string[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [selectedGuideId, setSelectedGuideId] = useState<string | null>(null);

  const [rating, setRating] = useState<number | null>(null);
  const [ratingSent, setRatingSent] = useState(false);
  const [outcomeSent, setOutcomeSent] = useState(false);

  useEffect(() => {
    setAnswers(questions.map(() => ''));
  }, [questions]);

  const resetWorkflow = useCallback(() => {
    setPhase('input');
    setError(null);
    setFaultRaw('');
    setDescription('');
    setVehicleModel('');
    setVehicleYear('');
    setObdRaw('{}');
    setSessionId(null);
    setQueryText('');
    setQuestions([]);
    setAnswers([]);
    setRecommendations([]);
    setExpandedId(null);
    setSelectedGuideId(null);
    setRating(null);
    setRatingSent(false);
    setOutcomeSent(false);
  }, []);

  const stepIndex = phase === 'input' ? 1 : phase === 'clarify' ? 2 : 3;

  const runQuery = async () => {
    setError(null);
    const codes = parseFaultCodes(faultRaw);
    if (codes.length === 0 && !description.trim()) {
      setError('Add at least one fault code or a symptom description.');
      return;
    }
    let obd_data: Record<string, unknown> = {};
    if (obdRaw.trim() && obdRaw.trim() !== '{}') {
      try {
        obd_data = JSON.parse(obdRaw) as Record<string, unknown>;
      } catch {
        setError('OBD data must be valid JSON.');
        return;
      }
    }
    const vehicle_context: Record<string, unknown> = {};
    if (vehicleModel.trim()) vehicle_context.model = vehicleModel.trim();
    if (vehicleYear.trim()) vehicle_context.year = vehicleYear.trim();

    setBusy(true);
    try {
      const res = await apiQuery(getAuthHeaders, {
        fault_codes: codes,
        obd_data,
        description: description.trim() || null,
        vehicle_context: Object.keys(vehicle_context).length ? vehicle_context : null,
        session_id: null,
      });
      setSessionId(res.session_id);
      setQueryText(res.query_text);
      if (res.needs_clarification && res.clarification_questions?.length) {
        setQuestions(res.clarification_questions);
        setPhase('clarify');
      } else {
        setRecommendations(res.recommendations);
        setPhase('results');
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Query failed');
    } finally {
      setBusy(false);
    }
  };

  const runClarify = async () => {
    if (!sessionId) return;
    setError(null);
    const responses = answers.map((a) => (a.trim() ? a.trim() : '—'));
    if (responses.length !== questions.length) {
      setError('Answer each question.');
      return;
    }
    setBusy(true);
    try {
      const res = await apiClarify(getAuthHeaders, { session_id: sessionId, responses });
      setQueryText(res.query_text);
      if (res.needs_clarification && res.clarification_questions?.length) {
        setQuestions(res.clarification_questions);
      } else {
        setRecommendations(res.recommendations);
        setPhase('results');
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Clarify failed');
    } finally {
      setBusy(false);
    }
  };

  const submitRating = async () => {
    if (!sessionId || rating === null) return;
    setBusy(true);
    try {
      await apiFeedbackRating(getAuthHeaders, {
        session_id: sessionId,
        rating,
        selected_guide: selectedGuideId,
      });
      setRatingSent(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Feedback failed');
    } finally {
      setBusy(false);
    }
  };

  const submitOutcome = async (outcome: 'success' | 'failure' | 'partial') => {
    if (!sessionId) return;
    setBusy(true);
    try {
      await apiFeedbackOutcome(getAuthHeaders, { session_id: sessionId, outcome });
      setOutcomeSent(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Outcome failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <Screen scroll>
      <StepPill n={stepIndex} total={3} />
      <Title>Diagnosis</Title>
      <Body muted>
        Session: {sessionId ? `${sessionId.slice(0, 8)}…` : '—'}
      </Body>

      {phase === 'input' && (
        <View>
          <Card>
            <Subtitle>1 · Vehicle & codes</Subtitle>
            <Field
              label="Fault codes (spaces, commas, or lines)"
              placeholder="P0301 2A87 29CC"
              value={faultRaw}
              onChangeText={setFaultRaw}
              multiline
              style={styles.multiline}
            />
            <Field
              label="Symptom description"
              placeholder="Rough idle, MIL on after cold start…"
              value={description}
              onChangeText={setDescription}
              multiline
              style={styles.multiline}
            />
            <Field label="Model (optional)" placeholder="F30" value={vehicleModel} onChangeText={setVehicleModel} />
            <Field label="Year (optional)" placeholder="2018" value={vehicleYear} onChangeText={setVehicleYear} />
            <Field
              label="OBD snapshot JSON (optional)"
              placeholder="{}"
              value={obdRaw}
              onChangeText={setObdRaw}
              multiline
              style={styles.jsonBox}
            />
            <PrimaryButton title="Search guides" onPress={runQuery} loading={busy} disabled={busy} />
          </Card>
        </View>
      )}

      {phase === 'clarify' && (
        <Card>
          <Subtitle>2 · Clarification</Subtitle>
          <Body muted>The model needs a bit more context. Answer each question below.</Body>
          {questions.map((q, i) => (
            <View key={i} style={styles.qBlock}>
              <Text style={styles.qText}>{q}</Text>
              <Field
                label={`Your answer ${i + 1}`}
                value={answers[i] || ''}
                onChangeText={(t) => {
                  const next = [...answers];
                  next[i] = t;
                  setAnswers(next);
                }}
                multiline
                style={styles.multiline}
              />
            </View>
          ))}
          <PrimaryButton title="Submit answers" onPress={runClarify} loading={busy} disabled={busy} />
          <GhostButton title="Start over" onPress={resetWorkflow} />
        </Card>
      )}

      {phase === 'results' && (
        <View>
          <Card>
            <Subtitle>3 · Ranked guides</Subtitle>
            <Text style={styles.queryText} selectable>
              {queryText}
            </Text>
            <Body muted>{recommendations.length} result(s)</Body>
          </Card>

          {recommendations.map((item) => {
            const open = expandedId === item.id;
            return (
              <Pressable
                key={item.id}
                onPress={() => setExpandedId(open ? null : item.id)}
                style={({ pressed }) => [styles.recCard, pressed && { opacity: 0.92 }]}
              >
                <View style={styles.recHeader}>
                  <Text style={styles.recTitle} numberOfLines={open ? undefined : 2}>
                    {item.title}
                  </Text>
                  <Text style={styles.recScore}>{(item.score * 100).toFixed(0)}%</Text>
                </View>
                <Text style={styles.recProc}>{item.procedure_name}</Text>
                {open && item.text ? <Text style={styles.recBody}>{item.text}</Text> : null}
                <Pressable
                  style={styles.pickBtn}
                  onPress={() => {
                    setSelectedGuideId(item.id);
                    setExpandedId(item.id);
                  }}
                >
                  <Text style={selectedGuideId === item.id ? styles.pickActive : styles.pickText}>
                    {selectedGuideId === item.id ? 'Selected for feedback' : 'Use this guide for feedback'}
                  </Text>
                </Pressable>
              </Pressable>
            );
          })}

          <Card>
            <Subtitle>Feedback</Subtitle>
            <Body muted>Help MIST learn: rate this session and optionally record the repair outcome.</Body>
            <Text style={styles.rateLabel}>Rating {rating ? `${rating} / 5` : '—'}</Text>
            <View style={styles.stars}>
              {[1, 2, 3, 4, 5].map((n) => (
                <Pressable key={n} onPress={() => setRating(n)} style={styles.starHit}>
                  <Text style={[styles.star, rating !== null && n <= rating ? styles.starOn : styles.starOff]}>
                    ★
                  </Text>
                </Pressable>
              ))}
            </View>
            <PrimaryButton
              title={ratingSent ? 'Rating saved' : 'Submit rating'}
              onPress={submitRating}
              loading={busy}
              disabled={busy || rating === null || ratingSent}
            />

            <Text style={[styles.rateLabel, { marginTop: space.md }]}>Repair outcome</Text>
            <View style={styles.outcomeRow}>
              <Pressable
                style={[styles.outcomeBtn, outcomeSent && styles.outcomeDisabled]}
                disabled={outcomeSent || busy}
                onPress={() => submitOutcome('success')}
              >
                <Text style={styles.outcomeBtnText}>Fixed</Text>
              </Pressable>
              <Pressable
                style={[styles.outcomeBtn, outcomeSent && styles.outcomeDisabled]}
                disabled={outcomeSent || busy}
                onPress={() => submitOutcome('partial')}
              >
                <Text style={styles.outcomeBtnText}>Partial</Text>
              </Pressable>
              <Pressable
                style={[styles.outcomeBtn, outcomeSent && styles.outcomeDisabled]}
                disabled={outcomeSent || busy}
                onPress={() => submitOutcome('failure')}
              >
                <Text style={styles.outcomeBtnText}>Not fixed</Text>
              </Pressable>
            </View>
            {outcomeSent ? <Body muted>Outcome recorded.</Body> : null}
          </Card>

          <PrimaryButton title="New diagnosis" onPress={resetWorkflow} />
        </View>
      )}

      {error ? (
        <Card style={{ borderColor: colors.danger }}>
          <Text style={{ color: colors.danger }}>{error}</Text>
        </Card>
      ) : null}

      <View style={{ height: space.xl }} />
    </Screen>
  );
}

const styles = StyleSheet.create({
  multiline: { minHeight: 72, textAlignVertical: 'top' },
  jsonBox: { minHeight: 100, fontFamily: 'monospace', fontSize: font.caption },
  qBlock: { marginBottom: space.md },
  qText: { color: colors.text, fontSize: font.body, marginBottom: space.sm },
  queryText: {
    color: colors.textMuted,
    fontSize: font.caption,
    marginBottom: space.sm,
    lineHeight: 20,
  },
  recCard: {
    backgroundColor: colors.bgCard,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    padding: space.md,
    marginBottom: space.sm,
  },
  recHeader: { flexDirection: 'row', justifyContent: 'space-between', gap: space.sm },
  recTitle: { color: colors.text, fontSize: font.headline, fontWeight: '600', flex: 1 },
  recScore: { color: colors.accent, fontWeight: '700' },
  recProc: { color: colors.textMuted, fontSize: font.caption, marginTop: 4 },
  recBody: { color: colors.text, fontSize: font.caption, marginTop: space.sm, lineHeight: 20 },
  pickBtn: { marginTop: space.sm, alignSelf: 'flex-start' },
  pickText: { color: colors.textMuted, fontSize: font.caption },
  pickActive: { color: colors.accent, fontSize: font.caption, fontWeight: '600' },
  rateLabel: { color: colors.textMuted, fontSize: font.caption, marginBottom: space.xs },
  stars: { flexDirection: 'row', marginBottom: space.md },
  starHit: { paddingRight: space.sm, paddingVertical: space.xs },
  star: { fontSize: 32, lineHeight: 36 },
  starOn: { color: colors.accent },
  starOff: { color: colors.border },
  outcomeRow: { flexDirection: 'row', flexWrap: 'wrap', gap: space.sm, marginBottom: space.sm },
  outcomeBtn: {
    paddingVertical: space.sm,
    paddingHorizontal: space.md,
    borderRadius: 8,
    backgroundColor: colors.bgElevated,
    borderWidth: 1,
    borderColor: colors.border,
  },
  outcomeDisabled: { opacity: 0.4 },
  outcomeBtnText: { color: colors.text, fontWeight: '600', fontSize: font.caption },
});
