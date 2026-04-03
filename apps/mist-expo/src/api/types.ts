export type HealthResponse = {
  status: string;
  service: string;
};

export type AuthCheckResponse =
  | {
      authenticated: false;
      registered: false;
      message: string;
    }
  | {
      authenticated: true;
      registered: false;
      email: string;
      message: string;
    }
  | {
      authenticated: true;
      registered: true;
      user: Record<string, unknown>;
    };

export type UserMe = {
  id: string;
  email: string;
  display_name?: string | null;
  status: string;
  tier?: string | null;
  roles: string[];
  created_at?: string | null;
  last_login_at?: string | null;
};

export type Recommendation = {
  id: string;
  title: string;
  procedure_name: string;
  procedure_id?: string | null;
  score: number;
  text?: string | null;
};

export type QueryResponse = {
  recommendations: Recommendation[];
  needs_clarification: boolean;
  clarification_questions?: string[] | null;
  session_id: string;
  query_text: string;
};

export type RepairOutcome = 'success' | 'failure' | 'partial';
