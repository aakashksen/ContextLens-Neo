// Mirrors docs/client-contract.schema.json.
export type Detail = 'brief' | 'standard' | 'detailed';

export interface Turn {
  id: string;
  conversation_id: string;
  seq: number;
  speaker: string;
  text: string;
  start_ms: number;
  end_ms: number;
  source: 'dataset' | 'synthetic' | 'asr';
  speaker_confidence?: number;
}

export interface EvidenceRef {
  turn_id: string;
  quote: string;
  speaker: string;
  start_ms: number;
}

export interface Finding {
  agent: string;
  kind: 'recall' | 'consistency' | 'context';
  statement: string;
  evidence: EvidenceRef[];
  confidence: number;
  label?: 'consistent' | 'update' | 'contradiction' | 'insufficient' | null;
}

export interface Card {
  mode: 'recall' | 'clarify' | 'respond';
  status: 'answered' | 'partial' | 'abstained';
  findings: Finding[];
  suggestion?: string | null;
  missing?: string[];
  latency_ms?: number;
  tokens?: number;
  cost_usd?: number;
}

export type StatusState = 'recording' | 'paused' | 'deleted' | 'error';

export type ServerMessage =
  | { type: 'transcript'; turn: Turn }
  | { type: 'turn_update'; turn: Turn }
  | { type: 'card'; card: Card }
  | { type: 'status'; state: StatusState; detail?: string | null };

export type ClientMessage =
  | { type: 'audio'; data: string; sample_rate?: number }
  | {
      type: 'press';
      request: { conversation_id: string; as_of_seq: number; question?: string | null; detail?: Detail };
    }
  | { type: 'pause' }
  | { type: 'resume' }
  | { type: 'delete' };

const STATES = ['recording', 'paused', 'deleted', 'error'];
const STATUSES = ['answered', 'partial', 'abstained'];

export function parseServerMessage(raw: unknown): ServerMessage | null {
  if (typeof raw !== 'string') return null;
  let m: any;
  try {
    m = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!m || typeof m !== 'object') return null;
  if (m.type === 'transcript' && m.turn && typeof m.turn.seq === 'number' && typeof m.turn.text === 'string') {
    return m as ServerMessage;
  }
  if (m.type === 'turn_update' && m.turn && typeof m.turn.seq === 'number' && typeof m.turn.id === 'string') {
    return m as ServerMessage;
  }
  if (m.type === 'card' && m.card && STATUSES.includes(m.card.status) && Array.isArray(m.card.findings)) {
    return m as ServerMessage;
  }
  if (m.type === 'status' && STATES.includes(m.state)) return m as ServerMessage;
  return null;
}
