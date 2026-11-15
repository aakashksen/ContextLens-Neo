// Mirrors docs/client-contract.md. Messages are one JSON text frame each with a `type`.
const STATES = ['recording', 'paused', 'deleted', 'error'];
const STATUSES = ['answered', 'partial', 'abstained'];

export function parseServerMessage(raw) {
  if (typeof raw !== 'string') return null;
  let m;
  try {
    m = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!m || typeof m !== 'object') return null;
  if (m.type === 'transcript' && m.turn && typeof m.turn.seq === 'number' && typeof m.turn.text === 'string') return m;
  if (m.type === 'turn_update' && m.turn && typeof m.turn.seq === 'number' && typeof m.turn.id === 'string') return m;
  if (m.type === 'card' && m.card && STATUSES.includes(m.card.status) && Array.isArray(m.card.findings)) return m;
  if (m.type === 'status' && STATES.includes(m.state)) return m;
  // Optional, not part of the contract yet: the backend may send its stored memory items.
  if (m.type === 'memory' && Array.isArray(m.items)) return m;
  return null;
}

export const audioMessage = (data, sampleRate = 16000) => ({ type: 'audio', data, sample_rate: sampleRate });
export const pressMessage = (conversationId, asOfSeq, detail, question) => ({
  type: 'press',
  request: { conversation_id: conversationId, as_of_seq: asOfSeq, detail, ...(question ? { question } : {}) },
});
