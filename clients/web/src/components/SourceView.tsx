import { useState } from 'react';
import type { EvidenceRef } from '../protocol';

export function formatTime(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

export default function SourceView({ evidence }: { evidence: EvidenceRef[] }) {
  const [open, setOpen] = useState(false);
  if (evidence.length === 0) return null;
  return (
    <div className="source">
      <button type="button" aria-expanded={open} onClick={() => setOpen(!open)}>
        {open ? 'Hide source' : 'Show source'}
      </button>
      {open && (
        <ul>
          {evidence.map((e, i) => (
            <li key={`${e.turn_id}-${i}`}>
              <span className="source-speaker">{e.speaker}</span> <time>{formatTime(e.start_ms)}</time>
              <blockquote>{e.quote}</blockquote>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
