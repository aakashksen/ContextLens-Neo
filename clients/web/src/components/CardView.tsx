import type { Card, Finding } from '../protocol';
import SourceView from './SourceView';

const STATUS_TECHNICAL = {
  answered: 'Answered',
  partial: 'Partial answer',
  abstained: 'Abstained: not enough information',
};
const STATUS_PLAIN = {
  answered: 'Here is what was said',
  partial: 'Only part of an answer',
  abstained: 'Not enough information',
};
const LABEL_TECHNICAL = {
  consistent: 'Consistent',
  update: 'Update',
  contradiction: 'Contradiction',
  insufficient: 'Insufficient',
};
const LABEL_PLAIN = {
  consistent: 'Matches what was said earlier',
  update: 'Changes what was said earlier',
  contradiction: 'Does not match what was said earlier',
  insufficient: 'Not enough to tell',
};

function FindingItem({ finding, plain }: { finding: Finding; plain: boolean }) {
  const label = finding.label ? (plain ? LABEL_PLAIN : LABEL_TECHNICAL)[finding.label] : null;
  return (
    <li className="finding">
      {label && <span className={`badge badge-${finding.label}`}>{label}</span>}
      <p>{finding.statement}</p>
      <SourceView evidence={finding.evidence} />
    </li>
  );
}

export default function CardView({ card, plain }: { card: Card; plain: boolean }) {
  const title = (plain ? STATUS_PLAIN : STATUS_TECHNICAL)[card.status];
  const missing = card.missing ?? [];
  return (
    <article className={`card card-${card.status}`} aria-label="Answer">
      <h2>{title}</h2>
      {card.findings.length > 0 && (
        <section className="recalled" aria-label="From the conversation">
          <h3>From the conversation</h3>
          <ul>
            {card.findings.map((f, i) => (
              <FindingItem key={i} finding={f} plain={plain} />
            ))}
          </ul>
        </section>
      )}
      {missing.length > 0 && (
        <section className="missing" aria-label="What is missing">
          <h3>What is missing</h3>
          <ul>
            {missing.map((m, i) => (
              <li key={i}>{m}</li>
            ))}
          </ul>
        </section>
      )}
      {card.suggestion && (
        <section className="suggestion" aria-label="Suggestion">
          <h3>Suggestion</h3>
          <p className="note">Not from the conversation. It is an idea, not something anyone said.</p>
          <p>{card.suggestion}</p>
        </section>
      )}
    </article>
  );
}
