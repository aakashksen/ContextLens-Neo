import type { Connection } from '../session';

const TEXT: Record<Connection, string> = {
  connecting: 'Connecting',
  recording: 'Recording',
  paused: 'Paused',
  deleted: 'Deleted',
  closed: 'Disconnected',
};

export default function RecordingIndicator({ connection }: { connection: Connection }) {
  return (
    <div role="status" aria-label="Recording indicator" className={`indicator indicator-${connection}`}>
      <span className="dot" aria-hidden="true" />
      {TEXT[connection]}
    </div>
  );
}
