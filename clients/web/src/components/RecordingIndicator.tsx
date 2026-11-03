import type { Connection } from '../session';

export type IndicatorState =
  | Connection
  | 'starting'
  | 'blocked'
  | 'mic-stopped'
  | 'error';

const TEXT: Record<IndicatorState, string> = {
  connecting: 'Connecting',
  starting: 'Starting microphone',
  recording: 'Recording',
  blocked: 'Microphone blocked',
  'mic-stopped': 'Microphone stopped',
  error: 'Error',
  paused: 'Paused',
  deleting: 'Deleting',
  deleted: 'Deleted',
  closed: 'Disconnected',
};

export default function RecordingIndicator({ state }: { state: IndicatorState }) {
  return (
    <div role="status" aria-label="Recording indicator" className={`indicator indicator-${state}`}>
      <span className="dot" aria-hidden="true" />
      {TEXT[state]}
    </div>
  );
}
