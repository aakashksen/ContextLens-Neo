import { useEffect, useRef } from 'react';

export default function DeleteConfirm({ onConfirm, onCancel }: { onConfirm: () => void; onCancel: () => void }) {
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => cancel.current?.focus(), []);
  return (
    <div
      role="alertdialog"
      aria-labelledby="delete-title"
      aria-describedby="delete-desc"
      className="confirm"
      onKeyDown={(e) => e.key === 'Escape' && onCancel()}
    >
      <h2 id="delete-title">Delete this conversation?</h2>
      <p id="delete-desc">The stored transcript will be removed and cannot be brought back.</p>
      <button type="button" ref={cancel} onClick={onCancel}>
        Cancel
      </button>
      <button type="button" className="danger" onClick={onConfirm}>
        Yes, delete
      </button>
    </div>
  );
}
