import type { Detail } from '../protocol';

export type TextSize = 'small' | 'medium' | 'large';

export interface SettingsValue {
  detail: Detail;
  plain: boolean;
  size: TextSize;
}

export default function Settings({ value, onChange }: { value: SettingsValue; onChange: (v: SettingsValue) => void }) {
  return (
    <fieldset className="settings">
      <legend>Settings</legend>
      <label>
        Detail
        <select value={value.detail} onChange={(e) => onChange({ ...value, detail: e.target.value as Detail })}>
          <option value="brief">Brief</option>
          <option value="standard">Standard</option>
          <option value="detailed">Detailed</option>
        </select>
      </label>
      <label>
        Text size
        <select value={value.size} onChange={(e) => onChange({ ...value, size: e.target.value as TextSize })}>
          <option value="small">Small</option>
          <option value="medium">Medium</option>
          <option value="large">Large</option>
        </select>
      </label>
      <label>
        <input type="checkbox" checked={value.plain} onChange={(e) => onChange({ ...value, plain: e.target.checked })} />
        Plain wording
      </label>
    </fieldset>
  );
}
