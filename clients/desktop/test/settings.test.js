import { describe, expect, it } from 'vitest';
import { DEFAULTS, normalizeSettings, validAccelerator, validWsUrl } from '../src/core/settings.js';

describe('settings', () => {
  it('defaults: team_routed, Ctrl+Shift+Space, microphone only', () => {
    expect(DEFAULTS).toMatchObject({ assistant: 'team_routed', hotkey: 'Ctrl+Shift+Space', mic: true, loopback: false, detail: 'standard' });
  });
  it('validates accelerators: modifier plus one key', () => {
    for (const ok of ['Ctrl+Shift+Space', 'Alt+F9', 'CommandOrControl+K', 'Ctrl+Alt+/']) expect(validAccelerator(ok)).toBe(true);
    for (const bad of ['Space', 'Ctrl', 'Ctrl+', 'Ctrl+Shift', 'Ctrl+Foo+K', '', 5, 'Ctrl+AB']) expect(validAccelerator(bad)).toBe(false);
  });
  it('validates server urls', () => {
    expect(validWsUrl('ws://127.0.0.1:8000')).toBe(true);
    expect(validWsUrl('http://x')).toBe(false);
    expect(validWsUrl('nonsense')).toBe(false);
  });
  it('keeps valid values, rejects invalid ones and keeps the old value', () => {
    const { settings, rejected } = normalizeSettings({ detail: 'brief', assistant: 'bogus', autoHideSeconds: 1, hotkey: 'nope', serverUrl: 'ws://h:1/' });
    expect(settings).toMatchObject({ detail: 'brief', assistant: 'team_routed', autoHideSeconds: 12, hotkey: 'Ctrl+Shift+Space', serverUrl: 'ws://h:1' });
    expect(rejected.sort()).toEqual(['assistant', 'autoHideSeconds', 'hotkey']);
  });
  it('refuses the same key for both hotkeys', () => {
    const { settings, rejected } = normalizeSettings({ askHotkey: 'ctrl+shift+space' });
    expect(rejected).toContain('askHotkey');
    expect(settings.hotkey).not.toBe(settings.askHotkey);
  });
});
