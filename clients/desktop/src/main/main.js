import { app, BrowserWindow, Tray, Menu, globalShortcut, ipcMain, nativeImage, screen, session, desktopCapturer } from 'electron';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Controller } from './controller.js';
import { BackendProcess } from '../core/backend.js';
import { checkAll } from '../core/health.js';
import { DEFAULTS, normalizeSettings } from '../core/settings.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const renderer = (f) => path.join(here, '..', 'renderer', f);
const asset = (f) => path.join(here, '..', '..', 'assets', f);
const SMOKE = process.env.CONTEXTLENS_DESKTOP_SMOKE === '1';

// No telemetry, no crash reporter, no auto-update: nothing here talks to anything but the local backend.
if (process.env.CONTEXTLENS_USER_DATA) app.setPath('userData', process.env.CONTEXTLENS_USER_DATA);

const settingsFile = () => path.join(app.getPath('userData'), 'settings.json');
function loadSettings() {
  let raw = {};
  try {
    raw = JSON.parse(fs.readFileSync(settingsFile(), 'utf8'));
  } catch {}
  const base = { ...DEFAULTS };
  if (process.env.CONTEXTLENS_SMOKE_URL) raw.serverUrl = process.env.CONTEXTLENS_SMOKE_URL;
  return normalizeSettings(raw, base).settings;
}
function saveSettings(s) {
  try {
    fs.mkdirSync(path.dirname(settingsFile()), { recursive: true });
    fs.writeFileSync(settingsFile(), JSON.stringify(s, null, 2));
  } catch {}
}

let controller;
let tray;
let mainWin;
let overlay;
let captureWin;
let quitting = false;
let lastCaptureKey = '';

const webPrefs = { preload: path.join(here, 'preload.cjs'), contextIsolation: true, sandbox: true, nodeIntegration: false };

function send(win, channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
}

function createMainWindow() {
  mainWin = new BrowserWindow({ width: 1040, height: 760, show: false, title: 'ContextLens', backgroundColor: '#14161a', webPreferences: webPrefs });
  mainWin.setMenuBarVisibility(false);
  mainWin.loadFile(renderer('index.html'));
  mainWin.on('close', (e) => {
    if (!quitting) {
      e.preventDefault();
      mainWin.hide();
    }
  });
  mainWin.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
}

function showMain() {
  if (!mainWin) return;
  mainWin.show();
  mainWin.focus();
}

function createOverlay() {
  overlay = new BrowserWindow({
    width: 420,
    height: 120,
    show: false,
    frame: false,
    transparent: true,
    resizable: false,
    skipTaskbar: true,
    focusable: false,
    hasShadow: false,
    alwaysOnTop: true,
    webPreferences: { ...webPrefs, backgroundThrottling: false },
  });
  overlay.setAlwaysOnTop(true, 'screen-saver');
  overlay.loadFile(renderer('overlay.html'));
}

let overlaySize = { w: 420, h: 120 };
function placeOverlay() {
  const wa = screen.getPrimaryDisplay().workArea;
  const { w, h } = overlaySize;
  const [v, hz] = controller.settings.corner.split('-');
  const x = hz === 'right' ? wa.x + wa.width - w - 16 : wa.x + 16;
  const y = v === 'bottom' ? wa.y + wa.height - h - 16 : wa.y + 16;
  overlay.setBounds({ x, y, width: w, height: h });
}

function showOverlay(channel, payload, { interactive = false } = {}) {
  if (!overlay || overlay.isDestroyed()) return;
  overlay.setFocusable(interactive);
  overlay.setIgnoreMouseEvents(!interactive, { forward: true });
  placeOverlay();
  send(overlay, channel, payload);
  if (SMOKE) return;
  overlay.showInactive();
  if (interactive) overlay.focus();
}

function hideOverlay() {
  if (overlay && !overlay.isDestroyed()) {
    overlay.hide();
    overlay.setFocusable(false);
  }
}

function createCapture() {
  captureWin = new BrowserWindow({ show: false, width: 200, height: 100, webPreferences: { ...webPrefs, backgroundThrottling: false } });
  captureWin.loadFile(renderer('capture.html'));
}

function wireMedia() {
  const ses = session.defaultSession;
  ses.setPermissionRequestHandler((_wc, permission, cb) => cb(permission === 'media'));
  ses.setPermissionCheckHandler((_wc, permission) => permission === 'media');
  // System audio: the renderer asks for display media with audio and gets the Windows loopback of the whole screen's sound.
  ses.setDisplayMediaRequestHandler(
    async (_req, cb) => {
      try {
        const sources = await desktopCapturer.getSources({ types: ['screen'] });
        cb({ video: sources[0], audio: 'loopback' });
      } catch {
        cb({});
      }
    },
    { useSystemPicker: false },
  );
}

const ACCEL = { press: 'hotkey', ask: 'askHotkey' };
function registerHotkeys() {
  globalShortcut.unregisterAll();
  const errors = [];
  const reg = (key, fn) => {
    const accel = controller.settings[key];
    let ok = false;
    try {
      ok = globalShortcut.register(accel, fn);
    } catch {}
    if (!ok) errors.push(`${accel} could not be registered (another app may use it)`);
  };
  reg(ACCEL.press, () => controller.press());
  reg(ACCEL.ask, () => askMode());
  controller.extra.hotkeyErrors = errors;
  controller.publish();
}

function askMode() {
  showOverlay('overlay-ask', {}, { interactive: true });
}

function buildMenu() {
  const st = controller.snapshot();
  const listening = st.capture.active;
  const menu = Menu.buildFromTemplate([
    { label: listening ? 'Listening' : st.userPaused ? 'Paused' : st.connection === 'recording' ? 'Starting...' : 'Not connected', enabled: false },
    { type: 'separator' },
    { label: `What was said? (${controller.settings.hotkey})`, click: () => controller.press() },
    { label: `Ask a question... (${controller.settings.askHotkey})`, click: () => askMode() },
    { label: st.userPaused ? 'Resume listening' : 'Pause listening', click: () => (st.userPaused ? controller.resume() : controller.pause()) },
    { label: 'Forget this conversation', click: () => controller.forget() },
    { type: 'separator' },
    { label: 'Open ContextLens', click: showMain },
    { label: st.backend.running ? 'Stop backend' : 'Start backend', click: () => (st.backend.running ? controller.stopBackend() : controller.startBackend()) },
    { type: 'separator' },
    { label: 'Quit', click: () => app.quit() },
  ]);
  tray.setContextMenu(menu);
  tray.setImage(nativeImage.createFromPath(asset(listening ? 'tray-live.png' : 'tray.png')));
  tray.setToolTip(listening ? 'ContextLens: listening' : 'ContextLens');
}

function onState(snapshot) {
  send(mainWin, 'state', snapshot);
  const key = JSON.stringify(snapshot.capture);
  if (key !== lastCaptureKey) {
    lastCaptureKey = key;
    send(captureWin, 'capture', snapshot.capture);
  }
  if (tray) buildMenu();
}

function setup() {
  controller = new Controller({
    settings: loadSettings(),
    saveSettings,
    backend: new BackendProcess(),
    health: checkAll,
    devRoot: path.resolve(here, '..', '..', '..', '..'),
    emit: (channel, payload) => {
      if (channel === 'state') return onState(payload);
      if (channel === 'card') showOverlay('overlay-card', { ...payload, seconds: controller.settings.autoHideSeconds });
      if (channel === 'notice') showOverlay('overlay-notice', payload);
      send(mainWin, channel, payload);
    },
  });

  ipcMain.handle('get-state', () => controller.snapshot());
  ipcMain.handle('press', (_e, question) => controller.press(question));
  ipcMain.handle('pause', () => controller.pause());
  ipcMain.handle('resume', () => controller.resume());
  ipcMain.handle('forget', () => controller.forget());
  ipcMain.handle('set-settings', (_e, patch) => {
    const r = controller.setSettings(patch);
    registerHotkeys();
    return r;
  });
  ipcMain.handle('start-backend', () => controller.startBackend());
  ipcMain.handle('stop-backend', () => controller.stopBackend());
  ipcMain.handle('refresh-health', () => controller.refreshHealth());
  ipcMain.handle('open-main', () => showMain());
  ipcMain.handle('overlay-hide', () => hideOverlay());
  ipcMain.handle('overlay-interactive', (_e, on) => {
    if (overlay && !overlay.isDestroyed()) overlay.setIgnoreMouseEvents(!on, { forward: true });
  });
  ipcMain.handle('overlay-size', (_e, h) => {
    overlaySize = { w: 420, h: Math.max(60, Math.min(700, Math.ceil(h))) };
    if (overlay && !overlay.isDestroyed() && overlay.isVisible()) placeOverlay();
  });
  ipcMain.on('audio-chunk', (_e, data, rate) => controller.sendAudio(data, rate));
  ipcMain.on('capture-error', (_e, text) => {
    controller.extra.captureError = text;
    controller.publish();
  });

  wireMedia();
  createMainWindow();
  createOverlay();
  if (!SMOKE) createCapture();
  if (!SMOKE) {
    tray = new Tray(nativeImage.createFromPath(asset('tray.png')));
    tray.on('click', showMain);
    buildMenu();
    registerHotkeys();
  }
  controller.connect();
  controller.refreshHealth();
  setInterval(() => controller.refreshHealth(), 10000).unref();
  if (!SMOKE && !fs.existsSync(settingsFile())) {
    saveSettings(controller.settings);
    showMain();
  }
  if (SMOKE) runSmoke();
}

// CONTEXTLENS_DESKTOP_SMOKE=1 with CONTEXTLENS_SMOKE_URL pointing at tools/fake_server.py: connect, feed fake audio,
// press, and check that the overlay really rendered a card. Prints one JSON line and exits.
async function runSmoke() {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const until = async (fn, ms = 10000) => {
    for (let i = 0; i < ms / 100; i++) {
      if (await fn()) return true;
      await wait(100);
    }
    return false;
  };
  const result = { connected: false, transcript: false, cardInOverlay: false, mainWindowTurns: false };
  result.connected = await until(() => controller.snapshot().connection === 'recording');
  controller.sendAudio('AAAA', 16000);
  result.transcript = await until(() => controller.snapshot().turns.length > 0);
  controller.press();
  result.cardInOverlay = await until(() => overlay.webContents.executeJavaScript("!!document.querySelector('#card .card-answered .quote-by')"));
  result.mainWindowTurns = await until(() => mainWin.webContents.executeJavaScript("document.querySelectorAll('#transcript li').length > 0"));
  console.log(`SMOKE ${JSON.stringify(result)}`);
  quitting = true;
  app.exit(Object.values(result).every(Boolean) ? 0 : 1);
}

if (!app.requestSingleInstanceLock() && !SMOKE) {
  app.quit();
} else {
  app.on('second-instance', showMain);
  app.whenReady().then(setup);
  app.on('window-all-closed', () => {});
  app.on('before-quit', () => {
    quitting = true;
    globalShortcut.unregisterAll();
    controller?.shutdown();
  });
}
