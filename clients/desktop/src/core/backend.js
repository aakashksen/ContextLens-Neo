import { spawn, execFile } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

// Finds the repo (a folder with src/contextlens) and its Python. The app runs `python -m contextlens.api.production`,
// the same entry as tools/run_everything.py, and owns that process: stop() ends the whole tree.
export function resolveBackend({ repoRoot, pythonPath, devRoot, exists = fs.existsSync, platform = process.platform }) {
  const roots = [repoRoot, devRoot].filter(Boolean);
  const root = roots.find((r) => exists(path.join(r, 'src', 'contextlens')));
  if (!root) return { error: 'ContextLens repository not found. Set its folder in Settings (a folder that contains src/contextlens).' };
  const venv = platform === 'win32' ? path.join('.venv', 'Scripts', 'python.exe') : path.join('.venv', 'bin', 'python');
  const python = pythonPath || path.join(root, venv);
  if (!exists(python)) return { error: `Python not found at ${python}. Create the venv (python -m venv .venv, then pip install -e .[server]) or set the Python path in Settings.` };
  return { root, python };
}

export function killTree(child) {
  if (process.platform === 'win32' && child.pid) execFile('taskkill', ['/pid', String(child.pid), '/T', '/F'], () => {});
  else child.kill('SIGTERM');
}

export class BackendProcess {
  constructor({ spawnFn = spawn, kill = killTree } = {}) {
    this.spawnFn = spawnFn;
    this.kill = kill;
    this.child = null;
    this.log = [];
    this.onChange = () => {};
  }

  get running() {
    return !!this.child;
  }

  start({ root, python, assistant }) {
    if (this.child) return;
    // Only the assistant is passed on. DATABASE_URL, SPEECH_ADDR and HF_TOKEN come from the user's own environment.
    const child = this.spawnFn(python, ['-m', 'contextlens.api.production'], {
      cwd: root,
      env: { ...process.env, CONTEXTLENS_ASSISTANT: assistant, PYTHONUNBUFFERED: '1' },
      windowsHide: true,
    });
    this.child = child;
    this.log = [];
    const keep = (b) => {
      this.log.push(...String(b).split(/\r?\n/).filter(Boolean));
      this.log = this.log.slice(-200);
      this.onChange();
    };
    child.stdout?.on('data', keep);
    child.stderr?.on('data', keep);
    child.on('error', (e) => {
      if (this.child === child) this.child = null;
      keep(`failed to start: ${e.message}`);
    });
    child.on('exit', (code) => {
      if (this.child === child) this.child = null;
      keep(`backend exited (code ${code})`);
    });
    this.onChange();
  }

  stop() {
    const child = this.child;
    if (!child) return;
    this.child = null;
    this.kill(child);
    this.onChange();
  }
}
