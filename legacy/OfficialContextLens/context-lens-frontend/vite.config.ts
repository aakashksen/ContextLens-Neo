import path from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { spawn } from 'child_process';
import readline from 'readline';

const isRealtime = process.env.VITE_REALTIME === '1';

const realtimePlugin = () => ({
  name: 'context-lens-realtime',
  configureServer(server: any) {
    if (!isRealtime) return;
    const clients = new Set<any>();
    const backendScript = path.resolve(__dirname, '..', 'context-lens-backend', 'tools', 'user-log-viewer.js');
    const child = spawn(process.execPath, [backendScript], {
      env: { ...process.env, USER_LOG_VIEWER_DEBUG: '0' }
    });

    const rl = readline.createInterface({ input: child.stdout });
    rl.on('line', (line) => {
      clients.forEach((res) => {
        res.write(`data: ${line}\n\n`);
      });
    });

    child.stderr.on('data', (chunk) => {
      process.stderr.write(chunk);
    });

    server.middlewares.use('/realtime', (req: any, res: any) => {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive'
      });
      res.write('retry: 1000\n\n');
      clients.add(res);
      req.on('close', () => {
        clients.delete(res);
      });
    });

    server.httpServer?.on('close', () => {
      try {
        rl.close();
      } catch {}
      if (child) {
        child.kill();
      }
    });
  }
});

export default defineConfig({
  plugins: [react(), realtimePlugin()],
  server: {
    host: '0.0.0.0', // Allow external connections
    port: 5173,
    strictPort: true
  },
  define: {
    //'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
    //'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.'),
    }
  }
});
