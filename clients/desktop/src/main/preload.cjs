const { contextBridge, ipcRenderer } = require('electron');

const INVOKE = ['get-state', 'press', 'pause', 'resume', 'forget', 'set-settings', 'start-backend', 'stop-backend', 'refresh-health', 'open-main', 'overlay-hide', 'overlay-interactive', 'overlay-size'];
const ON = ['state', 'card', 'notice', 'overlay-card', 'overlay-notice', 'overlay-ask', 'capture'];
const SEND = ['audio-chunk', 'capture-error'];

contextBridge.exposeInMainWorld('cl', {
  invoke: (channel, ...args) => (INVOKE.includes(channel) ? ipcRenderer.invoke(channel, ...args) : Promise.reject(new Error('blocked channel'))),
  send: (channel, ...args) => {
    if (SEND.includes(channel)) ipcRenderer.send(channel, ...args);
  },
  on: (channel, fn) => {
    if (ON.includes(channel)) ipcRenderer.on(channel, (_e, payload) => fn(payload));
  },
});
