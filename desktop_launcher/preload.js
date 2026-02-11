const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('launcher', {
  launchGame: (key) => ipcRenderer.invoke('launch-game', key),
});
