const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const games = {
  brick_breaker: {
    script: 'brick_breaker_game.py',
    args: ['--physical', '--camera-index', '1'],
  },
  subway_surfers: {
    script: 'subway_surfers_game.py',
    args: ['--physical', '--camera-index', '1'],
  },
  line: {
    script: 'line.py',
    args: ['--physical', '--camera-index', '1'],
  },
  hole: {
    script: 'hole.py',
    args: ['--physical', '--camera-index', '1'],
  },
};

function getScriptsDir() {
  return path.join(__dirname, '..', 'lib', 'wrappers', 'python');
}

function getPythonExecutable() {
  if (process.env.PYTHON_EXECUTABLE) {
    return process.env.PYTHON_EXECUTABLE;
  }

  const contourWallDir = path.join(__dirname, '..');
  const candidatePaths = [
    path.join(contourWallDir, '.venv', 'Scripts', 'python.exe'),
    path.join(contourWallDir, '..', '.venv', 'Scripts', 'python.exe'),
    path.join(contourWallDir, '..', '..', '.venv', 'Scripts', 'python.exe'),
  ];

  for (const candidate of candidatePaths) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return 'python';
}

function quoteArg(value) {
  if (!value) {
    return '""';
  }
  return `"${String(value).replace(/"/g, '\\"')}"`;
}

function launchInTerminal(gameKey) {
  const scriptsDir = getScriptsDir();
  const pythonExec = getPythonExecutable();
  const game = games[gameKey];
  const scriptPath = path.join(scriptsDir, game.script);
  const args = game.args || [];

  if (process.platform === 'win32') {
    const pythonPath = path.normalize(pythonExec);
    const scriptFile = path.normalize(scriptPath);
    const child = spawn(pythonPath, [scriptFile, ...args], {
      cwd: scriptsDir,
      detached: true,
      stdio: 'ignore',
      windowsHide: false,
    });
    child.unref();
    return;
  }

  if (process.platform === 'darwin') {
    const shellArgs = [scriptPath, ...args].map((arg) => `'${String(arg).replace(/'/g, "'\\''")}'`).join(' ');
    const osaScript = `tell application "Terminal" to do script "${pythonExec} ${shellArgs}"`;
    const child = spawn('osascript', ['-e', osaScript], {
      cwd: scriptsDir,
      detached: true,
      stdio: 'ignore',
    });
    child.unref();
    return;
  }

  const child = spawn('x-terminal-emulator', ['-e', pythonExec, scriptPath, ...args], {
    cwd: scriptsDir,
    detached: true,
    stdio: 'ignore',
  });
  child.unref();
}

function createWindow() {
  const win = new BrowserWindow({
    width: 560,
    height: 420,
    resizable: false,
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  win.loadFile(path.join(__dirname, 'index.html'));
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

ipcMain.handle('launch-game', async (event, key) => {
  const game = games[key];
  if (!game) {
    return { ok: false, message: 'Unknown game selection.' };
  }

  try {
    launchInTerminal(key);
    return { ok: true };
  } catch (error) {
    dialog.showErrorBox('Launch failed', String(error));
    return { ok: false, message: String(error) };
  }
});
