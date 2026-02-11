const statusEl = document.querySelector('[data-status]');

document.querySelectorAll('[data-game]').forEach((button) => {
  button.addEventListener('click', async () => {
    const key = button.getAttribute('data-game');
    statusEl.textContent = 'Launching...';
    const result = await window.launcher.launchGame(key);
    if (result.ok) {
      statusEl.textContent = 'Launched in a new terminal window.';
    } else {
      statusEl.textContent = result.message || 'Failed to launch.';
    }
  });
});
