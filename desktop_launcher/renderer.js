const statusEl = document.querySelector('[data-status]');
const nameInput = document.querySelector('[data-player]');

const apiConfig = {
  url: 'https://api.deltacraft.io/api/users',
  apiPassword: 'KaAsKrOkAnTjE123',
};

const storedName = localStorage.getItem('playerName');
if (storedName && nameInput) {
  nameInput.value = storedName;
}

if (nameInput) {
  nameInput.addEventListener('input', () => {
    localStorage.setItem('playerName', nameInput.value.trim());
  });
}

async function capturePhoto() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera access is not available.');
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoInputs = devices.filter((device) => device.kind === 'videoinput');
  if (!videoInputs.length) {
    throw new Error('No camera devices found.');
  }

  const preferredDevice = videoInputs[1] || videoInputs[0];
  const deviceQueue = [preferredDevice, ...videoInputs.filter((d) => d.deviceId !== preferredDevice.deviceId)];

  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  const canvas = document.createElement('canvas');

  const captureFromStream = async (stream) => {
    video.srcObject = stream;
    await video.play();

    const ready = await new Promise((resolve) => {
      const timeoutAt = Date.now() + 8000;
      const checkReady = () => {
        if (video.readyState >= 3 && video.videoWidth > 0 && video.videoHeight > 0) {
          resolve(true);
          return;
        }
        if (Date.now() >= timeoutAt) {
          resolve(false);
          return;
        }
        requestAnimationFrame(checkReady);
      };
      checkReady();
    });

    if (!ready) {
      throw new Error('Camera did not respond.');
    }

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to capture camera frame.');
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.88));
    if (!blob) {
      throw new Error('Failed to encode camera image.');
    }
    return blob;
  };

  let lastError = null;
  for (const device of deviceQueue) {
    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          deviceId: device?.deviceId ? { exact: device.deviceId } : undefined,
        },
        audio: false,
      });
      return await captureFromStream(stream);
    } catch (error) {
      lastError = error;
    } finally {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    }
  }

  throw lastError || new Error('Camera did not respond.');
}

async function registerUser(playerName) {
  const imageBlob = await capturePhoto();
  const imageBuffer = await imageBlob.arrayBuffer();
  const result = await window.launcher.uploadUser({
    playerName,
    imageBuffer,
  });

  if (!result.ok) {
    throw new Error(result.message || 'Upload failed.');
  }

  return result.data;
}

document.querySelectorAll('[data-game]').forEach((button) => {
  button.addEventListener('click', async () => {
    const key = button.getAttribute('data-game');
    const playerName = nameInput ? nameInput.value.trim() : '';
    let registrationError = null;

    if (playerName) {
      statusEl.textContent = 'Registering player...';
      try {
        await registerUser(playerName);
      } catch (error) {
        registrationError = error;
      }
    }

    statusEl.textContent = 'Launching...';
    const result = await window.launcher.launchGame(key, playerName);
    if (result.ok) {
      const name = nameInput ? nameInput.value.trim() : '';
      if (registrationError) {
        statusEl.textContent = `Launched for ${name}. Registration failed.`;
      } else {
        statusEl.textContent = name
          ? `Launched in a new terminal window for ${name}.`
          : 'Launched in a new terminal window.';
      }
    } else {
      statusEl.textContent = result.message || 'Failed to launch.';
    }
  });
});
