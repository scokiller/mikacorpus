'use strict';

// Mika Corpus R13 — robust Screen Wake Lock for iOS Home Screen PWA.
// Keeps the existing training engine untouched and only overrides wake-lock handling.
let mikaWakeHeartbeat = null;
let mikaWakeAcquireInFlight = false;
let mikaWakeRetryTimer = null;

function mikaWakeNote(message) {
  try { log(`WakeLock · ${message}`); } catch {}
}

function mikaWakeSetUi(state) {
  try {
    ui.gpu.dataset.wake = state;
    const base = String(ui.gpu.textContent || 'GPU').replace(/ · (Écran éveillé ✓|WakeLock ✗|WakeLock relâché)$/,'');
    const suffix = state === 'active' ? 'Écran éveillé ✓' : state === 'released' ? 'WakeLock relâché' : 'WakeLock ✗';
    ui.gpu.textContent = `${base} · ${suffix}`;
  } catch {}
}

function mikaWakeScheduleRetry(delay = 1000) {
  if (mikaWakeRetryTimer) clearTimeout(mikaWakeRetryTimer);
  if (!running || document.visibilityState !== 'visible') return;
  mikaWakeRetryTimer = setTimeout(() => {
    mikaWakeRetryTimer = null;
    requestWakeLock().catch(() => {});
  }, delay);
}

function mikaWakeStartHeartbeat() {
  if (mikaWakeHeartbeat) return;
  mikaWakeHeartbeat = setInterval(() => {
    if (!running || document.visibilityState !== 'visible') return;
    if (!wakeLock || wakeLock.released) requestWakeLock().catch(() => {});
  }, 10000);
}

function mikaWakeStopHeartbeat() {
  if (mikaWakeHeartbeat) clearInterval(mikaWakeHeartbeat);
  mikaWakeHeartbeat = null;
  if (mikaWakeRetryTimer) clearTimeout(mikaWakeRetryTimer);
  mikaWakeRetryTimer = null;
}

requestWakeLock = async function robustRequestWakeLock() {
  if (!running) return false;
  if (document.visibilityState !== 'visible') return false;
  if (wakeLock && !wakeLock.released) {
    mikaWakeSetUi('active');
    mikaWakeStartHeartbeat();
    return true;
  }
  if (mikaWakeAcquireInFlight) return false;
  if (!navigator.wakeLock || typeof navigator.wakeLock.request !== 'function') {
    mikaWakeNote('API indisponible dans cette PWA/iOS — verrou écran non acquis.');
    mikaWakeSetUi('unsupported');
    mikaWakeStartHeartbeat();
    return false;
  }

  mikaWakeAcquireInFlight = true;
  try {
    const sentinel = await navigator.wakeLock.request('screen');
    wakeLock = sentinel;
    if (!sentinel || sentinel.released) throw new Error('sentinelle déjà relâchée');
    mikaWakeNote('verrou écran acquis ✓');
    mikaWakeSetUi('active');
    sentinel.addEventListener('release', () => {
      if (wakeLock === sentinel) wakeLock = null;
      mikaWakeNote('verrou relâché par iOS; réacquisition programmée.');
      mikaWakeSetUi('released');
      mikaWakeScheduleRetry(500);
    }, { once: true });
    mikaWakeStartHeartbeat();
    return true;
  } catch (e) {
    wakeLock = null;
    const msg = e instanceof Error ? `${e.name || 'Error'}: ${e.message}` : String(e);
    mikaWakeNote(`échec acquisition: ${msg}`);
    mikaWakeSetUi('failed');
    mikaWakeStartHeartbeat();
    mikaWakeScheduleRetry(1500);
    return false;
  } finally {
    mikaWakeAcquireInFlight = false;
  }
};

releaseWakeLock = async function robustReleaseWakeLock() {
  mikaWakeStopHeartbeat();
  const current = wakeLock;
  wakeLock = null;
  try {
    if (current && !current.released) await current.release();
  } catch {}
};

for (const eventName of ['pageshow', 'focus']) {
  window.addEventListener(eventName, () => {
    if (running && document.visibilityState === 'visible') requestWakeLock().catch(() => {});
  });
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && running) {
    mikaWakeScheduleRetry(100);
  }
});
