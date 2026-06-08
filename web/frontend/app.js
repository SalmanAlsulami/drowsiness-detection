'use strict';

// ── DOM references ─────────────────────────────────────────────────────────
const $video         = document.getElementById('video');
const $canvas        = document.getElementById('canvas');
const $startBtn      = document.getElementById('start-btn');
const $stopBtn       = document.getElementById('stop-btn');
const $badge         = document.getElementById('status-badge');
const $badgeText     = document.getElementById('badge-text');
const $banner        = document.getElementById('alert-banner');
const $alertText     = document.getElementById('alert-text');
const $bannerIcon    = document.getElementById('banner-icon');
const $idleOverlay   = document.getElementById('idle-overlay');
const $cameraFrame   = document.getElementById('camera-frame');
const $noFace        = document.getElementById('no-face');
const $recDot        = document.getElementById('rec-dot');
const $camStatusText = document.getElementById('cam-status-text');
const $sessionTime   = document.getElementById('session-time');

// PERCLOS
const $arcFill           = document.getElementById('arc-fill');
const $arcPctText        = document.getElementById('arc-pct-text');
const $perclosFramesLabel = document.getElementById('perclos-frames-label');
const $perclosBlock      = document.getElementById('perclos-block');

// Eyes
const $eyeLeftCard  = document.getElementById('eye-left-card');
const $eyeLeftProb  = document.getElementById('eye-left-prob');
const $eyeRightCard = document.getElementById('eye-right-card');
const $eyeRightProb = document.getElementById('eye-right-prob');
const $consecLabel  = document.getElementById('consec-label');

// Yawn
const $yawnCount   = document.getElementById('yawn-count');
const $yawnBlock   = document.getElementById('yawn-block');
const $yawnProbLbl = document.getElementById('yawn-prob-label');
const $pip0        = document.getElementById('pip-0');
const $pip1        = document.getElementById('pip-1');

// Head
const $pitchVal    = document.getElementById('pitch-val');
const $headBlock   = document.getElementById('head-block');
const $pitchNeedle = document.getElementById('pitch-needle');

// Stats
const $fpsVal     = document.getElementById('fps-val');
const $latencyVal = document.getElementById('latency-val');

// ── Constants ──────────────────────────────────────────────────────────────
// Arc gauge: semicircle M 10 105 A 90 90 0 0 0 190 105, r=90
// Arc length = π × 90 ≈ 282.74
// Gauge display range: 0–70% (normalized to that max)
const ARC_LEN    = 282.74;
const ARC_MAX    = 0.70;
const PITCH_RANGE = 40;   // ±40° shown on track

// WebSocket: ws:// on http, wss:// on https (required for Railway)
const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;

// ── Session state ──────────────────────────────────────────────────────────
let ws           = null;
let stream       = null;
let sendLoop     = null;
let frameCount   = 0;
let lastFpsTick  = 0;
let lastSentAt   = 0;
let prevLevel    = '';
let sessionStart = null;
let sessionTimer = null;

// ── Session timer ──────────────────────────────────────────────────────────
function startSessionTimer() {
  sessionStart = Date.now();
  sessionTimer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
    const h   = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const m   = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const sec = String(elapsed % 60).padStart(2, '0');
    $sessionTime.textContent = `${h}:${m}:${sec}`;
  }, 1000);
}

function stopSessionTimer() {
  clearInterval(sessionTimer);
  sessionTimer = null;
  $sessionTime.textContent = '—';
}

// ── Button handlers ────────────────────────────────────────────────────────
$startBtn.addEventListener('click', startInspection);
$stopBtn.addEventListener('click',  stopInspection);

async function startInspection() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
    });
    $video.srcObject = stream;
    await $video.play();
    $idleOverlay.classList.add('hidden');
  } catch {
    alert('Camera access denied.\nPlease allow camera permission and reload the page.');
    return;
  }

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    $startBtn.classList.add('hidden');
    $stopBtn.classList.remove('hidden');

    $canvas.width  = 320;
    $canvas.height = 240;

    $recDot.classList.add('live');
    $camStatusText.textContent = 'LIVE';

    setBadge('safe', 'SAFE');
    setBanner('safe', '✓  DRIVER IS ALERT');

    startSessionTimer();
    lastFpsTick = Date.now();
    frameCount  = 0;

    sendLoop = setInterval(captureAndSend, 100); // 10 fps
  };

  ws.onmessage = ev => {
    const latency = Date.now() - lastSentAt;
    $latencyVal.textContent = latency + 'ms';
    renderResult(JSON.parse(ev.data));
  };

  ws.onerror = () => {
    setBadge('standby', 'ERROR');
  };

  ws.onclose = () => {
    if (stream) stopInspection();
  };
}

function stopInspection() {
  clearInterval(sendLoop);
  sendLoop = null;

  if (ws)     { ws.close();  ws = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  $video.srcObject = null;

  $startBtn.classList.remove('hidden');
  $stopBtn.classList.add('hidden');
  $idleOverlay.classList.remove('hidden');
  $noFace.classList.add('hidden');
  $recDot.classList.remove('live');
  $camStatusText.textContent = 'OFFLINE';
  $cameraFrame.className = 'camera-frame';

  setBadge('standby', 'STANDBY');
  setBanner('standby', 'SYSTEM READY — PRESS INITIATE MONITORING TO BEGIN');

  stopAudioAlert();
  stopSessionTimer();
  resetMetrics();
  prevLevel = '';
}

// ── Frame capture ──────────────────────────────────────────────────────────
// Draws to hidden canvas without CSS mirror transform,
// then sends the raw (un-mirrored) frame to the backend.
function captureAndSend() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !stream) return;

  const ctx = $canvas.getContext('2d');
  ctx.drawImage($video, 0, 0, $canvas.width, $canvas.height);

  lastSentAt = Date.now();
  ws.send($canvas.toDataURL('image/jpeg', 0.82));

  // FPS counter
  frameCount++;
  const now = Date.now();
  if (now - lastFpsTick >= 1000) {
    $fpsVal.textContent = String(frameCount);
    frameCount  = 0;
    lastFpsTick = now;
  }
}

// ── Result renderer ────────────────────────────────────────────────────────
function renderResult(d) {
  const lvl = d.alert_level; // 'safe' | 'warning' | 'danger'

  // Badge + banner + camera frame
  if (lvl === 'danger') {
    setBadge('danger', 'DANGER');
    setBanner('danger', '⚠  DANGER: DRIVER IS DROWSY');
    $cameraFrame.className = 'camera-frame state-danger';
  } else if (lvl === 'warning') {
    setBadge('warning', 'WARNING');
    setBanner('warning', '⚡  WARNING: DROWSINESS SIGNS DETECTED');
    $cameraFrame.className = 'camera-frame state-warning';
  } else {
    setBadge('safe', 'SAFE');
    setBanner('safe', '✓  DRIVER IS ALERT');
    $cameraFrame.className = 'camera-frame';
  }

  // Audio — only fire on level change
  if (lvl !== prevLevel) { triggerAudio(lvl); prevLevel = lvl; }

  // No-face badge
  $noFace.classList.toggle('hidden', d.face_found);

  // PERCLOS arc gauge
  updatePerclosGauge(d.perclos, d.perclos_alert);
  const frames = d.perclos_frames;
  $perclosFramesLabel.textContent = frames >= 30
    ? `${frames} frames tracked`
    : `${frames} frames — warming up (need 30)`;
  $perclosBlock.classList.toggle('alerted', d.perclos_alert);

  // Eyes
  renderEye($eyeLeftCard,  $eyeLeftProb,  d.eye_left_prob);
  renderEye($eyeRightCard, $eyeRightProb, d.eye_right_prob);
  $consecLabel.textContent = `consec. closed: ${d.consec_closed} frames`;

  // Yawn
  $yawnCount.textContent  = d.yawn_events;
  $yawnCount.style.color  = d.yawn_alert ? 'var(--danger)' : d.yawn_events > 0 ? 'var(--warn)' : '';
  $yawnBlock.classList.toggle('alerted', d.yawn_alert);
  updatePips(d.yawn_events, d.yawn_alert);
  if (d.yawn_prob != null) {
    $yawnProbLbl.textContent = `Probability: ${(d.yawn_prob * 100).toFixed(1)}%`;
  }

  // Head pose
  const p = d.pitch;
  $pitchVal.textContent   = (p >= 0 ? '+' : '') + p.toFixed(1) + '°';
  $pitchVal.style.color   = d.head_alert ? 'var(--danger)' : Math.abs(p) > 12 ? 'var(--warn)' : '';
  $headBlock.classList.toggle('alerted', d.head_alert);

  // Pitch needle: map ±PITCH_RANGE to 0-100% left
  const pos = Math.max(0, Math.min(100, ((p + PITCH_RANGE) / (PITCH_RANGE * 2)) * 100));
  $pitchNeedle.style.left = pos + '%';
  $pitchNeedle.style.background = d.head_alert ? 'var(--danger)' : Math.abs(p) > 12 ? 'var(--warn)' : 'var(--safe)';
}

// ── PERCLOS arc gauge ──────────────────────────────────────────────────────
// Arc: M 10 105 A 90 90 0 0 0 190 105 (counterclockwise = upward arc)
// Total arc length = π × 90 ≈ 282.74
// stroke-dasharray controls how much of the arc is filled.
function updatePerclosGauge(perclos, isAlert) {
  const ratio = Math.min(perclos / ARC_MAX, 1.0);
  const len   = ratio * ARC_LEN;

  $arcFill.setAttribute('stroke-dasharray', `${len.toFixed(2)} ${ARC_LEN}`);

  const color = isAlert
    ? 'var(--danger)'
    : perclos >= 0.20 ? 'var(--warn)' : 'var(--safe)';

  $arcFill.style.stroke    = color;
  $arcPctText.textContent  = Math.round(perclos * 100) + '%';
  $arcPctText.style.fill   = color;
}

// ── Eye card ───────────────────────────────────────────────────────────────
function renderEye(card, probEl, prob) {
  if (prob == null) {
    card.className = 'eye-card';
    probEl.textContent = '—';
    return;
  }
  const closed = prob > 0.5;
  card.className = 'eye-card ' + (closed ? 'closed' : 'open');
  probEl.textContent = (prob * 100).toFixed(1) + '%';
}

// ── Yawn pips ──────────────────────────────────────────────────────────────
function updatePips(count, isAlert) {
  [$pip0, $pip1].forEach((pip, i) => {
    pip.className = 'pip';
    if (i < count) pip.classList.add(isAlert ? 'alerted' : 'active');
  });
}

// ── Badge / Banner helpers ─────────────────────────────────────────────────
function setBadge(cls, text) {
  $badge.className    = 'status-badge ' + cls;
  $badgeText.textContent = text;
}

function setBanner(cls, text) {
  $banner.className      = 'alert-banner ' + cls;
  $alertText.textContent = text;
  $bannerIcon.textContent = cls === 'danger'  ? '⚠'
                          : cls === 'warning' ? '⚡'
                          : '◈';
}

// ── Reset all metrics ──────────────────────────────────────────────────────
function resetMetrics() {
  updatePerclosGauge(0, false);
  $perclosFramesLabel.textContent = '0 frames — warming up (need 30)';
  $perclosBlock.classList.remove('alerted');

  renderEye($eyeLeftCard,  $eyeLeftProb,  null);
  renderEye($eyeRightCard, $eyeRightProb, null);
  $consecLabel.textContent = 'consec. closed: 0 frames';

  $yawnCount.textContent = '0';
  $yawnCount.style.color = '';
  $yawnProbLbl.textContent = 'Probability: —';
  $yawnBlock.classList.remove('alerted');
  updatePips(0, false);

  $pitchVal.textContent  = '+0.0°';
  $pitchVal.style.color  = '';
  $pitchNeedle.style.left = '50%';
  $pitchNeedle.style.background = 'var(--safe)';
  $headBlock.classList.remove('alerted');

  $fpsVal.textContent     = '—';
  $latencyVal.textContent = '—';
}

// ── Web Audio API alerts (mirrors alert_system.py) ─────────────────────────
// warning → single beep at 880 Hz
// danger  → repeating beep at 1200 Hz every 2 seconds
let _audioCtx   = null;
let _alertTimer = null;

function _ctx() {
  if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return _audioCtx;
}

function _beep(freq, dur) {
  const ctx  = _ctx();
  const osc  = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.type = 'sine';
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0, ctx.currentTime);
  gain.gain.linearRampToValueAtTime(0.30, ctx.currentTime + 0.04);
  gain.gain.linearRampToValueAtTime(0,    ctx.currentTime + dur - 0.04);
  osc.start(ctx.currentTime);
  osc.stop(ctx.currentTime + dur);
}

function triggerAudio(lvl) {
  stopAudioAlert();
  if (lvl === 'warning') {
    _beep(880, 0.4);
  } else if (lvl === 'danger') {
    _beep(1200, 0.6);
    _alertTimer = setInterval(() => _beep(1200, 0.6), 2000);
  }
}

function stopAudioAlert() {
  if (_alertTimer) { clearInterval(_alertTimer); _alertTimer = null; }
  prevLevel = '';
}
