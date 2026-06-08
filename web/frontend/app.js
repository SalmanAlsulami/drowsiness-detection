'use strict';

// ── DOM ────────────────────────────────────────────────────────────────────
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
const $scanLine      = document.getElementById('scan-line');
const $noFace        = document.getElementById('no-face');
const $recDot        = document.getElementById('rec-dot');
const $camStatusText = document.getElementById('cam-status-text');
const $sessionTime   = document.getElementById('session-time');

// Face state
const $faceIcon  = document.getElementById('face-state-icon');
const $faceTxt   = document.getElementById('face-state-text');

// PERCLOS
const $gaugeFill        = document.getElementById('gauge-fill');
const $arcPctText       = document.getElementById('arc-pct-text');
const $perclosFramesLbl = document.getElementById('perclos-frames-label');
const $perclosBlock     = document.getElementById('perclos-block');

// Eyes
const $eyeLeftCard  = document.getElementById('eye-left-card');
const $eyeLeftProb  = document.getElementById('eye-left-prob');
const $eyeRightCard = document.getElementById('eye-right-card');
const $eyeRightProb = document.getElementById('eye-right-prob');
const $consecLabel  = document.getElementById('consec-label');

// Yawn
const $yawnCount  = document.getElementById('yawn-count');
const $yawnBlock  = document.getElementById('yawn-block');
const $yawnProbLbl= document.getElementById('yawn-prob-label');
const $pip0       = document.getElementById('pip-0');
const $pip1       = document.getElementById('pip-1');

// Head
const $pitchVal    = document.getElementById('pitch-val');
const $headBlock   = document.getElementById('head-block');
const $pitchNeedle = document.getElementById('pitch-needle');

// Stats
const $fpsVal     = document.getElementById('fps-val');
const $latencyVal = document.getElementById('latency-val');

// ── Constants ──────────────────────────────────────────────────────────────
const GAUGE_C    = 301.59;  // 2π × 48  (circle r=48, viewBox 120×120)
const GAUGE_MAX  = 0.70;
const PITCH_RANGE= 40;      // ±40° on track

const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;

// ── State ──────────────────────────────────────────────────────────────────
let ws = null, stream = null, sendLoop = null;
let frameCount = 0, lastFpsTick = 0, lastSentAt = 0;
let prevLevel = '';
let sessionStart = null, sessionTimer = null;

// ── Session timer ──────────────────────────────────────────────────────────
function startSessionTimer() {
  sessionStart = Date.now();
  sessionTimer = setInterval(() => {
    const s = Math.floor((Date.now() - sessionStart) / 1000);
    $sessionTime.textContent =
      String(Math.floor(s / 3600)).padStart(2,'0') + ':' +
      String(Math.floor((s % 3600) / 60)).padStart(2,'0') + ':' +
      String(s % 60).padStart(2,'0');
  }, 1000);
}
function stopSessionTimer() {
  clearInterval(sessionTimer);
  $sessionTime.textContent = '—';
}

// ── Buttons ────────────────────────────────────────────────────────────────
$startBtn.addEventListener('click', () => {
  $startBtn.classList.add('pressed');
  $startBtn.addEventListener('animationend', () => $startBtn.classList.remove('pressed'), { once: true });
  startInspection();
});
$stopBtn.addEventListener('click', stopInspection);

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
    $scanLine.classList.add('active');
    setBadge('safe', 'SAFE');
    setBanner('safe', '✓ DRIVER IS ALERT');
    $cameraFrame.className = 'camera-frame state-safe';
    startSessionTimer();
    lastFpsTick = Date.now(); frameCount = 0;
    sendLoop = setInterval(captureAndSend, 100);
  };

  ws.onmessage = ev => {
    $latencyVal.textContent = (Date.now() - lastSentAt) + 'ms';
    renderResult(JSON.parse(ev.data));
  };

  ws.onerror = () => setBadge('badge-standby', 'ERROR');
  ws.onclose = () => { if (stream) stopInspection(); };
}

function stopInspection() {
  clearInterval(sendLoop); sendLoop = null;
  if (ws)     { ws.close(); ws = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  $video.srcObject = null;

  $startBtn.classList.remove('hidden');
  $stopBtn.classList.add('hidden');
  $idleOverlay.classList.remove('hidden');
  $noFace.classList.add('hidden');
  $recDot.classList.remove('live');
  $camStatusText.textContent = 'OFFLINE';
  $scanLine.classList.remove('active');
  $cameraFrame.className = 'camera-frame';

  setBadge('badge-standby', 'STANDBY');
  setBanner('standby', 'SYSTEM READY — PRESS INITIATE MONITORING TO BEGIN');
  stopAudioAlert();
  stopSessionTimer();
  resetMetrics();
  prevLevel = '';
}

// ── Capture ────────────────────────────────────────────────────────────────
function captureAndSend() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !stream) return;
  const ctx = $canvas.getContext('2d');
  ctx.drawImage($video, 0, 0, $canvas.width, $canvas.height);
  lastSentAt = Date.now();
  ws.send($canvas.toDataURL('image/jpeg', 0.82));
  frameCount++;
  const now = Date.now();
  if (now - lastFpsTick >= 1000) {
    $fpsVal.textContent = String(frameCount);
    frameCount = 0; lastFpsTick = now;
  }
}

// ── Render ─────────────────────────────────────────────────────────────────
function renderResult(d) {
  const lvl = d.alert_level;

  if (lvl === 'danger') {
    setBadge('danger', 'DANGER');
    setBanner('danger', '⚠ DANGER: DRIVER IS DROWSY');
    $cameraFrame.className = 'camera-frame state-danger';
    setFaceState('danger');
  } else if (lvl === 'warning') {
    setBadge('warning', 'WARNING');
    setBanner('warning', '⚡ WARNING: DROWSINESS SIGNS DETECTED');
    $cameraFrame.className = 'camera-frame state-warning';
    setFaceState('drowsy');
  } else {
    setBadge('safe', 'SAFE');
    setBanner('safe', '✓ DRIVER IS ALERT');
    $cameraFrame.className = 'camera-frame state-safe';
    setFaceState('alert');
  }

  if (lvl !== prevLevel) { triggerAudio(lvl); prevLevel = lvl; }

  $noFace.classList.toggle('hidden', d.face_found);

  // PERCLOS
  updateGauge(d.perclos, d.perclos_alert);
  $perclosFramesLbl.textContent = d.perclos_frames >= 30
    ? `${d.perclos_frames} frames tracked`
    : `${d.perclos_frames} frames — warming up`;
  $perclosBlock.classList.toggle('alerted', d.perclos_alert);

  // Eyes
  renderEye($eyeLeftCard,  $eyeLeftProb,  d.eye_left_prob);
  renderEye($eyeRightCard, $eyeRightProb, d.eye_right_prob);
  $consecLabel.textContent = `closed: ${d.consec_closed} frames`;

  // Yawn
  $yawnCount.textContent = d.yawn_events;
  $yawnCount.style.color = d.yawn_alert ? 'var(--danger)' : d.yawn_events > 0 ? 'var(--warn)' : '';
  $yawnBlock.classList.toggle('alerted', d.yawn_alert);
  $yawnBlock.classList.toggle('warned',  d.yawn_events > 0 && !d.yawn_alert);
  updatePips(d.yawn_events, d.yawn_alert);
  if (d.yawn_prob != null)
    $yawnProbLbl.textContent = `Prob: ${(d.yawn_prob * 100).toFixed(1)}%`;

  // Head
  const p = d.pitch;
  $pitchVal.textContent = (p >= 0 ? '+' : '') + p.toFixed(1) + '°';
  $pitchVal.style.color = d.head_alert ? 'var(--danger)' : Math.abs(p) > 12 ? 'var(--warn)' : '';
  $headBlock.classList.toggle('alerted', d.head_alert);

  const pos = Math.max(0, Math.min(100, ((p + PITCH_RANGE) / (PITCH_RANGE * 2)) * 100));
  $pitchNeedle.style.left = pos + '%';
  const nc = d.head_alert ? 'var(--danger)' : Math.abs(p) > 12 ? 'var(--warn)' : 'var(--safe)';
  $pitchNeedle.style.background = nc;
  const ns = d.head_alert ? 'rgba(255,59,48,.65)' : Math.abs(p) > 12 ? 'rgba(255,149,0,.55)' : 'rgba(48,209,88,.55)';
  $pitchNeedle.style.boxShadow = `0 0 8px ${ns}`;
}

// ── Gauge (circular) ───────────────────────────────────────────────────────
function updateGauge(perclos, isAlert) {
  const ratio  = Math.min(perclos / GAUGE_MAX, 1.0);
  const offset = (GAUGE_C * (1 - ratio)).toFixed(2);
  $gaugeFill.style.strokeDashoffset = offset;

  const color = isAlert         ? 'var(--danger)'
    : perclos >= 0.20           ? 'var(--warn)'
    :                             'var(--safe)';
  $gaugeFill.style.stroke    = color;
  $arcPctText.textContent    = Math.round(perclos * 100) + '%';
  $arcPctText.style.fill     = isAlert ? 'var(--danger)' : perclos >= 0.20 ? 'var(--warn)' : 'var(--text-hi)';
}

// ── Face state icon ────────────────────────────────────────────────────────
function setFaceState(state) {
  $faceIcon.className = 'face-icon ' + state;
  if (state === 'danger') {
    $faceTxt.textContent = 'DANGER';
    $faceTxt.style.color = 'var(--danger)';
  } else if (state === 'drowsy') {
    $faceTxt.textContent = 'DROWSY';
    $faceTxt.style.color = 'var(--warn)';
  } else {
    $faceTxt.textContent = 'ALERT';
    $faceTxt.style.color = 'var(--safe)';
  }
}

// ── Eye card ───────────────────────────────────────────────────────────────
function renderEye(card, probEl, prob) {
  if (prob == null) {
    card.className = 'eye-card';
    probEl.textContent = '—';
    return;
  }
  card.className = prob > 0.5 ? 'eye-card closed' : 'eye-card open';
  probEl.textContent = (prob * 100).toFixed(1) + '%';
}

// ── Pips ───────────────────────────────────────────────────────────────────
function updatePips(count, isAlert) {
  [$pip0, $pip1].forEach((pip, i) => {
    pip.className = 'pip';
    if (i < count) pip.classList.add(isAlert ? 'alerted' : 'active');
  });
}

// ── Badge / Banner ─────────────────────────────────────────────────────────
function setBadge(cls, text) {
  $badge.className   = 'badge ' + cls;
  $badgeText.textContent = text;
}
function setBanner(cls, text) {
  $banner.className      = 'banner banner-' + cls;
  $alertText.textContent = text;
  $bannerIcon.textContent = cls === 'danger' ? '⚠' : cls === 'warning' ? '⚡' : cls === 'safe' ? '✓' : '○';
}

// ── Reset ──────────────────────────────────────────────────────────────────
function resetMetrics() {
  updateGauge(0, false);
  $perclosFramesLbl.textContent = '0 frames — warming up';
  $perclosBlock.classList.remove('alerted');

  renderEye($eyeLeftCard,  $eyeLeftProb,  null);
  renderEye($eyeRightCard, $eyeRightProb, null);
  $consecLabel.textContent = 'closed: 0 frames';

  $yawnCount.textContent = '0'; $yawnCount.style.color = '';
  $yawnProbLbl.textContent = 'Prob: —';
  $yawnBlock.classList.remove('alerted', 'warned');
  updatePips(0, false);

  $pitchVal.textContent = '+0.0°'; $pitchVal.style.color = '';
  $pitchNeedle.style.left = '50%';
  $pitchNeedle.style.background = 'var(--safe)';
  $pitchNeedle.style.boxShadow  = '0 0 8px rgba(48,209,88,.55)';
  $headBlock.classList.remove('alerted');

  $fpsVal.textContent = '—'; $latencyVal.textContent = '—';
  setFaceState('alert');
}

// ── Web Audio ──────────────────────────────────────────────────────────────
let _audioCtx = null, _alertTimer = null;
function _ctx() {
  if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return _audioCtx;
}
function _beep(freq, dur) {
  const ctx = _ctx(), osc = ctx.createOscillator(), g = ctx.createGain();
  osc.connect(g); g.connect(ctx.destination);
  osc.type = 'sine'; osc.frequency.value = freq;
  g.gain.setValueAtTime(0, ctx.currentTime);
  g.gain.linearRampToValueAtTime(0.28, ctx.currentTime + 0.04);
  g.gain.linearRampToValueAtTime(0,    ctx.currentTime + dur - 0.04);
  osc.start(ctx.currentTime); osc.stop(ctx.currentTime + dur);
}
function triggerAudio(lvl) {
  stopAudioAlert();
  if      (lvl === 'warning') { _beep(880, 0.4); }
  else if (lvl === 'danger')  { _beep(1200, 0.6); _alertTimer = setInterval(() => _beep(1200, 0.6), 2000); }
}
function stopAudioAlert() {
  if (_alertTimer) { clearInterval(_alertTimer); _alertTimer = null; }
}
