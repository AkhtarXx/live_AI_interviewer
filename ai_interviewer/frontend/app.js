/**
 * AI Interviewer Frontend — app.js
 *
 * Architecture matches Google's official reference implementation:
 * - MediaHandler class for audio/video capture and playback
 * - AudioWorklet for low-latency PCM capture with downsampling to 16kHz
 * - Raw binary WebSocket messages for audio (no JSON+base64 overhead)
 * - JSON text messages for video frames and events
 * - Scheduled audio playback with nextStartTime queuing (gapless)
 * - Barge-in: stops playback immediately on interruption events
 * - Live transcript display for both user and AI
 * - Auto-reconnect with exponential backoff on connection drops
 * - Error event display in transcript panel
 */

// ─── MediaHandler Class ──────────────────────────────────────────────
class MediaHandler {
  constructor() {
    this.audioContext = null;
    this.playbackContext = null;
    this.mediaStream = null;
    this.audioWorkletNode = null;
    this.videoStream = null;
    this.videoInterval = null;
    this.nextStartTime = 0;
    this.scheduledSources = [];
    this.isRecording = false;
    this.videoCanvas = document.createElement("canvas");
    this.canvasCtx = this.videoCanvas.getContext("2d");
  }

  async initializeAudio() {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      await this.audioContext.audioWorklet.addModule("/static/pcm-processor.js");
    }
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }
  }

  async startAudio(onAudioData) {
    await this.initializeAudio();
    // Buffer for accumulating PCM16 bytes into optimal 20ms chunks (640 bytes at 16kHz)
    // Google docs: "Send audio in chunks of 20ms to 40ms"
    this._audioSendBuffer = new Uint8Array(0);
    const OPTIMAL_CHUNK_BYTES = 640; // 20ms at 16kHz PCM16 (320 samples × 2 bytes)

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.audioWorkletNode = new AudioWorkletNode(this.audioContext, "pcm-processor");

      this.audioWorkletNode.port.onmessage = (event) => {
        if (this.isRecording) {
          const downsampled = this.downsampleBuffer(
            event.data, this.audioContext.sampleRate, 16000
          );
          const pcm16 = this.convertFloat32ToInt16(downsampled);

          // Accumulate into buffer, send in optimal 20ms chunks
          const newBytes = new Uint8Array(pcm16);
          const combined = new Uint8Array(this._audioSendBuffer.length + newBytes.length);
          combined.set(this._audioSendBuffer);
          combined.set(newBytes, this._audioSendBuffer.length);
          this._audioSendBuffer = combined;

          while (this._audioSendBuffer.length >= OPTIMAL_CHUNK_BYTES) {
            const chunk = this._audioSendBuffer.slice(0, OPTIMAL_CHUNK_BYTES);
            this._audioSendBuffer = this._audioSendBuffer.slice(OPTIMAL_CHUNK_BYTES);
            onAudioData(chunk.buffer);
          }
        }
      };

      source.connect(this.audioWorkletNode);
      const muteGain = this.audioContext.createGain();
      muteGain.gain.value = 0;
      this.audioWorkletNode.connect(muteGain);
      muteGain.connect(this.audioContext.destination);
      this.isRecording = true;
    } catch (e) {
      console.error("Error starting audio:", e);
      throw e;
    }
  }

  stopAudio() {
    this.isRecording = false;
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
      this.mediaStream = null;
    }
    if (this.audioWorkletNode) {
      this.audioWorkletNode.disconnect();
      this.audioWorkletNode = null;
    }
  }

  async startVideo(videoElement, onFrame) {
    try {
      this.videoStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      videoElement.srcObject = this.videoStream;
      this.videoInterval = setInterval(() => {
        this.captureFrame(videoElement, onFrame);
      }, 1000);
    } catch (e) {
      console.error("Error starting video:", e);
      throw e;
    }
  }

  stopVideo(videoElement) {
    if (this.videoStream) {
      this.videoStream.getTracks().forEach((t) => t.stop());
      this.videoStream = null;
    }
    if (this.videoInterval) {
      clearInterval(this.videoInterval);
      this.videoInterval = null;
    }
    if (videoElement) videoElement.srcObject = null;
  }

  captureFrame(videoElement, onFrame) {
    if (!this.videoStream) return;
    this.videoCanvas.width = 640;
    this.videoCanvas.height = 480;
    this.canvasCtx.drawImage(videoElement, 0, 0, 640, 480);
    const base64 = this.videoCanvas.toDataURL("image/jpeg", 0.7).split(",")[1];
    onFrame(base64);
  }

  async ensurePlaybackContext() {
    // Create a separate playback context at 24kHz if needed
    // This avoids conflicts with the mic capture context (which may be at 48kHz)
    if (!this.playbackContext) {
      this.playbackContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 24000,
      });
      console.log("[ensurePlaybackContext] Created playback AudioContext at 24kHz, state:", this.playbackContext.state);
    }
    if (this.playbackContext.state === "suspended") {
      console.log("[ensurePlaybackContext] Resuming suspended playback context...");
      await this.playbackContext.resume();
      console.log("[ensurePlaybackContext] Playback context state after resume:", this.playbackContext.state);
    }
    return this.playbackContext;
  }

  async playAudio(arrayBuffer) {
    // Ensure we have a running playback context
    const ctx = await this.ensurePlaybackContext();

    const pcmData = new Int16Array(arrayBuffer);
    if (pcmData.length === 0) {
      console.warn("[playAudio] Empty PCM data");
      return;
    }
    const float32Data = new Float32Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      float32Data[i] = pcmData[i] / 32768.0;
    }

    const buffer = ctx.createBuffer(1, float32Data.length, 24000);
    buffer.getChannelData(0).set(float32Data);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const now = ctx.currentTime;
    this.nextStartTime = Math.max(now, this.nextStartTime);
    source.start(this.nextStartTime);
    this.nextStartTime += buffer.duration;

    this.scheduledSources.push(source);
    source.onended = () => {
      const idx = this.scheduledSources.indexOf(source);
      if (idx > -1) this.scheduledSources.splice(idx, 1);
    };
  }

  stopAudioPlayback() {
    this.scheduledSources.forEach((s) => {
      try { s.stop(); } catch (_) {}
    });
    this.scheduledSources = [];
    const ctx = this.playbackContext || this.audioContext;
    if (ctx) this.nextStartTime = ctx.currentTime;
  }

  downsampleBuffer(buffer, sampleRate, outSampleRate) {
    if (outSampleRate === sampleRate) return buffer;
    const ratio = sampleRate / outSampleRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    let offsetResult = 0, offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
      let accum = 0, count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i]; count++;
      }
      result[offsetResult] = accum / count;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }

  convertFloat32ToInt16(buffer) {
    const buf = new Int16Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) {
      buf[i] = Math.min(1, Math.max(-1, buffer[i])) * 0x7FFF;
    }
    return buf.buffer;
  }
}


// ─── Application Logic ──────────────────────────────────────────────
const uploadScreen = document.getElementById("upload-screen");
const interviewScreen = document.getElementById("interview-screen");
const resumeUpload = document.getElementById("resume-upload");
const startBtn = document.getElementById("start-btn");
const uploadStatus = document.getElementById("upload-status");
const connectionStatus = document.getElementById("connection-status");
const aiAvatar = document.querySelector(".ai-avatar");
const userVideo = document.getElementById("user-video");

let resumeText = "";
let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 3;
const mediaHandler = new MediaHandler();

// Audio buffer for data captured while WebSocket is disconnected/reconnecting
let audioBuffer = [];
let isReconnecting = false;
let waitingForAudio = false;

// Real-time scoring sidebar state
let scoringData = { ratings: [], averageScore: 0 };

// Screen share state
let screenShareStream = null;
let screenShareInterval = null;

// Speaker name → CSS class mapping (fixes the .aisha vs .ai mismatch)
const SPEAKER_CSS_MAP = { "you": "you", "aisha": "ai" };

// ─── 1. Resume Upload ───────────────────────────────────────────────
resumeUpload.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  if (file.type !== "application/pdf") {
    uploadStatus.style.color = "#e94560";
    uploadStatus.innerText = "Please upload a PDF file.";
    return;
  }

  uploadStatus.style.color = "#4CAF50";
  uploadStatus.innerText = "Extracting resume details...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/upload-resume", { method: "POST", body: formData });
    const result = await response.json();
    if (result.error) throw new Error(result.error);
    resumeText = result.resume_text;
    uploadStatus.innerText = `✓ Resume processed (${resumeText.length} chars). Ready to start!`;
    startBtn.disabled = false;
  } catch (err) {
    uploadStatus.style.color = "#e94560";
    uploadStatus.innerText = err.message;
  }
});

// ─── 2. Start Interview ──────────────────────────────────────────────
startBtn.addEventListener("click", async () => {
  // Pre-initialize playback context on user gesture (required by browser autoplay policy)
  await mediaHandler.ensurePlaybackContext();
  console.log("[START] Playback context pre-initialized on user click, state:", mediaHandler.playbackContext?.state);
  
  uploadScreen.style.display = "none";
  interviewScreen.style.display = "block";
  reconnectAttempts = 0;
  await startInterview();
});


// ─── 3. Main Interview Session with Auto-Reconnect ───────────────────
async function startInterview() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/ws/interview`);
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    connectionStatus.innerText = "● Connected";
    connectionStatus.classList.add("active");
    connectionStatus.classList.remove("reconnecting");
    reconnectAttempts = 0;

    // Send initial config with resume text, language, and model preference
    const language = document.getElementById("language-selector").value;
    const model = document.getElementById("model-selector").value;
    ws.send(JSON.stringify({ resume_text: resumeText, language, model }));

    // Flush any audio data buffered during reconnection
    if (audioBuffer.length > 0) {
      console.log(`Flushing ${audioBuffer.length} buffered audio chunks`);
      for (const chunk of audioBuffer) {
        ws.send(chunk);
      }
      audioBuffer = [];
    }
    isReconnecting = false;

    // Start audio capture — sends raw binary PCM over WebSocket
    // Buffers audio data while disconnected so it can be flushed on reconnect
    try {
      await mediaHandler.startAudio((pcmData) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(pcmData);
        } else if (isReconnecting) {
          audioBuffer.push(pcmData);
        }
      });
      console.log("Audio capture started successfully");
    } catch (e) {
      console.error("Failed to start audio capture:", e);
    }

    // Start video capture — sends JSON with base64 JPEG
    try {
      await mediaHandler.startVideo(userVideo, (base64Frame) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "image", data: base64Frame }));
        }
      });
      console.log("Video capture started successfully");
    } catch (e) {
      console.error("Failed to start video capture:", e);
    }
  };

  ws.onmessage = async (event) => {
    if (event.data instanceof ArrayBuffer) {
      // Debug: log audio output from server
      if (!window._audioOutCount) window._audioOutCount = 0;
      window._audioOutCount++;
      if (window._audioOutCount <= 5 || window._audioOutCount % 100 === 0) {
        console.log(`[AUDIO OUT] Chunk #${window._audioOutCount}: ${event.data.byteLength} bytes, PlaybackContext state: ${mediaHandler.playbackContext?.state || 'not created yet'}`);
      }
      // Hide thinking indicator on first audio chunk after turn_complete
      if (waitingForAudio) {
        waitingForAudio = false;
        const thinkingEl = document.querySelector(".thinking-indicator");
        if (thinkingEl) thinkingEl.style.display = "none";
      }
      aiAvatar.classList.add("speaking");
      await mediaHandler.playAudio(event.data);
      clearTimeout(window._speakingTimeout);
      window._speakingTimeout = setTimeout(() => {
        aiAvatar.classList.remove("speaking");
      }, 800);
    } else {
      try {
        const msg = JSON.parse(event.data);
        handleEvent(msg);
      } catch (_) {}
    }
  };

  ws.onclose = (event) => {
    connectionStatus.classList.remove("active");

    // Normal close (1000) or interview ended — don't reconnect
    if (event.code === 1000) {
      connectionStatus.innerText = "● Disconnected";
      return;
    }

    // Unexpected close — attempt auto-reconnect
    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
      reconnectAttempts++;
      isReconnecting = true;
      const delay = Math.min(1000 * Math.pow(2, reconnectAttempts - 1), 8000);
      connectionStatus.innerText = `● Reconnecting (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`;
      connectionStatus.classList.add("reconnecting");
      console.warn(`WebSocket closed unexpectedly (code ${event.code}). Reconnecting in ${delay}ms... (buffering audio)`);
      setTimeout(() => startInterview(), delay);
    } else {
      isReconnecting = false;
      audioBuffer = [];
      connectionStatus.innerText = "● Connection lost";
      showConnectionLostUI();
    }
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
  };
}

function showConnectionLostUI() {
  // Remove any existing overlay
  const existing = document.getElementById("connection-lost-overlay");
  if (existing) existing.remove();

  const overlay = document.createElement("div");
  overlay.id = "connection-lost-overlay";
  overlay.className = "connection-lost-overlay";
  overlay.innerHTML = `
    <div class="connection-lost-card">
      <div class="connection-lost-icon">⚠️</div>
      <h3>Connection Lost</h3>
      <p>Unable to reach the interview server after multiple attempts.</p>
      <div class="connection-lost-actions">
        <button id="retry-connection-btn" class="btn primary-btn">Retry Connection</button>
        <button id="end-interview-btn" class="btn danger-btn">End Interview</button>
      </div>
    </div>
  `;

  document.getElementById("interview-screen").appendChild(overlay);

  document.getElementById("retry-connection-btn").onclick = () => {
    overlay.remove();
    reconnectAttempts = 0;
    startInterview();
  };

  document.getElementById("end-interview-btn").onclick = () => {
    overlay.remove();
    cleanup();
    document.getElementById("interview-screen").style.display = "none";
    document.getElementById("upload-screen").style.display = "";
  };

  // Also add a system message in the transcript
  const transcriptBox = document.getElementById("transcript-box");
  if (transcriptBox) {
    appendSystemMessage(transcriptBox, "Connection lost. You can retry or end the interview.");
  }
}



// ─── 4. Handle Events from Backend ───────────────────────────────────
function handleEvent(msg) {
  const transcriptBox = document.getElementById("transcript-box");

  switch (msg.type) {
    case "user_transcript":
      if (transcriptBox) appendTranscript(transcriptBox, "You", msg.text);
      break;

    case "model_transcript":
      if (transcriptBox) appendTranscript(transcriptBox, "Aisha", msg.text);
      break;

    case "interrupted":
      // IMMEDIATE audio kill — stop all scheduled sources AND reset timeline
      // This prevents any queued audio from playing after the interrupt
      mediaHandler.stopAudioPlayback();
      // Also reset the playback context timeline to prevent stale scheduling
      if (mediaHandler.playbackContext) {
        mediaHandler.nextStartTime = mediaHandler.playbackContext.currentTime;
      }
      aiAvatar.classList.remove("speaking");
      // Brief "listening" visual pulse to show Aisha heard the interruption
      aiAvatar.classList.add("listening");
      setTimeout(() => aiAvatar.classList.remove("listening"), 600);
      waitingForAudio = false;
      { const thinkingEl = document.querySelector(".thinking-indicator");
        if (thinkingEl) thinkingEl.style.display = "none"; }
      break;

    case "turn_complete":
      aiAvatar.classList.remove("speaking");
      waitingForAudio = true;
      { const thinkingEl = document.querySelector(".thinking-indicator");
        if (thinkingEl) thinkingEl.style.display = "block"; }
      break;

    case "audio_started":
      waitingForAudio = false;
      { const thinkingEl = document.querySelector(".thinking-indicator");
        if (thinkingEl) thinkingEl.style.display = "none"; }
      break;

    case "reconnecting":
      connectionStatus.innerText = "● Reconnecting...";
      connectionStatus.classList.add("reconnecting");
      break;

    case "interview_ending":
      showReportLoading();
      break;

    case "interview_report":
      renderReport(msg.report);
      break;

    case "error":
      console.error("Server error:", msg.message || msg.error);
      if (transcriptBox) {
        appendSystemMessage(transcriptBox, msg.message || msg.error || "An error occurred");
      }
      break;

    case "token_warning":
      console.warn("Token usage high:", msg.total_tokens);
      if (transcriptBox) {
        appendSystemMessage(transcriptBox, "Interview is getting long — Aisha will wrap up soon.");
      }
      break;

    case "tool_call":
      // Silent — don't show tool calls to the user
      console.log("Tool called:", msg.name, msg.args);
      if (msg.name === "rate_answer") {
        try {
          const args = typeof msg.args === "string" ? JSON.parse(msg.args) : msg.args;
          scoringData.ratings.push({
            topic: args.question_topic,
            score: args.score,
            reasoning: args.reasoning
          });
          const total = scoringData.ratings.reduce((sum, r) => sum + r.score, 0);
          scoringData.averageScore = total / scoringData.ratings.length;
          renderSidebar();
        } catch (e) {
          console.error("Error processing rate_answer:", e);
        }
      }
      break;
  }
}

function renderSidebar() {
  const placeholder = document.getElementById("scoring-placeholder");
  const scoreBars = document.getElementById("score-bars");
  const averageEl = document.getElementById("average-score");
  const topicCountEl = document.getElementById("topic-count");

  // Hide placeholder when ratings exist
  if (placeholder) {
    placeholder.style.display = scoringData.ratings.length > 0 ? "none" : "block";
  }

  // Render score bars
  if (scoreBars) {
    const existingCount = scoreBars.children.length;
    scoreBars.innerHTML = "";
    scoringData.ratings.forEach((rating, index) => {
      const entry = document.createElement("div");
      entry.className = "score-entry";
      if (index >= existingCount) {
        entry.classList.add("slideIn");
      }
      const fillWidth = (rating.score / 10) * 100;
      entry.innerHTML = `
        <div class="score-label-row">
          <span class="score-topic">${rating.topic}</span>
          <span class="score-value">${rating.score}/10</span>
        </div>
        <div class="score-bar">
          <div class="score-bar-fill" style="width: ${fillWidth}%"></div>
        </div>
      `;
      scoreBars.appendChild(entry);
    });
  }

  // Update summary
  if (averageEl) {
    averageEl.textContent = `Average: ${scoringData.averageScore.toFixed(1)}`;
  }
  if (topicCountEl) {
    const count = scoringData.ratings.length;
    topicCountEl.textContent = `${count} topic${count !== 1 ? "s" : ""} assessed`;
  }
}


function appendTranscript(container, speaker, text) {
  const line = document.createElement("div");
  const cssClass = SPEAKER_CSS_MAP[speaker.toLowerCase()] || speaker.toLowerCase();
  line.className = `transcript-line ${cssClass}`;
  line.innerHTML = `<strong>${speaker}:</strong> ${text}`;
  container.appendChild(line);
  container.scrollTop = container.scrollHeight;
}

function appendSystemMessage(container, text) {
  const line = document.createElement("div");
  line.className = "transcript-line system";
  line.innerHTML = `<strong>System:</strong> ${text}`;
  container.appendChild(line);
  container.scrollTop = container.scrollHeight;
}

// ─── 5. Report Display ───────────────────────────────────────────────
function showReportLoading() {
  cleanup();
  document.getElementById("interview-screen").style.display = "none";
  document.getElementById("report-screen").style.display = "block";
  document.getElementById("report-loading").style.display = "flex";
  document.getElementById("report-content").innerHTML = "";
}

function renderReport(report) {
  document.getElementById("report-loading").style.display = "none";
  const container = document.getElementById("report-content");

  const score = report.overall_score || 0;
  const rec = report.recommendation || report.model_recommendation || "N/A";
  const recBadge = getRecBadge(rec);

  container.innerHTML = `
    <div class="report-header">
      <div class="score-circle">
        <span class="score-number">${score}</span>
        <span class="score-label">/ 10</span>
      </div>
      <div class="report-meta">
        <h3>${report.candidate_name || "Candidate"}</h3>
        <span class="rec-badge ${recBadge.cls}">${recBadge.text}</span>
      </div>
    </div>

    <div class="report-summary">
      <p>${report.summary || report.model_impression || ""}</p>
    </div>

    ${report.strengths ? `
    <div class="report-section">
      <h4>💪 Strengths</h4>
      <ul>${report.strengths.map(s => `<li>${s}</li>`).join("")}</ul>
    </div>` : ""}

    ${report.improvements ? `
    <div class="report-section">
      <h4>📈 Areas for Improvement</h4>
      <ul>${report.improvements.map(s => `<li>${s}</li>`).join("")}</ul>
    </div>` : ""}

    ${report.question_analysis ? `
    <div class="report-section">
      <h4>📝 Question Analysis</h4>
      <div class="question-list">
        ${report.question_analysis.map(q => `
          <div class="question-item">
            <div class="q-header">
              <span class="q-quality ${q.answer_quality}">${q.answer_quality}</span>
              ${q.score ? `<span class="q-score">${q.score}/10</span>` : ""}
              <span class="q-text">${q.question}</span>
            </div>
            <p class="q-notes">${q.notes}</p>
          </div>
        `).join("")}
      </div>
    </div>` : ""}

    ${report.thought_process && report.thought_process.length > 0 ? `
    <div class="report-section thought-process">
      <h4>🧠 Interviewer's Thought Process</h4>
      <ul>${report.thought_process.map(t => `<li>${t}</li>`).join("")}</ul>
    </div>` : ""}

    ${report.detailed_feedback ? `
    <div class="report-section feedback-section">
      <h4>💬 Detailed Feedback</h4>
      <p>${report.detailed_feedback}</p>
    </div>` : ""}
  `;
}

function getRecBadge(rec) {
  const map = {
    strong_hire: { text: "⭐ Strong Hire", cls: "badge-strong" },
    hire: { text: "✅ Hire", cls: "badge-hire" },
    maybe: { text: "🤔 Maybe", cls: "badge-maybe" },
    no_hire: { text: "❌ No Hire", cls: "badge-no" },
  };
  return map[rec] || { text: rec, cls: "badge-maybe" };
}

// ─── 6. Screen Sharing ───────────────────────────────────────────────
// Feature detection: hide Share Screen button if getDisplayMedia is unavailable
if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
  const shareBtn = document.getElementById("share-screen-btn");
  if (shareBtn) shareBtn.style.display = "none";
}

async function startScreenShare() {
  try {
    screenShareStream = await navigator.mediaDevices.getDisplayMedia({ video: true });
  } catch (e) {
    console.error("Screen share rejected or failed:", e);
    return;
  }

  const screenVideo = document.getElementById("screen-share-video");
  const preview = document.getElementById("screen-share-preview");
  const shareBtn = document.getElementById("share-screen-btn");
  const stopBtn = document.getElementById("stop-share-btn");

  screenVideo.srcObject = screenShareStream;
  preview.style.display = "block";
  shareBtn.style.display = "none";
  stopBtn.style.display = "";

  // Capture screen share frames at the same interval as camera (1000ms)
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  screenShareInterval = setInterval(() => {
    if (!screenShareStream || screenVideo.readyState < 2) return;
    canvas.width = 640;
    canvas.height = 480;
    ctx.drawImage(screenVideo, 0, 0, 640, 480);
    const base64 = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "image", data: base64 }));
    }
  }, 1000);

  // Handle browser-initiated stop (user clicks browser's "Stop sharing" button)
  const track = screenShareStream.getVideoTracks()[0];
  if (track) {
    track.onended = () => stopScreenShare();
  }
}

function stopScreenShare() {
  if (screenShareInterval) {
    clearInterval(screenShareInterval);
    screenShareInterval = null;
  }

  if (screenShareStream) {
    screenShareStream.getTracks().forEach((t) => t.stop());
    screenShareStream = null;
  }

  const screenVideo = document.getElementById("screen-share-video");
  const preview = document.getElementById("screen-share-preview");
  const shareBtn = document.getElementById("share-screen-btn");
  const stopBtn = document.getElementById("stop-share-btn");

  if (screenVideo) screenVideo.srcObject = null;
  if (preview) preview.style.display = "none";
  if (shareBtn) shareBtn.style.display = "";
  if (stopBtn) stopBtn.style.display = "none";
}

document.getElementById("share-screen-btn").addEventListener("click", startScreenShare);
document.getElementById("stop-share-btn").addEventListener("click", stopScreenShare);

// ─── 7. Cleanup ──────────────────────────────────────────────────────
function cleanup() {
  stopScreenShare();
  mediaHandler.stopAudio();
  mediaHandler.stopVideo(userVideo);
  mediaHandler.stopAudioPlayback();
  // Close playback context to release audio resources
  if (mediaHandler.playbackContext) {
    mediaHandler.playbackContext.close().catch(() => {});
    mediaHandler.playbackContext = null;
  }
}

document.getElementById("end-btn").addEventListener("click", () => {
  cleanup();
  if (ws) ws.close(1000); // Normal close — don't trigger reconnect
  document.getElementById("interview-screen").style.display = "none";
  document.getElementById("report-screen").style.display = "block";
  document.getElementById("report-content").innerHTML = `
    <div class="report-summary">
      <p>Interview ended manually. No report was generated because the AI interviewer didn't complete the session.</p>
      <p>For a full report, let Aisha finish the interview naturally.</p>
    </div>
  `;
});
