# AI Interviewer — Real-Time Multimodal Interview System

I built this to prepare for my own interviews — an AI that actually interviews you in real-time, adapts to your resume, and gives honest feedback. It turned out well enough that I decided to open-source it.

Upload your resume, and an AI interviewer conducts a personalized adaptive interview — seeing you via camera, hearing you via microphone, and generating a structured feedback report when done. Powered by Google's Gemini Live API.

This project is under active development. More features and improvements are coming.

## Features

- **Solo Interview**: A single AI interviewer (Aisha) conducts a full adaptive interview
- **Panel Interview**: 3 AI interviewers (Aisha, Bilal, Sara) with different voices, roles, and focus areas — simulating a real panel experience
- **Multimodal**: Sees the candidate via webcam, hears via microphone, reads their resume
- **Multi-Model**: Choose between Gemini 2.5 Flash (emotion-aware, proactive) or Gemini 3.1 Flash (lower latency)
- **Multi-Language**: Supports 10 languages — English, Spanish, French, German, Hindi, Mandarin, Japanese, Korean, Portuguese, Arabic
- **Real-Time Scoring**: Live topic coverage tracker and confidence meter during the interview
- **Screen Sharing**: Share your screen for coding/whiteboard questions
- **Structured Reports**: AI-generated interview feedback with per-question analysis, strengths, improvements, and recommendation
- **Session Resilience**: Auto-reconnection, context window compression, and session resumption for uninterrupted interviews
- **Barge-In Support**: Interrupt the interviewer naturally — the system handles it gracefully

## Tech Stack

- **Backend**: Python, FastAPI, WebSocket
- **Frontend**: Vanilla JS, Web Audio API (AudioWorklet)
- **AI**: Google Gemini Live API (native audio), Google Search grounding
- **PDF Parsing**: PyMuPDF

## Project Structure

```
ai_interviewer/
├── backend/
│   ├── main.py                       # FastAPI server, WebSocket endpoints (solo + panel)
│   ├── gemini_live.py                # GeminiLive wrapper — core session engine
│   └── app/services/
│       ├── resume_extractor.py       # PDF text extraction
│       ├── report_generator.py       # Post-interview report generation
│       └── panel_orchestrator.py     # Panel interview orchestrator (multi-voice)
├── frontend/
│   ├── index.html                    # 3 screens: upload → interview → report
│   ├── style.css                     # Glassmorphic dark theme
│   ├── app.js                        # MediaHandler + interview logic
│   └── pcm-processor.js             # AudioWorklet for low-latency PCM capture
├── Dockerfile
├── deploy.sh
├── requirements.txt
└── .env.example
```

## Quick Start

### Prerequisites

- Python 3.11+
- A [Gemini API key](https://aistudio.google.com/apikey) with the Generative Language API enabled

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-interviewer.git
cd ai-interviewer

# Create and activate virtual environment
python -m venv ai_interviewer/venv
source ai_interviewer/venv/bin/activate

# Install dependencies
pip install -r ai_interviewer/requirements.txt

# Set up environment variables
cp ai_interviewer/.env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Run

```bash
cd ai_interviewer/backend
uvicorn main:app --host localhost --port 8080
```

Open `http://localhost:8080` → upload a PDF resume → choose your model and language → start the interview.

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | — | Your Google Gemini API key |
| `MODEL_KEY` | No | `gemini-2.5-flash` | Model to use (`gemini-2.5-flash` or `gemini-3.1-flash`) |
| `AISHA_VOICE` | No | `Aoede` | Voice for the AI interviewer ([30 voices available](https://ai.google.dev/gemini-api/docs/live#voices)) |
| `PORT` | No | `8080` | Server port |


## Architecture

### Solo Mode
```
Browser (mic) → AudioWorklet → 16kHz PCM → WebSocket (binary) ─┐
Browser (cam) → Canvas JPEG → WebSocket (JSON)                  ├→ FastAPI → GeminiLive → Gemini Live API
Browser (screen) → getDisplayMedia → JPEG → WebSocket (JSON)   ─┘         ↓
                                                                    Audio + Events → WebSocket → Browser
```

### Panel Mode
```
Same browser input → FastAPI /ws/panel-interview → PanelOrchestrator
                                                        ↓
                                          ┌─────────────┼─────────────┐
                                          ↓             ↓             ↓
                                     Aisha (HR)    Bilal (Tech)   Sara (Mgmt)
                                     Aoede voice   Orus voice     Kore voice
                                          ↓             ↓             ↓
                                     Separate Gemini sessions, shared context
```

Each panelist gets their own Gemini Live session with a unique voice. The orchestrator handles turn rotation and injects conversation history so later panelists can reference earlier answers.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Raw binary WebSocket for audio | Zero base64 overhead, lowest latency |
| AudioWorklet (not ScriptProcessor) | Runs on separate thread, glitch-free capture |
| Queue-based I/O | Decouples capture from session logic |
| 20ms audio chunks (640 bytes) | Optimal per Google's documentation |
| Separate playback AudioContext at 24kHz | Avoids mic capture conflicts |
| 1 FPS video, `MEDIA_RESOLUTION_LOW` | Saves tokens — audio already uses ~25/sec |
| Thinking OFF | Saves 2-5s latency per response |
| `silence_duration_ms=800` | Balances pause tolerance vs responsiveness |

## Multi-Model Support

| Model | Features | First Audio Latency |
|-------|----------|-------------------|
| Gemini 2.5 Flash | Emotion-aware dialog, proactive audio | ~2.6s |
| Gemini 3.1 Flash Live | Lower latency, better tool use | ~0.9s |

The backend auto-configures everything based on model selection — API version, affective/proactive features, thinking mode, and greeting method.

## Interview Tools

The AI interviewer uses three tools during the interview (invisible to the candidate):

- **rate_answer**: Silently scores each answer (topic, score 1-10, reasoning) — feeds the live scoring sidebar and final report
- **lookup_topic**: Researches unfamiliar topics via Google Search grounding for better follow-up questions
- **end_interview**: Ends the session and triggers report generation with overall impression and recommendation

## Connection Management

- **Session resumption**: Transparent reconnection using Gemini's resumption tokens
- **Context window compression**: Sliding window prevents session termination on long interviews
- **Proactive reconnect**: Reconnects at 9 minutes before the connection limit
- **GoAway handling**: Graceful reconnection with exponential backoff
- **Frontend auto-reconnect**: 3 retry attempts with audio buffering during reconnection

## Docker Deployment

```bash
cd ai_interviewer
docker build -t ai-interviewer .
docker run -p 8080:8080 -e GEMINI_API_KEY=your_key_here ai-interviewer
```

### Google Cloud Run

```bash
cd ai_interviewer
export GEMINI_API_KEY=your_key_here
./deploy.sh
```

## API Endpoints

| Endpoint | Type | Description |
|----------|------|-------------|
| `GET /` | HTTP | Serves the frontend |
| `POST /upload-resume` | HTTP | Extracts text from uploaded PDF |
| `/ws/interview` | WebSocket | Solo interview session |
| `/ws/panel-interview` | WebSocket | Panel interview session |

## Contributing

Contributions are welcome. Feel free to open issues or submit pull requests.

## License

MIT
