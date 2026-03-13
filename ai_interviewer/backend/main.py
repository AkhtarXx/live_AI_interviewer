"""
AI Interviewer Backend — FastAPI server with WebSocket endpoint.

Architecture matches Google's official reference implementation:
- Queue-based input routing (audio, video, text)
- Raw binary WebSocket messages for audio (low latency)
- JSON text messages for video frames and events
- CORS middleware for cross-origin support
- Tools: end_interview, rate_answer, lookup_topic + Google Search grounding
- Session resumption + context window compression for unlimited interviews
- Post-interview report generation via standard Gemini API
"""

import asyncio
import base64
import json
import logging
import os

from pathlib import Path
from dotenv import load_dotenv

# Search for .env in multiple locations
_backend_dir = Path(__file__).resolve().parent
_project_root = _backend_dir.parent
_workspace_root = _project_root.parent
load_dotenv(dotenv_path=_workspace_root / ".env")
load_dotenv(dotenv_path=_project_root / ".env")
load_dotenv(dotenv_path=_backend_dir / ".env")
load_dotenv()

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from google.genai import types

from gemini_live import GeminiLive
from app.services.resume_extractor import extract_text_from_pdf
from app.services.report_generator import generate_interview_report
from app.services.panel_orchestrator import PanelOrchestrator, PANEL_PRESETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Available Live API models
MODELS = {
    "gemini-2.5-flash": {
        "id": "gemini-2.5-flash-native-audio-preview-12-2025",
        "label": "Gemini 2.5 Flash (Full Features)",
        "supports_affective": True,
        "supports_proactive": True,
        "thinking_mode": "budget",  # uses thinkingBudget
    },
    "gemini-3.1-flash": {
        "id": "gemini-3.1-flash-live-preview",
        "label": "Gemini 3.1 Flash (Lower Latency)",
        "supports_affective": False,
        "supports_proactive": False,
        "thinking_mode": "level",  # uses thinkingLevel
    },
}
DEFAULT_MODEL_KEY = os.getenv("MODEL_KEY", "gemini-2.5-flash")
# Backward compat: if MODEL env var is set directly, use it
MODEL = os.getenv("MODEL", MODELS[DEFAULT_MODEL_KEY]["id"])

app = FastAPI(title="AI Interviewer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="../frontend"), name="static")


@app.get("/")
async def root() -> HTMLResponse:
    """Serves the main frontend page."""
    with open("../frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)) -> dict:
    """Extracts text from an uploaded PDF resume."""
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are currently supported."}

    contents = await file.read()
    try:
        text_content = extract_text_from_pdf(contents)
        return {"message": "Success", "resume_text": text_content}
    except Exception as e:
        logger.error("Failed to process PDF: %s", e)
        return {"error": f"Failed to process PDF: {str(e)}"}


def build_system_instruction(resume_text: str) -> str:
    """Constructs the AI interviewer system prompt.
    
    Structure follows Google's Live API best practices:
    1. Agent persona
    2. Conversational rules (one-time elements + conversational loop)
    3. Tool invocation instructions
    4. Guardrails
    """
    return f"""**Persona:**
You are Aisha, a warm senior technical interviewer. Friendly but professional. Speak naturally with casual warmth. React genuinely. Use the candidate's first name. You can see them via video.

**Rules:**
1. Greet warmly, introduce yourself, make brief small talk.
2. Ask about their most recent role or an exciting project from the resume.
3. Weave between behavioral and technical questions based on their answers. Adapt difficulty — go harder if they nail it, ease up if they struggle. Probe vague answers deeper.
4. When you have enough signal, give brief verbal feedback and call end_interview.

**Resume:**
{resume_text}

**Tools (call silently, never mention to candidate):**
- rate_answer: Call after EVERY answer with topic, score (1-10), reasoning.
- lookup_topic: Research unfamiliar topics the candidate mentions.
- end_interview: Call when done, after giving verbal feedback.

**Guardrails:**
- One question at a time. Keep responses concise. Never mention scoring.
"""


# ─── Tool Declarations ────────────────────────────────────────────────

END_INTERVIEW_TOOL = types.FunctionDeclaration(
    name="end_interview",
    description="End the interview session and trigger report generation. **Invocation Condition:** Invoke this tool *only after* you have finished asking all questions, given brief verbal feedback to the candidate, and are ready to close the session. Do not invoke mid-conversation. SILENT EXECUTION — do not narrate calling this tool.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "overall_impression": types.Schema(
                type="STRING",
                description="Your overall impression of the candidate in 2-3 sentences",
            ),
            "recommendation": types.Schema(
                type="STRING",
                description="One of: strong_hire, hire, maybe, no_hire",
                enum=["strong_hire", "hire", "maybe", "no_hire"],
            ),
        },
        required=["overall_impression", "recommendation"],
    ),
)

RATE_ANSWER_TOOL = types.FunctionDeclaration(
    name="rate_answer",
    description="Silently record a score for the candidate's answer. **Invocation Condition:** Invoke this tool *immediately after* the candidate finishes answering each question. You must call this for every answer without exception. Do not mention scoring to the candidate. SILENT EXECUTION — do not narrate calling this tool.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "question_topic": types.Schema(
                type="STRING",
                description="Brief topic of the question (e.g. 'system design', 'React hooks')",
            ),
            "score": types.Schema(
                type="INTEGER",
                description="Score from 1-10 for the answer quality",
            ),
            "reasoning": types.Schema(
                type="STRING",
                description="Brief reasoning for the score in 1-2 sentences",
            ),
        },
        required=["question_topic", "score", "reasoning"],
    ),
)

LOOKUP_TOPIC_TOOL = types.FunctionDeclaration(
    name="lookup_topic",
    description="Research an unfamiliar topic using Google Search grounding. **Invocation Condition:** Invoke this tool when the candidate mentions a technology, company, framework, or concept you are not deeply familiar with, so you can ask better follow-up questions. SILENT EXECUTION — do not narrate calling this tool.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "topic": types.Schema(
                type="STRING",
                description="The topic to research",
            ),
            "reason": types.Schema(
                type="STRING",
                description="Why you need to look this up",
            ),
        },
        required=["topic", "reason"],
    ),
)


# ─── WebSocket Interview Endpoint ─────────────────────────────────────

@app.websocket("/ws/interview")
async def websocket_interview(websocket: WebSocket) -> None:
    """WebSocket endpoint for the live interview session."""
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    if not GEMINI_API_KEY:
        await websocket.close(reason="GEMINI_API_KEY not configured.")
        return

    # Wait for initial config with resume text
    initial_msg = await websocket.receive_text()
    config_data = json.loads(initial_msg)
    resume_text = config_data.get("resume_text", "No resume provided.")
    language = config_data.get("language", "en")
    selected_model_key = config_data.get("model", DEFAULT_MODEL_KEY)

    # Resolve model config
    model_config = MODELS.get(selected_model_key, MODELS[DEFAULT_MODEL_KEY])
    model_id = model_config["id"]
    logger.info("Selected model: %s (%s)", selected_model_key, model_id)

    instruction = build_system_instruction(resume_text)

    # Append language directive for non-English interviews
    if language != "en":
        language_names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "hi": "Hindi",
            "zh": "Mandarin Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "pt": "Portuguese",
            "ar": "Arabic",
        }
        language_name = language_names.get(language, "English")
        instruction += f"\n\nLANGUAGE: RESPOND IN {language_name.upper()}. YOU MUST RESPOND UNMISTAKABLY IN {language_name.upper()}. Greet the candidate in {language_name}."

    # Session state — capped to prevent unbounded memory growth in long interviews
    # Context window compression handles the Gemini side; this caps our local storage
    MAX_TRANSCRIPT_LINES = 500
    MAX_THOUGHT_TEXTS = 50
    transcript_lines: list[dict] = []
    thought_texts: list[str] = []
    answer_ratings: list[dict] = []
    interview_ended = asyncio.Event()
    end_interview_data: dict = {}

    # ─── Tool Handlers ────────────────────────────────────────────

    def handle_end_interview(overall_impression: str, recommendation: str) -> str:
        """Called by the model when it decides the interview is done."""
        end_interview_data["overall_impression"] = overall_impression
        end_interview_data["recommendation"] = recommendation
        interview_ended.set()
        logger.info("Model called end_interview: %s", recommendation)
        return "Interview ended. Report will be generated."

    def handle_rate_answer(question_topic: str, score: int, reasoning: str) -> str:
        """Silently records per-question ratings during the interview."""
        rating = {
            "question_topic": question_topic,
            "score": score,
            "reasoning": reasoning,
        }
        answer_ratings.append(rating)
        logger.info("Answer rated: %s = %d/10", question_topic, score)
        return "Rating recorded."

    def handle_lookup_topic(topic: str, reason: str) -> str:
        """Returns instruction to use Google Search grounding for the topic."""
        logger.info("Lookup topic: %s (reason: %s)", topic, reason)
        return f"Use your Google Search grounding capability to find current information about: {topic}. Reason: {reason}"

    # ─── Queues & Callbacks ───────────────────────────────────────

    audio_input_queue: asyncio.Queue = asyncio.Queue()
    video_input_queue: asyncio.Queue = asyncio.Queue()

    audio_output_count = 0

    async def audio_output_callback(data: bytes) -> None:
        nonlocal audio_output_count
        audio_output_count += 1
        if audio_output_count <= 5 or audio_output_count % 100 == 0:
            logger.info("Audio OUTPUT #%d: %d bytes to browser", audio_output_count, len(data))
        try:
            await websocket.send_bytes(data)
        except Exception as e:
            logger.error("Failed to send audio to browser: %s", e)

    # Track what Aisha was saying when interrupted (for context recovery)
    last_model_transcript_chunks: list[str] = []

    async def audio_interrupt_callback() -> None:
        """Called when user interrupts (barge-in). Gemini automatically stops
        generating and listens. We track what Aisha was saying for context."""
        interrupted_text = ' '.join(last_model_transcript_chunks)
        logger.info("User interrupted (barge-in). Aisha was saying: '%s...'",
                     interrupted_text[:80] if interrupted_text else "(nothing yet)")
        # Clear the tracking buffer for next turn
        last_model_transcript_chunks.clear()

    # ─── Build Gemini Client with all tools ───────────────────────

    tools_list = [
        types.Tool(function_declarations=[
            END_INTERVIEW_TOOL,
            RATE_ANSWER_TOOL,
            LOOKUP_TOPIC_TOOL,
        ]),
        types.Tool(google_search=types.GoogleSearch()),
    ]

    voice_name = os.environ.get("AISHA_VOICE", "Aoede")

    gemini_client = GeminiLive(
        api_key=GEMINI_API_KEY,
        model=model_id,
        system_instruction=instruction,
        input_sample_rate=16000,
        voice_name=voice_name,
        tools=tools_list,
        tool_mapping={
            "end_interview": handle_end_interview,
            "rate_answer": handle_rate_answer,
            "lookup_topic": handle_lookup_topic,
        },
        enable_affective=model_config["supports_affective"],
        enable_proactive=model_config["supports_proactive"],
        thinking_mode=model_config["thinking_mode"],
    )

    # ─── Client Receiver ──────────────────────────────────────────

    audio_chunk_count = 0
    video_frame_count = 0

    async def receive_from_client() -> None:
        """Routes incoming WebSocket messages to the appropriate queue."""
        nonlocal audio_chunk_count, video_frame_count
        try:
            while True:
                message = await websocket.receive()
                if message.get("bytes"):
                    audio_chunk_count += 1
                    if audio_chunk_count <= 5 or audio_chunk_count % 100 == 0:
                        logger.info("Audio chunk #%d received: %d bytes", audio_chunk_count, len(message["bytes"]))
                    await audio_input_queue.put(message["bytes"])
                elif message.get("text"):
                    text = message["text"]
                    try:
                        payload = json.loads(text)
                        if isinstance(payload, dict) and payload.get("type") == "image":
                            video_frame_count += 1
                            if video_frame_count <= 3 or video_frame_count % 30 == 0:
                                logger.info("Video frame #%d received", video_frame_count)
                            image_data = base64.b64decode(payload["data"])
                            await video_input_queue.put(image_data)
                            continue
                    except json.JSONDecodeError:
                        pass
        except WebSocketDisconnect:
            logger.info("Client disconnected. Audio chunks: %d, Video frames: %d", audio_chunk_count, video_frame_count)
        except Exception as e:
            logger.error("Error receiving from client: %s", e)

    receive_task = asyncio.create_task(receive_from_client())

    # ─── Session Runner with Reconnection Support ─────────────────

    async def run_session(resumption_handle: str | None = None) -> str | None:
        """Runs the Gemini Live session. Returns resumption token if go_away received."""
        async for event in gemini_client.start_session(
            audio_input_queue=audio_input_queue,
            video_input_queue=video_input_queue,
            audio_output_callback=audio_output_callback,
            audio_interrupt_callback=audio_interrupt_callback,
            resumption_handle=resumption_handle,
        ):
            if event:
                event_type = event.get("type")

                # Collect transcripts for the report (capped)
                if event_type == "user_transcript":
                    if len(transcript_lines) < MAX_TRANSCRIPT_LINES:
                        transcript_lines.append({
                            "speaker": "user",
                            "text": event["text"],
                        })
                elif event_type == "model_transcript":
                    transcript_lines.append({
                        "speaker": "model",
                        "text": event["text"],
                    })
                    if len(transcript_lines) > MAX_TRANSCRIPT_LINES:
                        # Keep last N lines, drop oldest
                        transcript_lines[:] = transcript_lines[-MAX_TRANSCRIPT_LINES:]
                    # Track what Aisha is saying for barge-in context recovery
                    last_model_transcript_chunks.append(event["text"])
                elif event_type == "thought":
                    if len(thought_texts) < MAX_THOUGHT_TEXTS:
                        thought_texts.append(event["text"])

                # Handle go_away — return token for reconnection
                # Use timeLeft to finish current turn gracefully before reconnecting
                if event_type == "go_away":
                    time_left = event.get("time_left")
                    logger.warning("GoAway received (time_left=%s), will reconnect...", time_left)
                    await websocket.send_json({
                        "type": "reconnecting",
                        "message": "Session reconnecting...",
                    })
                    # If we have time, wait briefly for current turn to finish
                    # Otherwise reconnect immediately
                    if time_left and time_left != "proactive":
                        try:
                            wait_ms = int(float(time_left))
                            # Wait up to 3s max for current turn, then reconnect
                            grace = min(wait_ms / 1000, 3.0)
                            if grace > 0.5:
                                logger.info("Waiting %.1fs for current turn before reconnect", grace)
                                await asyncio.sleep(grace)
                        except (ValueError, TypeError):
                            pass
                    return event.get("resumption_token")

                # Handle errors — forward to frontend
                if event_type == "error":
                    error_msg = event.get("error", "Unknown error")
                    logger.error("Session error: %s", error_msg)
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "message": error_msg,
                        })
                    except Exception:
                        pass
                    # If recoverable, continue; if not, check for resumption
                    if not event.get("recoverable", False):
                        token = event.get("resumption_token")
                        if token:
                            return token
                        break

                # Forward all other events to frontend
                if event_type not in ("go_away",):
                    try:
                        await websocket.send_json(event)
                    except Exception:
                        break

                # Clear model transcript buffer on turn complete (normal end of speech)
                if event_type == "turn_complete":
                    last_model_transcript_chunks.clear()

                # Check if interview was ended by the model
                if interview_ended.is_set():
                    break

        return None

    # ─── Main Session Loop with Reconnection ──────────────────────

    try:
        resumption_handle = None
        max_reconnects = 5
        session_start_time = None

        for attempt in range(max_reconnects + 1):
            session_start_time = asyncio.get_event_loop().time()

            token = await run_session(resumption_handle=resumption_handle)

            if interview_ended.is_set():
                break

            if token and attempt < max_reconnects:
                # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s
                delay = min(0.5 * (2 ** attempt), 8.0)
                logger.info("Reconnecting with resumption token (attempt %d/%d, delay %.1fs)...",
                            attempt + 1, max_reconnects, delay)
                resumption_handle = token

                # Flush stale data from queues before reconnecting
                flushed_audio = 0
                while not audio_input_queue.empty():
                    try:
                        audio_input_queue.get_nowait()
                        flushed_audio += 1
                    except asyncio.QueueEmpty:
                        break
                flushed_video = 0
                while not video_input_queue.empty():
                    try:
                        video_input_queue.get_nowait()
                        flushed_video += 1
                    except asyncio.QueueEmpty:
                        break
                if flushed_audio or flushed_video:
                    logger.info("Flushed stale queue data: %d audio, %d video", flushed_audio, flushed_video)

                await asyncio.sleep(delay)
                continue
            else:
                break

        # Generate report if the model ended the interview
        if interview_ended.is_set():
            logger.info("Generating interview report...")
            await websocket.send_json({
                "type": "interview_ending",
                "message": "Generating your interview report...",
            })

            report = await generate_interview_report(
                api_key=GEMINI_API_KEY,
                resume_text=resume_text,
                transcript_lines=transcript_lines,
                answer_ratings=answer_ratings,
                thought_summaries=thought_texts,
            )

            # Merge model's own impression
            report["model_impression"] = end_interview_data.get("overall_impression", "")
            report["model_recommendation"] = end_interview_data.get("recommendation", "")

            await websocket.send_json({
                "type": "interview_report",
                "report": report,
            })
            logger.info("Report sent to client.")

    except Exception as e:
        logger.error("Session error: %s", e)
    finally:
        receive_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


# ─── Panel Interview WebSocket Endpoint ────────────────────────────────

@app.websocket("/ws/panel-interview")
async def websocket_panel_interview(websocket: WebSocket) -> None:
    """WebSocket endpoint for panel interview with multiple interviewers and voice switching."""
    await websocket.accept()
    logger.info("Panel interview WebSocket accepted.")

    if not GEMINI_API_KEY:
        await websocket.close(reason="GEMINI_API_KEY not configured.")
        return

    # Wait for initial config
    initial_msg = await websocket.receive_text()
    config_data = json.loads(initial_msg)
    resume_text = config_data.get("resume_text", "No resume provided.")
    language = config_data.get("language", "en")
    selected_model_key = config_data.get("model", DEFAULT_MODEL_KEY)
    panel_preset = config_data.get("panel", "standard")

    model_config = MODELS.get(selected_model_key, MODELS[DEFAULT_MODEL_KEY])
    model_id = model_config["id"]
    panel = PANEL_PRESETS.get(panel_preset, PANEL_PRESETS["standard"])

    logger.info("Panel interview: model=%s, panel=%s (%d panelists)",
                model_id, panel_preset, len(panel))

    # Language directive
    language_directive = ""
    if language != "en":
        language_names = {
            "en": "English", "es": "Spanish", "fr": "French", "de": "German",
            "hi": "Hindi", "zh": "Mandarin Chinese", "ja": "Japanese",
            "ko": "Korean", "pt": "Portuguese", "ar": "Arabic",
        }
        lang = language_names.get(language, "English")
        language_directive = f"\n\nLANGUAGE: RESPOND IN {lang.upper()}. YOU MUST RESPOND UNMISTAKABLY IN {lang.upper()}."

    # Session state
    transcript_lines: list[dict] = []
    thought_texts: list[str] = []
    answer_ratings: list[dict] = []
    interview_ended = asyncio.Event()
    end_interview_data: dict = {}

    def handle_end_interview(overall_impression: str, recommendation: str) -> str:
        end_interview_data["overall_impression"] = overall_impression
        end_interview_data["recommendation"] = recommendation
        interview_ended.set()
        return "Interview ended."

    def handle_rate_answer(question_topic: str, score: int, reasoning: str) -> str:
        answer_ratings.append({"question_topic": question_topic, "score": score, "reasoning": reasoning})
        return "Rating recorded."

    def handle_lookup_topic(topic: str, reason: str) -> str:
        return f"Use Google Search grounding for: {topic}. Reason: {reason}"

    tools_list = [
        types.Tool(function_declarations=[END_INTERVIEW_TOOL, RATE_ANSWER_TOOL, LOOKUP_TOPIC_TOOL]),
        types.Tool(google_search=types.GoogleSearch()),
    ]
    tool_mapping = {
        "end_interview": handle_end_interview,
        "rate_answer": handle_rate_answer,
        "lookup_topic": handle_lookup_topic,
    }

    # Queues — shared across all panelist sessions
    audio_input_queue: asyncio.Queue = asyncio.Queue()
    video_input_queue: asyncio.Queue = asyncio.Queue()

    async def audio_output_callback(data: bytes) -> None:
        try:
            await websocket.send_bytes(data)
        except Exception as e:
            logger.error("Failed to send audio: %s", e)

    async def audio_interrupt_callback() -> None:
        logger.info("Panel: user interrupted")

    # Event callback — forwards events to browser with panelist info
    async def event_callback(event: dict) -> None:
        event_type = event.get("type")

        if event_type == "user_transcript":
            transcript_lines.append({"speaker": "user", "text": event["text"]})
        elif event_type == "model_transcript":
            panelist_name = event.get("panelist", "Interviewer")
            transcript_lines.append({"speaker": panelist_name, "text": event["text"]})
        elif event_type == "thought":
            if len(thought_texts) < 50:
                thought_texts.append(event["text"])

        try:
            await websocket.send_json(event)
        except Exception:
            pass

    # Client receiver
    async def receive_from_client() -> None:
        try:
            while True:
                message = await websocket.receive()
                if message.get("bytes"):
                    await audio_input_queue.put(message["bytes"])
                elif message.get("text"):
                    try:
                        payload = json.loads(message["text"])
                        if isinstance(payload, dict) and payload.get("type") == "image":
                            image_data = base64.b64decode(payload["data"])
                            await video_input_queue.put(image_data)
                    except json.JSONDecodeError:
                        pass
        except WebSocketDisconnect:
            logger.info("Panel: client disconnected")
        except Exception as e:
            logger.error("Panel receive error: %s", e)

    receive_task = asyncio.create_task(receive_from_client())

    # Create orchestrator
    orchestrator = PanelOrchestrator(
        api_key=GEMINI_API_KEY,
        model_id=model_id,
        model_config=model_config,
        resume_text=resume_text,
        panel=panel,
        tools=tools_list,
        tool_mapping=tool_mapping,
        language_directive=language_directive,
    )

    # Send panel info to frontend
    await websocket.send_json({
        "type": "panel_info",
        "panelists": [
            {"name": p.name, "role": p.role, "emoji": p.emoji, "voice": p.voice}
            for p in panel
        ],
    })

    # ─── Main Panel Loop ──────────────────────────────────────────
    try:
        # First turn: first panelist greets
        panelist = orchestrator.current_panelist
        ended = await orchestrator.run_panelist_turn(
            panelist=panelist,
            audio_input_queue=audio_input_queue,
            video_input_queue=video_input_queue,
            audio_output_callback=audio_output_callback,
            audio_interrupt_callback=audio_interrupt_callback,
            event_callback=event_callback,
        )

        # Subsequent turns: wait for candidate audio, then next panelist responds
        while not ended and not interview_ended.is_set():
            # Wait for candidate to speak (audio chunks arriving = candidate talking)
            try:
                # Wait up to 60s for candidate to start speaking
                chunk = await asyncio.wait_for(audio_input_queue.get(), timeout=60.0)
                await audio_input_queue.put(chunk)  # Put it back for the session to consume
            except asyncio.TimeoutError:
                logger.info("Panel: candidate silent for 60s, ending")
                break

            # Let candidate speak — collect audio for a few seconds
            # The Gemini session will handle VAD and turn detection
            panelist = orchestrator.next_panelist()

            # Add candidate's contribution to history from transcripts
            recent_user = [t for t in transcript_lines if t["speaker"] == "user"]
            if recent_user:
                last_user = recent_user[-1]["text"]
                orchestrator.add_to_history("Candidate", last_user)

            ended = await orchestrator.run_panelist_turn(
                panelist=panelist,
                audio_input_queue=audio_input_queue,
                video_input_queue=video_input_queue,
                audio_output_callback=audio_output_callback,
                audio_interrupt_callback=audio_interrupt_callback,
                event_callback=event_callback,
            )

        # Generate report
        if interview_ended.is_set():
            logger.info("Panel interview ended, generating report...")
            await websocket.send_json({"type": "interview_ending", "message": "Generating panel report..."})

            report = await generate_interview_report(
                api_key=GEMINI_API_KEY,
                resume_text=resume_text,
                transcript_lines=transcript_lines,
                answer_ratings=answer_ratings,
                thought_summaries=thought_texts,
            )
            report["model_impression"] = end_interview_data.get("overall_impression", "")
            report["model_recommendation"] = end_interview_data.get("recommendation", "")
            report["panel_mode"] = True
            report["panelists"] = [{"name": p.name, "role": p.role} for p in panel]

            await websocket.send_json({"type": "interview_report", "report": report})

    except Exception as e:
        logger.error("Panel session error: %s", e)
    finally:
        receive_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
