"""
Panel Interview Orchestrator — manages multi-interviewer sessions with voice switching.

Each panelist gets their own Gemini Live session with a unique voice.
The orchestrator handles turn rotation, context passing between panelists,
and coordinating the interview flow.

Architecture:
- PanelConfig: defines the panel composition
- Panelist: individual interviewer config (name, voice, role, focus)
- PanelOrchestrator: manages the interview lifecycle
  - One active Gemini session at a time (voice switching requires new session)
  - Conversation history shared across all panelists via context injection
  - Smart rotation: picks next panelist based on conversation flow
"""

import asyncio
import logging
from dataclasses import dataclass, field

from google.genai import types
from gemini_live import GeminiLive

logger = logging.getLogger(__name__)


# ─── Panel Configuration ──────────────────────────────────────────────

@dataclass
class Panelist:
    """One interviewer on the panel."""
    name: str
    voice: str
    role: str
    focus: str
    style: str
    emoji: str = "👤"


# Pre-built panel configurations
PANEL_PRESETS = {
    "standard": [
        Panelist(
            name="Aisha", voice="Aoede", role="HR Lead",
            focus="culture fit, teamwork, and motivation",
            style="warm, friendly, and encouraging",
            emoji="👩‍💼",
        ),
        Panelist(
            name="Bilal", voice="Orus", role="Senior Engineer",
            focus="system design, coding, and technical problem-solving",
            style="direct, analytical, and precise",
            emoji="👨‍💻",
        ),
        Panelist(
            name="Sara", voice="Kore", role="Engineering Manager",
            focus="leadership, communication, and strategic thinking",
            style="thoughtful, strategic, and insightful",
            emoji="👩‍💼",
        ),
    ],
}


def build_panelist_prompt(
    panelist: Panelist,
    panel: list[Panelist],
    resume_text: str,
    conversation_history: str,
    is_first_turn: bool,
    language_directive: str = "",
) -> str:
    """Build system instruction for a specific panelist's turn."""

    panel_intro = ", ".join(
        f"{p.name} ({p.role})" for p in panel
    )

    context_block = ""
    if conversation_history:
        context_block = f"""
**CONVERSATION SO FAR:**
{conversation_history}

Continue naturally from where the conversation left off. Do NOT repeat questions already asked. React to the candidate's latest answer before asking your own question."""

    first_turn_instruction = ""
    if is_first_turn:
        first_turn_instruction = f"""
This is the START of the interview. Greet the candidate warmly, introduce yourself and briefly mention the other panelists ({panel_intro}). Then ask your first question."""
    else:
        first_turn_instruction = """
Another panelist just finished. Jump in naturally — react to what was said, then ask your question."""

    return f"""**Persona:**
You are {panelist.name}, a {panelist.role}. You are {panelist.style}.
Your focus area: {panelist.focus}.

**Panel:** This is a panel interview with: {panel_intro}.
{first_turn_instruction}

**Candidate's Resume:**
{resume_text}
{context_block}

**Rules:**
- Speak as {panelist.name} only. Stay in character.
- Ask ONE question related to your focus area.
- Keep your response to 3-4 sentences max (brief reaction + question).
- Be conversational and natural, not scripted.
- You can reference what other panelists asked or what the candidate said.

**Tool Instructions:**
- rate_answer: After the candidate answers, silently rate their response. Do NOT mention scoring.
- end_interview: Only call this if you are the LAST panelist and the interview should wrap up. Give brief verbal feedback first.
{language_directive}"""


# ─── Panel Orchestrator ───────────────────────────────────────────────

class PanelOrchestrator:
    """Manages a panel interview with multiple interviewers and voice switching."""

    def __init__(
        self,
        api_key: str,
        model_id: str,
        model_config: dict,
        resume_text: str,
        panel: list[Panelist],
        tools: list,
        tool_mapping: dict,
        language_directive: str = "",
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.model_config = model_config
        self.resume_text = resume_text
        self.panel = panel
        self.tools = tools
        self.tool_mapping = tool_mapping
        self.language_directive = language_directive

        # State
        self.conversation_history: list[dict] = []  # [{speaker, text}]
        self.current_panelist_index = 0
        self.turn_count = 0

    @property
    def current_panelist(self) -> Panelist:
        return self.panel[self.current_panelist_index]

    def next_panelist(self) -> Panelist:
        """Rotate to next panelist (round-robin)."""
        self.current_panelist_index = (self.current_panelist_index + 1) % len(self.panel)
        return self.current_panelist

    def get_conversation_summary(self) -> str:
        """Build conversation history string for context injection."""
        if not self.conversation_history:
            return ""
        lines = []
        for entry in self.conversation_history[-20:]:  # Last 20 exchanges
            lines.append(f"{entry['speaker']}: {entry['text']}")
        return "\n".join(lines)

    def add_to_history(self, speaker: str, text: str):
        """Add an exchange to conversation history."""
        self.conversation_history.append({"speaker": speaker, "text": text})

    def create_panelist_session(self, panelist: Panelist) -> GeminiLive:
        """Create a new GeminiLive session for a specific panelist."""
        prompt = build_panelist_prompt(
            panelist=panelist,
            panel=self.panel,
            resume_text=self.resume_text,
            conversation_history=self.get_conversation_summary(),
            is_first_turn=(self.turn_count == 0),
            language_directive=self.language_directive,
        )

        return GeminiLive(
            api_key=self.api_key,
            model=self.model_id,
            system_instruction=prompt,
            input_sample_rate=16000,
            voice_name=panelist.voice,
            tools=self.tools,
            tool_mapping=self.tool_mapping,
            enable_affective=self.model_config.get("supports_affective", True),
            enable_proactive=self.model_config.get("supports_proactive", True),
            thinking_mode=self.model_config.get("thinking_mode", "budget"),
        )

    async def run_panelist_turn(
        self,
        panelist: Panelist,
        audio_input_queue: asyncio.Queue,
        video_input_queue: asyncio.Queue,
        audio_output_callback,
        audio_interrupt_callback,
        event_callback,
    ) -> bool:
        """Run one panelist's turn. Returns True if interview should end."""

        session = self.create_panelist_session(panelist)
        self.turn_count += 1
        interview_ended = False
        panelist_transcript: list[str] = []

        # Notify frontend which panelist is speaking
        await event_callback({
            "type": "panelist_switch",
            "name": panelist.name,
            "role": panelist.role,
            "voice": panelist.voice,
            "emoji": panelist.emoji,
            "index": self.panel.index(panelist),
        })

        trigger_text = (
            f"You are {panelist.name}. Greet the candidate and ask your first question."
            if self.turn_count == 1
            else f"You are {panelist.name}. React to what was said and ask your question."
        )

        async for event in session.start_session(
            audio_input_queue=audio_input_queue,
            video_input_queue=video_input_queue,
            audio_output_callback=audio_output_callback,
            audio_interrupt_callback=audio_interrupt_callback,
        ):
            if not event:
                continue

            event_type = event.get("type")

            # Collect panelist's transcript for history
            if event_type == "model_transcript":
                panelist_transcript.append(event["text"])

            # Check if this panelist ended the interview
            if event_type == "tool_call" and event.get("name") == "end_interview":
                interview_ended = True

            # Forward event to main handler (adds panelist name)
            event["panelist"] = panelist.name
            await event_callback(event)

            if event_type == "turn_complete":
                break

            if event_type == "error" and not event.get("recoverable", False):
                break

        # Save panelist's contribution to history
        full_text = " ".join(panelist_transcript)
        if full_text.strip():
            self.add_to_history(panelist.name, full_text.strip())

        return interview_ended
