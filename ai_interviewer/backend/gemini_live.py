"""
GeminiLive: Wrapper for the Gemini Live API session management.

Modeled after Google's official reference implementation at:
https://github.com/google-gemini/gemini-live-api-examples

Features enabled:
- v1alpha API version (required for native audio features)
- LiveConnectConfig with full typed configuration
- Affective dialog (emotion-aware responses)
- Proactive audio (model decides when to respond)
- Thinking budget (model reasons before answering)
- Input + Output audio transcription (live subtitles)
- Barge-in / interruption handling
- Queue-based send/receive architecture
- Tool call support (extensible)
- Session resumption (transparent reconnection)
- Context window compression (unlimited session length)
- VAD tuning (interview-optimized silence detection)
- Robust error handling (RESPONSE_REJECTED, malformed calls)
"""

import asyncio
import inspect
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiLive:
    """Manages the lifecycle of a Gemini Live API session."""

    def __init__(
        self,
        api_key: str,
        model: str,
        system_instruction: str,
        input_sample_rate: int = 16000,
        voice_name: str = "Puck",
        tools: list | None = None,
        tool_mapping: dict | None = None,
        enable_affective: bool = True,
        enable_proactive: bool = True,
        thinking_mode: str = "budget",
    ):
        self.model = model
        self.system_instruction = system_instruction
        self.input_sample_rate = input_sample_rate
        self.voice_name = voice_name
        self.tools = tools or []
        self.tool_mapping = tool_mapping or {}
        self.enable_affective = enable_affective
        self.enable_proactive = enable_proactive
        self.thinking_mode = thinking_mode

        # Session resumption token — stored across reconnections
        self.resumption_token: str | None = None

        # v1alpha required for 2.5 (affective/proactive). 3.1 uses default API.
        if self.enable_affective or self.enable_proactive:
            self.client = genai.Client(
                api_key=api_key,
                http_options={"api_version": "v1alpha"},
            )
        else:
            self.client = genai.Client(api_key=api_key)

    def _build_config(self, resumption_handle: str | None = None) -> types.LiveConnectConfig:
        """Builds the typed LiveConnectConfig with features enabled per model capabilities.
        
        Gemini 2.5 Flash: affective dialog, proactive audio, thinkingBudget
        Gemini 3.1 Flash: lower latency, better tool use, thinkingLevel
        """
        # Thinking OFF for lowest latency — voice interview doesn't need deep reasoning.
        # Community reports 2-5s extra latency from thinking. Interview questions
        # don't require chain-of-thought — the model is smart enough without it.
        if self.thinking_mode == "level":
            thinking = types.ThinkingConfig(thinking_level="minimal")
        else:
            thinking = types.ThinkingConfig(thinking_budget=0)

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice_name,
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=self.system_instruction)]
            ),
            # Live transcription for both directions
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            # Thinking config (budget for 2.5, level for 3.1)
            thinking_config=thinking,
            # Context window compression — prevents session termination on long interviews
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            # Session resumption — transparent reconnection on drops
            session_resumption=types.SessionResumptionConfig(
                handle=resumption_handle,
            ),
            # VAD tuning — balanced for interview: allow brief pauses for thinking
            # but don't wait too long (1500ms was adding 1.5s to every response).
            # 800ms silence = good balance between pause tolerance and responsiveness.
            # prefix_padding_ms=100 filters coughs but doesn't delay real speech detection.
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                    prefix_padding_ms=100,
                    silence_duration_ms=800,
                ),
            ),
            # Reduce video token consumption — audio already uses ~25 tokens/sec,
            # LOW resolution prevents video from burning through context window
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
            tools=self.tools,
        )

        # Native audio features — only available on Gemini 2.5 (require v1alpha)
        if self.enable_affective:
            config.enable_affective_dialog = True
        if self.enable_proactive:
            config.proactivity = types.ProactivityConfig(proactive_audio=True)

        logger.info("Config built: model=%s, affective=%s, proactive=%s, thinking=%s",
                     self.model, self.enable_affective, self.enable_proactive, self.thinking_mode)
        return config

    async def start_session(
        self,
        audio_input_queue: asyncio.Queue,
        video_input_queue: asyncio.Queue,
        audio_output_callback,
        audio_interrupt_callback=None,
        resumption_handle: str | None = None,
    ):
        """
        Starts the Gemini Live session, yielding events as they occur.
        Supports session resumption via resumption_handle parameter.

        Yields:
            dict: Event dicts with type and payload.
        """
        config = self._build_config(resumption_handle=resumption_handle)

        async with self.client.aio.live.connect(
            model=self.model, config=config
        ) as session:
            logger.info("Connected to Gemini Live API (v1alpha). Resumption=%s",
                        "yes" if resumption_handle else "new session")

            # Send initial text to trigger the model's greeting.
            # With proactive_audio=True the model waits for meaningful
            # input before speaking — this kick-starts the conversation.
            # Gemini 3.1: send_client_content only works for initial history seeding,
            # must use send_realtime_input(text=...) for conversation text.
            # Gemini 2.5: send_client_content works throughout the conversation.
            if not resumption_handle:
                if self.thinking_mode == "level":
                    # Gemini 3.1 Flash Live — use send_realtime_input
                    await session.send_realtime_input(
                        text="The candidate has joined. Begin the interview."
                    )
                else:
                    # Gemini 2.5 Flash — use send_client_content
                    await session.send_client_content(
                        turns=types.Content(
                            parts=[types.Part(text="The candidate has joined. Begin the interview.")]
                        )
                    )

            async def send_audio():
                """Drains audio queue and forwards PCM chunks to Gemini.
                Sends audioStreamEnd when audio pauses for >1s (Google best practice)."""
                audio_stream_active = False
                try:
                    while True:
                        try:
                            chunk = await asyncio.wait_for(audio_input_queue.get(), timeout=1.0)
                            if not audio_stream_active:
                                audio_stream_active = True
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=chunk,
                                    mime_type=f"audio/pcm;rate={self.input_sample_rate}",
                                )
                            )
                        except asyncio.TimeoutError:
                            # No audio for 1s — flush cached audio on server
                            if audio_stream_active:
                                audio_stream_active = False
                                try:
                                    await session.send_realtime_input(audio_stream_end=True)
                                    logger.debug("Sent audioStreamEnd (1s silence)")
                                except Exception:
                                    pass
                except asyncio.CancelledError:
                    pass

            async def send_video():
                """Drains video queue and forwards JPEG frames to Gemini."""
                try:
                    while True:
                        frame = await video_input_queue.get()
                        logger.debug("Sending video frame: %d bytes", len(frame))
                        await session.send_realtime_input(
                            video=types.Blob(
                                data=frame, mime_type="image/jpeg"
                            )
                        )
                except asyncio.CancelledError:
                    pass

            async def transcription_flush():
                """Periodic flush to prevent inputTranscription from stopping during long speech.
                Community-discovered workaround: send turnComplete=False every 15s.
                Note: For Gemini 3.1, send_client_content is only for initial history,
                so we skip this flush (3.1 may handle transcription differently)."""
                if self.thinking_mode == "level":
                    # Gemini 3.1 — skip flush, send_client_content not supported mid-session
                    return
                try:
                    while True:
                        await asyncio.sleep(15)
                        try:
                            await session.send_client_content(
                                turns=None, turn_complete=False
                            )
                            logger.debug("Transcription flush sent (15s interval)")
                        except Exception as e:
                            logger.debug("Transcription flush failed (non-critical): %s", e)
                except asyncio.CancelledError:
                    pass

            async def proactive_reconnect_timer():
                """Proactively trigger reconnection before the ~10 min connection limit.
                Community reports sessions die at 8-12 min. We reconnect at 9 min
                to avoid abrupt disconnects. Uses the stored resumption token."""
                try:
                    await asyncio.sleep(9 * 60)  # 9 minutes
                    if self.resumption_token:
                        logger.warning("Proactive reconnect: 9 min reached, triggering graceful reconnect")
                        await event_queue.put({
                            "type": "go_away",
                            "time_left": "proactive",
                            "resumption_token": self.resumption_token,
                        })
                    else:
                        logger.debug("Proactive reconnect: no resumption token available, skipping")
                except asyncio.CancelledError:
                    pass

            event_queue: asyncio.Queue = asyncio.Queue()

            audio_from_model_count = 0
            # Token usage tracking — docs say usageMetadata comes periodically
            last_token_usage = {"total": 0, "audio_tokens": 0, "text_tokens": 0}

            async def receive_loop():
                """Processes all responses from Gemini and routes them."""
                nonlocal audio_from_model_count
                audio_started_after_turn = True  # Start True so we don't emit on first audio before any turn_complete
                try:
                    while True:
                        async for response in session.receive():
                            # --- Token usage tracking ---
                            if response.usage_metadata:
                                usage = response.usage_metadata
                                last_token_usage["total"] = getattr(usage, 'total_token_count', 0) or 0
                                # Log periodically so we can monitor context window usage
                                if last_token_usage["total"] > 0:
                                    logger.info("Token usage: %d total tokens consumed", last_token_usage["total"])
                                    # Warn if approaching context limit (128K for 3.1, 32K for 2.5)
                                    if last_token_usage["total"] > 100000:
                                        logger.warning("HIGH TOKEN USAGE: %d tokens — approaching context limit!", last_token_usage["total"])
                                        await event_queue.put({
                                            "type": "token_warning",
                                            "total_tokens": last_token_usage["total"],
                                        })

                            # --- Session resumption token updates ---
                            if response.session_resumption_update:
                                update = response.session_resumption_update
                                if update.resumable and update.new_handle:
                                    self.resumption_token = update.new_handle
                                    logger.debug("Session resumption token updated.")

                            # --- GoAway: server is about to disconnect ---
                            if response.go_away is not None:
                                time_left = getattr(response.go_away, 'time_left', None)
                                logger.warning("GoAway received. Time left: %s", time_left)
                                await event_queue.put({
                                    "type": "go_away",
                                    "time_left": str(time_left) if time_left else None,
                                    "resumption_token": self.resumption_token,
                                })

                            server_content = response.server_content
                            tool_call = response.tool_call

                            if server_content:
                                # Forward audio chunks
                                if server_content.model_turn:
                                    for part in server_content.model_turn.parts:
                                        # Detect thought parts from Gemini's thinking process
                                        if getattr(part, 'thought', False) and part.text:
                                            await event_queue.put({
                                                "type": "thought",
                                                "text": part.text,
                                            })

                                        if part.inline_data:
                                            audio_from_model_count += 1
                                            if audio_from_model_count <= 5 or audio_from_model_count % 100 == 0:
                                                logger.info("Model audio chunk #%d: %d bytes (mime=%s)",
                                                            audio_from_model_count,
                                                            len(part.inline_data.data) if part.inline_data.data else 0,
                                                            part.inline_data.mime_type)
                                            # Emit audio_started on first audio chunk after turn_complete
                                            if not audio_started_after_turn:
                                                audio_started_after_turn = True
                                                await event_queue.put({"type": "audio_started"})

                                            if inspect.iscoroutinefunction(
                                                audio_output_callback
                                            ):
                                                await audio_output_callback(
                                                    part.inline_data.data
                                                )
                                            else:
                                                audio_output_callback(
                                                    part.inline_data.data
                                                )

                                # Forward user transcription
                                if (
                                    server_content.input_transcription
                                    and server_content.input_transcription.text
                                ):
                                    await event_queue.put({
                                        "type": "user_transcript",
                                        "text": server_content.input_transcription.text,
                                    })

                                # Forward model transcription
                                if (
                                    server_content.output_transcription
                                    and server_content.output_transcription.text
                                ):
                                    await event_queue.put({
                                        "type": "model_transcript",
                                        "text": server_content.output_transcription.text,
                                    })

                                # Turn complete signal
                                if server_content.turn_complete:
                                    audio_started_after_turn = False
                                    await event_queue.put({"type": "turn_complete"})

                                # Generation complete signal
                                if getattr(server_content, 'generation_complete', False):
                                    await event_queue.put({"type": "generation_complete"})

                                # Barge-in / interruption handling
                                if server_content.interrupted:
                                    if audio_interrupt_callback:
                                        if inspect.iscoroutinefunction(
                                            audio_interrupt_callback
                                        ):
                                            await audio_interrupt_callback()
                                        else:
                                            audio_interrupt_callback()
                                    await event_queue.put({"type": "interrupted"})

                            # Handle tool calls
                            if tool_call:
                                await self._handle_tool_calls(
                                    session, tool_call, event_queue
                                )

                except Exception as e:
                    error_str = str(e)
                    logger.error("Receive loop error: %s", error_str)

                    # Check if this is a recoverable error
                    if "RESPONSE_REJECTED" in error_str:
                        await event_queue.put({
                            "type": "error",
                            "error": error_str,
                            "recoverable": True,
                        })
                    else:
                        await event_queue.put({
                            "type": "error",
                            "error": error_str,
                            "recoverable": False,
                            "resumption_token": self.resumption_token,
                        })
                finally:
                    await event_queue.put(None)

            # Launch all tasks
            send_audio_task = asyncio.create_task(send_audio())
            send_video_task = asyncio.create_task(send_video())
            transcription_flush_task = asyncio.create_task(transcription_flush())
            proactive_reconnect_task = asyncio.create_task(proactive_reconnect_timer())
            receive_task = asyncio.create_task(receive_loop())

            try:
                while True:
                    event = await event_queue.get()
                    if event is None:
                        break
                    yield event
            finally:
                send_audio_task.cancel()
                send_video_task.cancel()
                transcription_flush_task.cancel()
                proactive_reconnect_task.cancel()
                receive_task.cancel()

    async def _handle_tool_calls(self, session, tool_call, event_queue):
        """Processes tool/function calls from Gemini and returns results.
        Handles malformed calls gracefully."""
        function_responses = []
        for fc in tool_call.function_calls:
            func_name = fc.name
            args = fc.args or {}

            if func_name in self.tool_mapping:
                try:
                    tool_func = self.tool_mapping[func_name]
                    if inspect.iscoroutinefunction(tool_func):
                        result = await tool_func(**args)
                    else:
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            None, lambda fn=func_name, a=args: self.tool_mapping[fn](**a)
                        )
                except TypeError as e:
                    # Malformed function call — bad args from model
                    logger.warning("Malformed tool call '%s': %s (args=%s)", func_name, e, args)
                    result = f"Error: malformed arguments — {e}"
                except Exception as e:
                    logger.error("Tool '%s' execution error: %s", func_name, e)
                    result = f"Error: {e}"

                function_responses.append(
                    types.FunctionResponse(
                        name=func_name,
                        id=fc.id,
                        response={"result": result},
                    )
                )
                await event_queue.put({
                    "type": "tool_call",
                    "name": func_name,
                    "args": args,
                    "result": result,
                })
            else:
                # Unknown tool — log and send error response so model doesn't hang
                logger.warning("Unknown tool called: '%s'", func_name)
                function_responses.append(
                    types.FunctionResponse(
                        name=func_name,
                        id=fc.id,
                        response={"result": f"Error: unknown tool '{func_name}'"},
                    )
                )

        if function_responses:
            try:
                await session.send_tool_response(
                    function_responses=function_responses
                )
            except Exception as e:
                logger.error("Failed to send tool response: %s", e)
                await event_queue.put({
                    "type": "error",
                    "error": f"Tool response send failed: {e}",
                    "recoverable": True,
                })
