"""
Interview Report Generator — produces a structured feedback report
after the live interview ends, using the collected transcripts and
real-time answer ratings from the rate_answer tool.
"""

import json
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

REPORT_MODEL = "gemini-2.0-flash"


def _build_report_prompt(resume_text: str, transcript: str, ratings_text: str, thought_summaries: list[str] | None = None) -> str:
    """Builds the prompt for generating the interview report."""
    ratings_section = ""
    if ratings_text:
        ratings_section = f"""
REAL-TIME ANSWER RATINGS (scored by the interviewer during the session):
{ratings_text}

Use these ratings to inform your per-question analysis. They reflect the interviewer's live assessment.
"""

    thought_section = ""
    if thought_summaries:
        thoughts_text = "\n".join(f"- {t}" for t in thought_summaries)
        thought_section = f"""
THOUGHT SUMMARIES (internal reasoning from the interviewer during the session):
{thoughts_text}

Include these thought summaries in the report's thought_process field.
"""

    return f"""You are an expert interview evaluator. Based on the interview transcript, resume, and real-time ratings below, generate a structured interview report.

CANDIDATE'S RESUME:
{resume_text}

INTERVIEW TRANSCRIPT:
{transcript}
{ratings_section}{thought_section}
Generate a JSON report with this exact structure:
{{
  "candidate_name": "extracted from resume",
  "overall_score": 1-10,
  "summary": "2-3 sentence overall assessment",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "improvements": ["area 1", "area 2"],
  "question_analysis": [
    {{
      "question": "the question asked",
      "answer_quality": "excellent/good/fair/poor",
      "score": 1-10,
      "notes": "brief evaluation"
    }}
  ],
  "thought_process": ["thought summary 1", "thought summary 2"],
  "recommendation": "strong_hire / hire / maybe / no_hire",
  "detailed_feedback": "3-4 sentences of constructive feedback for the candidate"
}}

Return ONLY valid JSON, no markdown formatting.
"""


async def generate_interview_report(
    api_key: str,
    resume_text: str,
    transcript_lines: list[dict],
    answer_ratings: list[dict] | None = None,
    thought_summaries: list[str] | None = None,
) -> dict:
    """
    Generates a structured interview report from transcripts and ratings.

    Args:
        api_key: Gemini API key.
        resume_text: The candidate's resume text.
        transcript_lines: List of {"speaker": "user"|"model", "text": "..."} dicts.
        answer_ratings: Optional list of {"question_topic", "score", "reasoning"} dicts.
        thought_summaries: Optional list of thought texts captured during the interview.

    Returns:
        dict: Structured interview report.
    """
    # Build transcript string
    transcript_str = "\n".join(
        f"{'Candidate' if t['speaker'] == 'user' else 'Interviewer'}: {t['text']}"
        for t in transcript_lines
    )

    if not transcript_str.strip():
        return {
            "summary": "No transcript was captured during the interview.",
            "overall_score": 0,
            "recommendation": "incomplete",
        }

    # Build ratings string
    ratings_text = ""
    if answer_ratings:
        ratings_text = "\n".join(
            f"- {r['question_topic']}: {r['score']}/10 — {r['reasoning']}"
            for r in answer_ratings
        )

    client = genai.Client(api_key=api_key)
    prompt = _build_report_prompt(resume_text, transcript_str, ratings_text, thought_summaries)

    try:
        response = await client.aio.models.generate_content(
            model=REPORT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        report = json.loads(response.text)
        logger.info("Interview report generated successfully.")
        return report

    except Exception as e:
        logger.error("Failed to generate report: %s", e)
        return {
            "summary": f"Report generation failed: {str(e)}",
            "overall_score": 0,
            "recommendation": "error",
        }
