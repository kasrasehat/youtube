"""Agent implementations for transcript extraction, correction, translation, and dialogue conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from utils.llm_client import invoke_llm
from utils.prompt_loader import load_prompt
from utils.youtube import download_video, fetch_transcript_text, sanitize_filename


@dataclass
class VideoTranscriptAgent:
    """Agent that downloads the video file and saves its transcript."""

    video_dir: Path
    transcript_dir: Path

    def run(self, url: str) -> Tuple[Path, Path, str]:
        """Download the video and transcript, returning paths and transcript text."""
        video_path, title = download_video(url, self.video_dir)
        transcript_text = fetch_transcript_text(url)

        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_filename = f"{sanitize_filename(title)}.txt"
        transcript_path = self.transcript_dir / transcript_filename
        transcript_path.write_text(transcript_text, encoding="utf-8")

        return video_path, transcript_path, transcript_text


@dataclass
class TranscriptCorrectionAgent:
    """Agent that fixes transcript wording while preserving all key points."""

    prompt_name: str = "system_modify"

    def run(self, transcript: str, model_name: str) -> str:
        """Correct grammar, spelling, and wording issues without removing content."""
        system_prompt = load_prompt(self.prompt_name)
        payload = {
            "task": "correct_transcript",
            "instructions": "Fix errors but keep meaning and all important points.",
            "transcript": transcript,
        }
        return invoke_llm(system_prompt=system_prompt, user_payload=payload, model_name=model_name)


@dataclass
class TranslationAgent:
    """Agent that translates the transcript into Turkish (Istanbul usage)."""

    prompt_name: str = "system_translate"

    def run(self, transcript: str, model_name: str) -> str:
        """Translate a transcript into Turkish as spoken in Istanbul."""
        system_prompt = load_prompt(self.prompt_name)
        payload = {
            "task": "translate_transcript",
            "target_language": "Turkish (Istanbul)",
            "transcript": transcript,
        }
        return invoke_llm(system_prompt=system_prompt, user_payload=payload, model_name=model_name)


@dataclass
class DialogueAgent:
    """Agent that improves correctness and converts text into a dialogue style."""

    prompt_name: str = "system_dialogue"

    def run(self, transcript: str, model_name: str) -> str:
        """Convert text into a dialogue style with light expansion and summary."""
        system_prompt = load_prompt(self.prompt_name)
        payload = {
            "task": "dialogue_conversion",
            "transcript": transcript,
        }
        return invoke_llm(system_prompt=system_prompt, user_payload=payload, model_name=model_name)

