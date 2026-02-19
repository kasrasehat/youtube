"""Main entry point for downloading YouTube media and processing transcripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # Allow running this file directly (e.g., via debugger) without module mode.
    sys.path.insert(0, str(REPO_ROOT))

from utils.agents import (
    DialogueAgent,
    TranscriptCorrectionAgent,
    TranslationAgent,
    VideoTranscriptAgent,
)


def extract_transcript_passage(full_text: str, max_chars: int = 6000) -> str:
    """Extract a manageable transcript passage for downstream LLM processing."""
    compact = " ".join(full_text.split())
    if len(compact) <= max_chars:
        return compact
    trimmed = compact[:max_chars].rsplit(" ", 1)[0]
    return trimmed + "..."


def save_text(path: Path, text: str) -> None:
    """Write text content to disk, ensuring the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_text(path: Path) -> str:
    """Read text content from disk."""
    return path.read_text(encoding="utf-8")


def build_output_paths(stem: str, output_dir: Path) -> dict[str, Path]:
    """Build output file paths for each transcript processing stage."""
    return {
        "passage": output_dir / f"{stem}_passage.txt",
        "corrected": output_dir / f"{stem}_corrected.txt",
        "translated": output_dir / f"{stem}_tr_istanbul.txt",
        "dialogue": output_dir / f"{stem}_dialogue.txt",
    }


def main() -> None:
    """Run the multi-agent pipeline on a YouTube URL."""
    parser = argparse.ArgumentParser(description="YouTube video and transcript processor.")
    parser.add_argument(
        "--url",
        default="https://www.youtube.com/watch?v=TVUibwoVXZc",
        help="YouTube video URL to process.",
    )
    parser.add_argument(
        "--model",
        default="gpt5-low",
        help="LLM model name or alias (e.g., gpt-4o, gpt-4o-mini, gpt-5).",
    )
    parser.add_argument(
        "--passage-chars",
        type=int,
        default=600000,
        help="Maximum number of characters to keep from the transcript passage.",
    )
    args = parser.parse_args()

    video_dir = REPO_ROOT / "data" / "video"
    transcript_dir = REPO_ROOT / "data" / "transcript"
    output_dir = REPO_ROOT / "data" / "output"

    video_agent = VideoTranscriptAgent(video_dir=video_dir, transcript_dir=transcript_dir)
    video_path, transcript_path, transcript_text = video_agent.run(args.url)
    passage = extract_transcript_passage(transcript_text, max_chars=args.passage_chars)

    output_paths = build_output_paths(Path(transcript_path).stem, output_dir)
    if output_paths["passage"].exists():
        passage = load_text(output_paths["passage"])
    else:
        save_text(output_paths["passage"], passage)

    if output_paths["corrected"].exists():
        corrected = load_text(output_paths["corrected"])
    else:
        correction_agent = TranscriptCorrectionAgent()
        corrected = correction_agent.run(passage, model_name=args.model)
        if type(corrected) == list:
            save_text(output_paths["corrected"], corrected[0]["text"])
            corrected = corrected[0]["text"]
        else:
            save_text(output_paths["corrected"], corrected)

    if output_paths["translated"].exists():
        translated = load_text(output_paths["translated"])
    else:
        translation_agent = TranslationAgent()
        translated = translation_agent.run(corrected, model_name=args.model)
        if type(translated) == list:
            save_text(output_paths["translated"], translated[0]["text"])
            translated = translated[0]["text"]
        else:
            save_text(output_paths["translated"], translated)

    if output_paths["dialogue"].exists():
        dialogue = load_text(output_paths["dialogue"])
    else:
        dialogue_agent = DialogueAgent()
        dialogue = dialogue_agent.run(translated, model_name=args.model)
        if type(dialogue) == list:
            save_text(output_paths["dialogue"], dialogue[0]["text"])
            dialogue = dialogue[0]["text"]
        else:
            save_text(output_paths["dialogue"], dialogue)
    

    print("Video saved to:", video_path)
    print("Transcript saved to:", transcript_path)
    print("Passage saved to:", output_paths["passage"])
    print("Corrected transcript saved to:", output_paths["corrected"])
    print("Translated transcript saved to:", output_paths["translated"])
    print("Dialogue transcript saved to:", output_paths["dialogue"])


if __name__ == "__main__":
    main()
