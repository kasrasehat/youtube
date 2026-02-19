"""YouTube video download and transcript retrieval utilities."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from common URL formats."""
    parsed = urlparse(url)
    if parsed.hostname in {"youtu.be"}:
        return parsed.path.lstrip("/")
    if parsed.hostname in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
        query = parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]
    raise ValueError(f"Unsupported YouTube URL format: {url}")


def sanitize_filename(text: str) -> str:
    """Make a filesystem-safe filename from a title string."""
    safe = "".join(ch if ch.isalnum() or ch in {" ", "-", "_"} else "_" for ch in text)
    safe = "_".join(part for part in safe.split() if part)
    return safe[:120] if safe else "youtube_video"


def download_video(url: str, output_dir: Path) -> tuple[Path, str]:
    """Download the highest resolution progressive MP4 and return path + title."""
    output_dir.mkdir(parents=True, exist_ok=True)
    yt = YouTube(url)
    stream = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
    )
    if stream is None:
        raise RuntimeError("No progressive MP4 stream found for this video.")
    video_path = Path(stream.download(output_path=str(output_dir)))
    return video_path, yt.title or "youtube_video"


def fetch_transcript_text(url: str) -> str:
    """Retrieve the transcript text for a YouTube video."""
    video_id = extract_video_id(url)
    # The library has had multiple API shapes across versions. Support both.
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        entries = YouTubeTranscriptApi.get_transcript(video_id)  # type: ignore[attr-defined]
        return " ".join(entry.get("text", "").strip() for entry in entries if entry.get("text"))

    # Fallback: newer/alternate API that returns objects with `.text`
    entries = YouTubeTranscriptApi().fetch(video_id)  # type: ignore[call-arg,attr-defined]
    return " ".join(getattr(entry, "text", "").strip() for entry in entries if getattr(entry, "text", None))

