## YouTube Video Processing (Multi‑Agent Pipeline)

This repository provides a **professional, extensible structure** for downloading a YouTube video + transcript and running a **multi‑agent LLM pipeline** to:

- Extract the video and transcript
- Fix transcript wording/correctness (without deleting key points)
- Translate to **Turkish (Istanbul)**
- Produce a final **correct, conversational dialogue** (“Host” / “Guest”)

> Important: the `data/` directory is intentionally **ignored by git** so you never push large videos/transcripts.

## What it does (end‑to‑end)

Given a YouTube URL, `app/main.py`:

1. Downloads the video to `data/video/`
2. Fetches the transcript and saves it to `data/transcript/`
3. Extracts a passage from the transcript (size controlled by `--passage-chars`)
4. Runs agents:
   - **Correction agent** → `*_corrected.txt`
   - **Translation agent (TR / Istanbul)** → `*_tr_istanbul.txt`
   - **Dialogue agent** → `*_dialogue.txt`

Outputs are stored under `data/output/`.

## Repository structure

```text
youtube/
  app/
    main.py                # CLI entry point + pipeline orchestration
  utils/
    agents.py              # Multi-agent implementations
    youtube.py             # Video download + transcript retrieval helpers
    llm_client.py          # ChatOpenAI wrapper + model alias routing (+ optional prompt caching)
    prompt_loader.py       # Loads prompts from /prompts
  prompts/
    system_extract.txt     # Passage selection instructions (prompt template)
    system_modify.txt      # Correction instructions
    system_translate.txt   # Translation instructions (TR / Istanbul)
    system_dialogue.txt    # Dialogue conversion instructions
  data/                    # Ignored by git; contains downloaded and generated artifacts
    input/
    output/
    video/
    transcript/
  requirements.txt
  .gitignore
  LICENSE
  README.md
```

## Setup (Windows + Python 3.12)

### Prerequisites

- **Python 3.12** installed (recommended: `py` launcher available)
- An OpenAI API key exported as an environment variable

### Create + activate venv

PowerShell:

```powershell
cd E:\codes_py\youtube
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

## Configuration (environment variables)

Required:

- **`OPENAI_API_KEY`**: your OpenAI API key used by `utils/llm_client.py`

Optional (prompt caching via Responses API, when supported):

- **`PROMPT_CACHE_RETENTION`**: retention window (seconds). Use `0` to disable.
- **`PROMPT_CACHE_SHARDS`**: integer shard count to spread cache keys (default `1`)
- **`PROMPT_VERSION`**: string to bump when prompts change (default `v1`)

Example (PowerShell):

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:PROMPT_CACHE_RETENTION="0"
```

## Run

### Run as a module (recommended)

```powershell
python -m app.main --url "https://www.youtube.com/watch?v=VIDEO_ID" --model gpt-4o
```

### Run the file directly (debugger-friendly)

`app/main.py` adds the repo root to `sys.path` so imports work when launched as a script:

```powershell
python .\app\main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

### CLI arguments

- **`--url`**: YouTube URL (has a default in the script for convenience)
- **`--model`**: model or alias (examples: `gpt-4o`, `gpt-4o-mini`, `gpt5-low`, `gpt5-1`)
- **`--passage-chars`**: max characters taken from transcript for the downstream LLM steps

## Agents (multi‑agent design)

Agents are defined in `utils/agents.py`:

- **`VideoTranscriptAgent`**
  - Downloads the video into `data/video/`
  - Fetches transcript text and writes it into `data/transcript/`
  - Returns `(video_path, transcript_path, transcript_text)` for the pipeline
- **`TranscriptCorrectionAgent`**
  - Fixes grammar/wording while **preserving meaning and key points**
- **`TranslationAgent`**
  - Translates to **Turkish (Istanbul)**, preserving names/numbers/technical terms
- **`DialogueAgent`**
  - Final correctness pass + converts output to a conversation (Host/Guest)

Each agent loads its prompt template from `prompts/` and calls the LLM via `utils/llm_client.py`.

## Prompts

All prompts are plain text files in `prompts/`. Customize them to change style, verbosity, speaker names, etc.

The code loads them via:

- `utils/prompt_loader.py` → `load_prompt("<prompt_basename>")`

## Outputs

Local artifacts created during a run:

- **Video**: `data/video/*.mp4`
- **Raw transcript**: `data/transcript/*.txt`
- **Processing outputs**: `data/output/*_{passage,corrected,tr_istanbul,dialogue}.txt`

### Git safety

`data/` is ignored by `.gitignore` to prevent committing large files (videos/transcripts).

## Troubleshooting

### YouTube download fails with `HTTP Error 400: Bad Request`

This is common when YouTube changes internal endpoints. This project uses **`pytubefix`** (a maintained fork) to reduce breakage, but failures can still happen.

Try:

- `pip install -U pytubefix`
- Try a different video URL (age-restricted / region-locked videos often fail)
- If you need maximum robustness, consider replacing the download layer with `yt-dlp` (not implemented here yet)

### Transcript not available

Not every video has a transcript. `youtube-transcript-api` may raise exceptions if:

- transcripts are disabled
- the video is private
- captions are not available in any language

### `pip` “hash mismatch / require-hashes” errors

If your machine has `pip` configured with `--require-hashes`, installs can fail unless every dependency is pinned with hashes.

Options:

- Disable `require-hashes` in your pip config, or
- Move to a lockfile workflow (e.g., `pip-tools`) and commit a hashed lockfile

## License

MIT — see `LICENSE`.
