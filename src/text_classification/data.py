from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_synthetic_dataset(samples_per_decade: int = 50) -> pd.DataFrame:
    """Generate a small synthetic text dataset labeled by decade."""
    templates = {
        "1980s": [
            "arcade neon nights and cassette mixtapes",
            "retro synth melodies on vinyl records",
            "boombox beats and shoulder pads style",
        ],
        "1990s": [
            "dial up internet and grunge guitar riffs",
            "pixel games with floppy disk installs",
            "boy band anthems and skate park stories",
        ],
        "2000s": [
            "mp3 players and social network posts",
            "text messaging trends and reality tv moments",
            "early smartphones and indie blog culture",
        ],
        "2010s": [
            "streaming playlists and viral meme culture",
            "cloud apps with instant photo sharing",
            "on demand video and influencer content",
        ],
    }

    rows: list[dict[str, str]] = []
    for decade, phrases in templates.items():
        for idx in range(samples_per_decade):
            text = phrases[idx % len(phrases)]
            rows.append({"text": f"{text} sample {idx}", "label": decade})
    return pd.DataFrame(rows)


def load_dataset(path: str | Path | None = None, samples_per_decade: int = 50) -> pd.DataFrame:
    """Load a dataset from CSV or generate one if path is not provided."""
    if path is None:
        return generate_synthetic_dataset(samples_per_decade=samples_per_decade)

    csv_path = Path(path)
    data = pd.read_csv(csv_path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    return data[list(required_columns)]

