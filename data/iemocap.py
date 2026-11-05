"""IEMOCAP utterance index: labels, paths, transcripts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import config as _config
from config import EMOTION_MAP


UTT_LINE_RE = re.compile(
    r"\[(?P<start>[\d.]+)\s*-\s*(?P<end>[\d.]+)\]\s+"
    r"(?P<utt>\S+)\s+(?P<emot>\S+)\s+\[(?P<vad>[^\]]+)\]"
)
TRANS_RE = re.compile(
    r"^(?P<utt>\S+)\s+\[(?P<start>[\d.]+)-(?P<end>[\d.]+)\]:\s*(?P<text>.*)$"
)


@dataclass
class Utterance:
    utt_id: str
    session: int
    emotion: str
    label: int
    start: float
    end: float
    text: str
    wav_path: Path
    mocap_path: Path | None
    dialogue: str


def _dialogue_from_utt(utt_id: str) -> str:
    # Ses01F_impro01_F000 -> Ses01F_impro01
    parts = utt_id.rsplit("_", 1)
    return parts[0]


def _parse_eval_file(path: Path) -> dict[str, tuple[str, float, float]]:
    """Return utt_id -> (emotion, start, end)."""
    out: dict[str, tuple[str, float, float]] = {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    for m in UTT_LINE_RE.finditer(text):
        emot = m.group("emot").lower()
        if emot not in EMOTION_MAP:
            continue
        out[m.group("utt")] = (
            emot,
            float(m.group("start")),
            float(m.group("end")),
        )
    return out


def _parse_transcriptions(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = TRANS_RE.match(line.strip())
        if m:
            out[m.group("utt")] = m.group("text").strip()
    return out


def _resolve_wav(session_root: Path, dialogue: str, utt_id: str) -> Path | None:
    wav = session_root / "sentences" / "wav" / dialogue / f"{utt_id}.wav"
    return wav if wav.exists() else None


def _resolve_mocap(session_root: Path, dialogue: str, utt_id: str) -> Path | None:
    p = session_root / "sentences" / "MOCAP_rotated" / dialogue / f"{utt_id}.txt"
    return p if p.exists() else None


def build_utterance_index(
    sessions: list[int] | None = None,
    iemocap_root: Path | None = None,
    max_utterances: int | None = None,
) -> list[Utterance]:
    root = Path(iemocap_root or _config.IEMOCAP_ROOT)
    sessions = sessions or [1, 2, 3, 4]
    items: list[Utterance] = []

    for sid in sessions:
        session_root = root / f"Session{sid}"
        eval_dir = session_root / "dialog" / "EmoEvaluation"
        trans_dir = session_root / "dialog" / "transcriptions"
        if not eval_dir.exists():
            continue

        eval_files = sorted(
            p for p in eval_dir.glob("Ses*.txt") if not p.name.startswith("._")
        )
        for ef in eval_files:
            dialogue = ef.stem
            labels = _parse_eval_file(ef)
            texts = _parse_transcriptions(trans_dir / f"{dialogue}.txt")
            for utt_id, (emot, start, end) in labels.items():
                wav = _resolve_wav(session_root, dialogue, utt_id)
                if wav is None:
                    continue
                mocap = _resolve_mocap(session_root, dialogue, utt_id)
                items.append(
                    Utterance(
                        utt_id=utt_id,
                        session=sid,
                        emotion=emot,
                        label=EMOTION_MAP[emot],
                        start=start,
                        end=end,
                        text=texts.get(utt_id, ""),
                        wav_path=wav,
                        mocap_path=mocap,
                        dialogue=dialogue,
                    )
                )
                if max_utterances and len(items) >= max_utterances:
                    return items
    return items


def summarize_index(items: list[Utterance]) -> dict:
    from collections import Counter

    by_session = Counter(u.session for u in items)
    by_label = Counter(u.label for u in items)
    has_mocap = sum(1 for u in items if u.mocap_path is not None)
    return {
        "n_utterances": len(items),
        "by_session": dict(sorted(by_session.items())),
        "by_label": dict(sorted(by_label.items())),
        "with_mocap": has_mocap,
        "mocap_coverage": has_mocap / max(len(items), 1),
    }
