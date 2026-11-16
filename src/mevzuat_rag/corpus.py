"""Parse legislation text files into article-level chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

LAWS = {
    "6698": "Kişisel Verilerin Korunması Kanunu",
    "5411": "Bankacılık Kanunu",
    "6493": "Ödeme ve Menkul Kıymet Mutabakat Sistemleri, Ödeme Hizmetleri ve Elektronik Para Kuruluşları Hakkında Kanun",
}

ARTICLE_RE = re.compile(r"^\s*(?:MADDE|Madde)\s+(\d+)\s*[-–—]", re.MULTILINE)
# Long articles (e.g. definition lists) are split so one chunk stays within the embedder's window.
MAX_CHUNK_CHARS = 1200


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<law>:<article>#<part>"
    article_id: str  # "<law>:<article>", the unit citations and evaluation use
    law: str
    article: int
    title: str
    text: str

    @property
    def citation(self) -> str:
        return f"{LAWS[self.law]} (No. {self.law}) Madde {self.article}"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _title_before(text: str, start: int) -> str:
    """The article heading is the last non-empty line above 'MADDE n-'."""
    lines = [line.strip() for line in text[:start].splitlines() if line.strip()]
    if not lines:
        return ""
    candidate = re.sub(r"\d+$", "", lines[-1]).strip()  # drop footnote markers
    # A heading is short and does not end like a sentence; otherwise it is the previous article's body.
    return candidate if len(candidate) <= 90 and not candidate.endswith((".", ";", ":", ",")) else ""


def _split(body: str) -> list[str]:
    if len(body) <= MAX_CHUNK_CHARS:
        return [body]
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.;])\s+", body):
        if current and len(current) + len(sentence) > MAX_CHUNK_CHARS:
            parts.append(current.strip())
            current = ""
        current += sentence + " "
    if current.strip():
        parts.append(current.strip())
    return parts


def parse_law(law: str, text: str) -> list[Chunk]:
    matches = list(ARTICLE_RE.finditer(text))
    chunks: list[Chunk] = []
    seen: set[int] = set()
    for index, match in enumerate(matches):
        article = int(match.group(1))
        # Amending provisions quoted inside the text can repeat a number; keep the first (the law's own article).
        if article in seen:
            continue
        seen.add(article)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = _clean(text[match.start():end])
        title = _title_before(text, match.start())
        # The next article's heading sits at the end of this slice; it belongs to that article.
        next_title = _clean(_title_before(text, end)) if index + 1 < len(matches) else ""
        if next_title and body.endswith(next_title):
            body = body[: -len(next_title)].strip()
        for part_no, part in enumerate(_split(body)):
            chunks.append(
                Chunk(
                    chunk_id=f"{law}:{article}#{part_no}",
                    article_id=f"{law}:{article}",
                    law=law,
                    article=article,
                    title=title,
                    text=part,
                )
            )
    return chunks


def load_corpus(raw_dir: Path = RAW_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for law in LAWS:
        chunks.extend(parse_law(law, (raw_dir / f"{law}.txt").read_text(encoding="utf-8")))
    return chunks
