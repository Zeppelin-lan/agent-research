"""Small deterministic local text search tool."""

from pathlib import Path

from pydantic import BaseModel, Field


class DocumentSearchInput(BaseModel):
    query: str = Field(min_length=1)
    directory: Path = Field(default_factory=lambda: Path("documents"))
    max_results: int = Field(default=5, ge=1, le=50)


def document_search(inputs: DocumentSearchInput) -> list[dict[str, object]]:
    root = inputs.directory.resolve()
    if not root.is_dir():
        return []
    terms = inputs.query.casefold().split()
    matches: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(lines, 1):
            score = sum(term in line.casefold() for term in terms)
            if score:
                matches.append({"path": str(path), "line": number, "text": line.strip(), "score": score})
    return sorted(matches, key=lambda item: (-int(item["score"]), str(item["path"]), int(item["line"])))[:inputs.max_results]