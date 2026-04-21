import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 32
TOTAL_QUESTIONS = 50
RANDOM_SEED = 20260421

DATA_DIR = Path(__file__).resolve().parent
DOCS_DIR = DATA_DIR / "docs"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
GOLDEN_PATH = DATA_DIR / "golden_set.jsonl"


def load_docs(doc_dir: str) -> List[Dict]:
    directory = Path(doc_dir)
    docs: List[Dict] = []

    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"doc_name": path.stem, "path": str(path), "text": text})

    return docs


def chunk_document(text: str, doc_name: str, chunk_size: int, overlap: int) -> List[Dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    step = chunk_size - overlap
    chunks: List[Dict] = []
    chunk_index = 0

    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Avoid cutting in the middle of a word.
            while end < len(text) and not text[end].isspace():
                end += 1

        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue

        chunks.append(
            {
                "chunk_id": f"doc_{doc_name}_c_{chunk_index}",
                "doc_name": doc_name,
                "chunk_text": chunk_text,
                "char_start": start,
                "char_end": end,
            }
        )
        chunk_index += 1

        if end >= len(text):
            break

    return chunks


def build_chunks_from_docs(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    all_chunks: List[Dict] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc["text"], doc["doc_name"], chunk_size, overlap))
    return all_chunks


def _extract_answer_from_chunk(chunk_text: str) -> str:
    candidates = re.split(r"[\n\r]+|(?<=[.!?;:])\s+", chunk_text)
    candidates = [c.strip(" -\t") for c in candidates if c and c.strip(" -\t")]

    ranked = sorted(
        candidates,
        key=lambda item: (
            30 <= len(item) <= 220,
            any(char.isdigit() for char in item),
            len(item),
        ),
        reverse=True,
    )

    if ranked:
        return ranked[0]
    return chunk_text[:200].strip()


def _build_question(doc_name: str, answer: str) -> str:
    snippet = answer[:70].rstrip(",.;: ")
    return f"Theo tai lieu {doc_name}, quy dinh nao mo ta noi dung: '{snippet}'?"


def _difficulty(answer: str) -> str:
    if len(answer) <= 90:
        return "easy"
    if len(answer) <= 160:
        return "medium"
    return "hard"


def _select_balanced_chunks(chunks: List[Dict], target_n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["doc_name"]].append(chunk)

    for doc_chunks in grouped.values():
        rng.shuffle(doc_chunks)

    doc_names = sorted(grouped.keys())
    rng.shuffle(doc_names)

    selected: List[Dict] = []
    while len(selected) < target_n:
        added_in_round = False
        for doc_name in doc_names:
            if grouped[doc_name]:
                selected.append(grouped[doc_name].pop(0))
                added_in_round = True
                if len(selected) == target_n:
                    break
        if not added_in_round:
            break

    return selected


def generate_questions_from_chunks(chunks: List[Dict], target_n: int = TOTAL_QUESTIONS) -> List[Dict]:
    unique_chunks = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(unique_chunks) < target_n:
        raise ValueError(
            f"Not enough unique chunks to generate {target_n} questions. "
            f"Current unique chunks: {len(unique_chunks)}."
        )

    selected_chunks = _select_balanced_chunks(list(unique_chunks.values()), target_n, RANDOM_SEED)
    if len(selected_chunks) < target_n:
        raise ValueError(f"Could only select {len(selected_chunks)} chunks, expected {target_n}.")

    records: List[Dict] = []
    for chunk in selected_chunks:
        answer = _extract_answer_from_chunk(chunk["chunk_text"])
        records.append(
            {
                "question": _build_question(chunk["doc_name"], answer),
                "expected_answer": answer,
                "expected_chunk_id": chunk["chunk_id"],
                "source_doc": chunk["doc_name"],
                "difficulty": _difficulty(answer),
            }
        )

    return records


def validate_golden_set(records: List[Dict]) -> Dict:
    required_fields = {"question", "expected_answer", "expected_chunk_id", "source_doc", "difficulty"}
    missing_fields = 0
    for row in records:
        if not required_fields.issubset(row.keys()):
            missing_fields += 1

    chunk_ids = [row.get("expected_chunk_id", "") for row in records]
    unique_chunk_ids = len(set(chunk_ids))

    return {
        "total_records": len(records),
        "unique_expected_chunk_ids": unique_chunk_ids,
        "duplicate_expected_chunk_ids": len(records) - unique_chunk_ids,
        "missing_field_records": missing_fields,
    }


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_chunk_mode() -> List[Dict]:
    docs = load_docs(str(DOCS_DIR))
    if not docs:
        raise RuntimeError(f"No docs found in {DOCS_DIR}")

    chunks = build_chunks_from_docs(docs, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
    _write_jsonl(CHUNKS_PATH, chunks)

    print(f"Loaded docs: {len(docs)}")
    print(f"Total chunks: {len(chunks)}")
    stats = Counter(chunk["doc_name"] for chunk in chunks)
    for doc_name in sorted(stats):
        print(f"- {doc_name}: {stats[doc_name]} chunks")
    print(f"Saved to {CHUNKS_PATH.as_posix()}")

    return chunks


def run_golden_mode(target_n: int) -> List[Dict]:
    chunks = _read_jsonl(CHUNKS_PATH)
    if not chunks:
        raise RuntimeError(
            "Missing or empty chunks.jsonl. Run: python data/synthetic_gen.py --mode chunk"
        )

    records = generate_questions_from_chunks(chunks, target_n=target_n)
    _write_jsonl(GOLDEN_PATH, records)
    report = validate_golden_set(records)

    print(f"Selected {len(records)} unique chunks")
    print(f"Generated {len(records)} questions")
    print(f"Saved to {GOLDEN_PATH.as_posix()}")
    print(
        "Validation: "
        f"total={report['total_records']}, "
        f"unique_ids={report['unique_expected_chunk_ids']}, "
        f"duplicates={report['duplicate_expected_chunk_ids']}, "
        f"missing_fields={report['missing_field_records']}"
    )

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate chunk-aware synthetic benchmark dataset.")
    parser.add_argument(
        "--mode",
        choices=["all", "chunk", "golden"],
        default="all",
        help="all: chunk + golden, chunk: only chunks, golden: only golden set",
    )
    parser.add_argument("--n", type=int, default=TOTAL_QUESTIONS, help="Number of golden questions to generate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"all", "chunk"}:
        run_chunk_mode()
    if args.mode in {"all", "golden"}:
        run_golden_mode(args.n)


if __name__ == "__main__":
    main()
