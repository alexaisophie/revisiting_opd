import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import Dataset, concatenate_datasets


PREFIX_RE = re.compile(
    r"^\s*Solve the following math problem step by step\.\s*"
    r"The last line of your response should be of the form Answer: "
    r"\$Answer \(without quotes\) where \$Answer is the answer to the problem\.\s*",
    re.DOTALL,
)
SUFFIX_RE = re.compile(
    r"\s*Remember to put your answer on its own line after \"Answer:\"\.?\s*$",
    re.DOTALL,
)
GSM8K_ANSWER_RE = re.compile(r"####\s*(.+)\s*$", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local math train/eval parquets for TrustOPD experiments."
    )
    parser.add_argument("--raw_dir", type=Path, default=Path("data/math_raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/math_opd"))
    parser.add_argument("--eval_dir", type=Path, default=Path("data/eval_math"))
    parser.add_argument("--train_limit", type=int, default=20_000)
    parser.add_argument("--train_smoke_limit", type=int, default=512)
    parser.add_argument("--eval_smoke_limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260516)
    return parser.parse_args()


def clean_dapo_prompt(prompt: str) -> str:
    prompt = PREFIX_RE.sub("", prompt)
    prompt = SUFFIX_RE.sub("", prompt)
    return prompt.strip()


def prompt_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value:
        return value[0]["content"]
    if isinstance(value, tuple) and value:
        return value[0]["content"]
    raise ValueError(f"Unsupported prompt format: {type(value)}")


def dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise ValueError(f"Unsupported dict-like value: {type(value)}")


def make_record(
    *,
    data_source: str,
    split: str,
    index: int,
    question: str,
    answer: Any,
    source_index: Any = None,
) -> dict[str, Any]:
    question = str(question).strip()
    answer = str(answer).strip()
    if not question:
        raise ValueError(f"Empty question in {data_source} at index={index}")
    if not answer:
        raise ValueError(f"Empty answer in {data_source} at index={index}")

    return {
        "data_source": data_source,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "prompt": [{"role": "user", "content": question}],
        "extra_info": {
            "split": split,
            "index": index,
            "question": question,
            "answer": answer,
            "source": data_source,
            "source_index": "" if source_index is None else str(source_index),
        },
        "env_kwargs": {
            "task_type": "math",
            "question": question,
            "ground_truth": answer,
            "data_source": data_source,
        },
    }


def records_to_dataset(records: Iterable[dict[str, Any]]) -> Dataset:
    records = list(records)
    if not records:
        raise ValueError("No records to save")
    return Dataset.from_list(records)


def prepare_dapo_train(raw_dir: Path, train_limit: int, seed: int) -> Dataset:
    path = raw_dir / "dapo-math-17k.parquet"
    df = pd.read_parquet(path, columns=["prompt", "reward_model", "extra_info"])
    if train_limit > 0 and train_limit < len(df):
        df = df.sample(n=train_limit, random_state=seed)
    df = df.reset_index(drop=True)

    records = []
    for idx, row in df.iterrows():
        reward_model = dict_value(row["reward_model"])
        extra_info = dict_value(row["extra_info"])
        question = clean_dapo_prompt(prompt_content(row["prompt"]))
        records.append(
            make_record(
                data_source="dapo-math",
                split="train",
                index=idx,
                question=question,
                answer=reward_model["ground_truth"],
                source_index=extra_info.get("index", idx),
            )
        )
    return records_to_dataset(records)


def prepare_aime_2024(raw_dir: Path) -> Dataset:
    path = raw_dir / "aime-2024.parquet"
    df = pd.read_parquet(path, columns=["prompt", "reward_model", "extra_info"])

    rows = []
    seen_questions = set()
    for _, row in df.iterrows():
        reward_model = dict_value(row["reward_model"])
        extra_info = dict_value(row["extra_info"])
        question = str(extra_info.get("raw_problem") or clean_dapo_prompt(prompt_content(row["prompt"]))).strip()
        if question in seen_questions:
            continue
        seen_questions.add(question)
        rows.append((question, reward_model["ground_truth"], extra_info.get("index", len(rows))))

    records = [
        make_record(
            data_source="aime-2024",
            split="test",
            index=idx,
            question=question,
            answer=answer,
            source_index=source_index,
        )
        for idx, (question, answer, source_index) in enumerate(rows)
    ]
    return records_to_dataset(records)


def prepare_aime_2025(raw_dir: Path) -> Dataset:
    path = raw_dir / "aime-2025.parquet"
    df = pd.read_parquet(path)
    records = [
        make_record(
            data_source="aime-2025",
            split="test",
            index=idx,
            question=row["problem"],
            answer=row["answer"],
            source_index=row.get("problem_idx", idx),
        )
        for idx, row in df.iterrows()
    ]
    return records_to_dataset(records)


def prepare_math_500(raw_dir: Path) -> Dataset:
    path = raw_dir / "math-500.jsonl"
    df = pd.read_json(path, lines=True)
    records = [
        make_record(
            data_source="math-500",
            split="test",
            index=idx,
            question=row["problem"],
            answer=row["answer"],
            source_index=row.get("unique_id", idx),
        )
        for idx, row in df.iterrows()
    ]
    return records_to_dataset(records)


def extract_gsm8k_answer(answer: str) -> str:
    match = GSM8K_ANSWER_RE.search(answer)
    if not match:
        raise ValueError(f"Cannot extract GSM8K final answer from: {answer[:120]}")
    return match.group(1).strip()


def prepare_gsm8k(raw_dir: Path) -> Dataset:
    path = raw_dir / "gsm8k-test.parquet"
    df = pd.read_parquet(path)
    records = [
        make_record(
            data_source="gsm8k",
            split="test",
            index=idx,
            question=row["question"],
            answer=extract_gsm8k_answer(row["answer"]),
            source_index=idx,
        )
        for idx, row in df.iterrows()
    ]
    return records_to_dataset(records)


def save_dataset(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))
    print(f"[OK] {path}: {len(dataset)} rows")


def save_smoke(dataset: Dataset, path: Path, limit: int, seed: int) -> None:
    if limit <= 0:
        return
    count = min(limit, len(dataset))
    sampled = dataset.shuffle(seed=seed).select(range(count))
    save_dataset(sampled, path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.eval_dir.mkdir(parents=True, exist_ok=True)

    train = prepare_dapo_train(args.raw_dir, args.train_limit, args.seed)
    save_dataset(train, args.output_dir / "train.parquet")
    save_smoke(train, args.output_dir / "train_smoke.parquet", args.train_smoke_limit, args.seed)

    eval_datasets = {
        "aime_2024": prepare_aime_2024(args.raw_dir),
        "aime_2025": prepare_aime_2025(args.raw_dir),
        "math_500": prepare_math_500(args.raw_dir),
        "gsm8k": prepare_gsm8k(args.raw_dir),
    }
    for name, dataset in eval_datasets.items():
        save_dataset(dataset, args.eval_dir / f"{name}.parquet")

    eval_all = concatenate_datasets(list(eval_datasets.values()))
    save_dataset(eval_all, args.eval_dir / "eval_all.parquet")
    save_dataset(eval_all, args.output_dir / "test.parquet")
    save_smoke(eval_all, args.eval_dir / "eval_smoke.parquet", args.eval_smoke_limit, args.seed)


if __name__ == "__main__":
    main()
