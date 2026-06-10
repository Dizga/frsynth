from collections import Counter

from tqdm import tqdm

from agents import Generator
from schema import LabeledReport


def run_pilot(n: int, out_path: str) -> list[LabeledReport]:
    generator = Generator()

    examples: list[LabeledReport] = []
    failures = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for _ in tqdm(range(n), desc="generating"):
            example = generator.generate_report()
            if example is None:
                failures += 1
                continue
            f.write(example.model_dump_json(by_alias=True) + "\n")
            examples.append(example)

    print_summary(examples, failures, out_path)
    return examples


def print_summary(examples: list[LabeledReport], failures: int, out_path: str) -> None:
    total = len(examples) + failures
    print(f"\nWrote {len(examples)}/{total} examples to {out_path}  ({failures} parse failures)")

    if not examples:
        return

    category_counts = Counter(e.category for e in examples)
    severity_counts = Counter(e.severity for e in examples)
    distractor_count = sum(1 for e in examples if e.has_distractor)

    print("\ncategory distribution:")
    for label, count in category_counts.most_common():
        print(f"  {label:<24} {count}")

    print("\nseverity distribution:")
    for label, count in severity_counts.most_common():
        print(f"  {label:<24} {count}")

    print(f"\nwith distractor: {distractor_count}/{len(examples)}")

run_pilot(50, out_path="data/pilot.jsonl")