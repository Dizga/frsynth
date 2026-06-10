from argparse import ArgumentParser
from collections import Counter

from tqdm import tqdm

from agents import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from schema import LabeledReport


def generate_reports(n: int, out_path: str, concurrency = 8):
    generator = Generator()

    reports: list[LabeledReport] = []
    failures = 0

    with(
        open(out_path, 'a', encoding="utf-8") as f,
        ThreadPoolExecutor(concurrency) as pool
    ):
        futures = [pool.submit(generator.generate_report) for i in range(n)]
        for future in tqdm(as_completed(futures), total=n, desc="generating"):
            try:
                report = future.result()
            except Exception as e:
                print(f"Generation request failed {e}")
                report = None
            if report is None:
                failures += 1
                continue
            f.write(report.model_dump_json(by_alias=True) + "\n") 
            reports.append(report)

    print_summary(reports, failures, out_path)
    return reports


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

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate a batch of labeled reports.")
    parser.add_argument("--n", type=int, default=1000, help="number of reports to generate")
    parser.add_argument("--out", type=str, default="data/raw.jsonl", help="output file path")
    args = parser.parse_args()

    generate_reports(args.n, args.out)