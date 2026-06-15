import json
import random
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tqdm import tqdm

from agents import Critic
from schema import CriticVerdict, LabeledReport

STRONG_MODEL = "deepseek/deepseek-v4-pro"
NORMAL_MODEL = "deepseek/deepseek-v4-flash"


@dataclass
class Comparaison:
    report: LabeledReport
    strong: CriticVerdict
    normal: CriticVerdict


def load_reports(path: str) -> list[LabeledReport]:
    reports = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            reports.append(LabeledReport.model_validate_json(line))
    return reports


def compare_critics(
    in_path: str,
    n: int = 150,
    concurrency: int = 8,
    seed: int = 0,
    disagreements_path: str = "data/critic_disagreements.jsonl",
) -> list[Comparaison]:
    reports = load_reports(in_path)
    random.seed(seed)
    random.shuffle(reports)
    reports = reports[:n]

    strong = Critic(STRONG_MODEL)
    normal = Critic(NORMAL_MODEL)

    def judge_both(report: LabeledReport) -> Comparaison:
        return Comparaison(
            report,
            strong.critique(report),
            normal.critique(report),
        )

    comparaisons: list[Comparaison] = []
    failures = 0

    with ThreadPoolExecutor(concurrency) as pool:
        futures = [pool.submit(judge_both, r) for r in reports]
        for future in tqdm(as_completed(futures), total=len(reports), desc='Judging'):
            try:
                comparaisons.append(future.result())
            except Exception as e:
                print(f"Judge request failed: {e}")
                failures += 1

    print_summary(comparaisons, failures)
    save_disagreements(comparaisons, disagreements_path)
    return comparaisons


def _model_name(model: str) -> str:
    return model.split('/')[-1]


def _agreement(a: list, b: list) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def print_summary(comparaisons: list[Comparaison], failures: int) -> None:
    total = len(comparaisons) + failures
    print(f"\nCompared {len(comparaisons)}/{total} reports  ({failures} failures)")

    if not comparaisons:
        return

    gen_cat = [c.report.category for c in comparaisons]
    gen_sev = [c.report.severity for c in comparaisons]
    strong_cat = [c.strong.judged_category for c in comparaisons]
    strong_sev = [c.strong.judged_severity for c in comparaisons]
    normal_cat = [c.normal.judged_category for c in comparaisons]
    normal_sev = [c.normal.judged_severity for c in comparaisons]

    print("\nAgreement with generator label (raw, unverified):")
    print(f"  {'':<22}{'category':>10}{'severity':>10}")
    print(f"  {_model_name(STRONG_MODEL):<22}{_agreement(gen_cat, strong_cat):>10.1%}{_agreement(gen_sev, strong_sev):>10.1%}")
    print(f"  {_model_name(NORMAL_MODEL):<22}{_agreement(gen_cat, normal_cat):>10.1%}{_agreement(gen_sev, normal_sev):>10.1%}")

    strong_plaus = sum(1 for c in comparaisons if c.strong.plausible) / len(comparaisons)
    normal_plaus = sum(1 for c in comparaisons if c.normal.plausible) / len(comparaisons)
    print("\nplausible rate (fraction flagged plausible):")
    print(f"  {_model_name(STRONG_MODEL):<22}{strong_plaus:>10.1%}")
    print(f"  {_model_name(NORMAL_MODEL):<22}{normal_plaus:>10.1%}")

    strong_plaus_labels = [c.strong.plausible for c in comparaisons]
    normal_plaus_labels = [c.normal.plausible for c in comparaisons]
    print("\ninter-judge agreement (strong vs. normal):")
    print(f"  category    {_agreement(strong_cat, normal_cat):.1%}")
    print(f"  severity    {_agreement(strong_sev, normal_sev):.1%}")
    print(f"  plausible   {_agreement(strong_plaus_labels, normal_plaus_labels):.1%}")


def save_disagreements(comparaisons: list[Comparaison], out_path: str) -> None:
    disagreements = [
        c for c in comparaisons
        if c.strong.judged_category != c.normal.judged_category
        or c.strong.judged_severity != c.normal.judged_severity
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        for c in disagreements:
            record = {
                "report": c.report.model_dump(mode="json", by_alias=True),
                "strong": c.strong.model_dump(mode="json"),
                "normal": c.normal.model_dump(mode="json"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(disagreements)} strong/normal disagreements to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compare a strong judge critic against a normal one.")
    parser.add_argument("--in", dest="in_path", type=str, default="data/raw.jsonl", help="labeled reports to judge")
    parser.add_argument("--n", type=int, default=150, help="limit number of reports")
    parser.add_argument("--concurrency", type=int, default=8, help="number of concurrent requests")
    parser.add_argument("--seed", type=int, default=0, help="seed for the report subsample")
    parser.add_argument("--disagreements", dest="disagreements_path", type=str,
                        default="data/critic_disagreements.jsonl", help="where to write strong/normal disagreements")
    args = parser.parse_args()

    compare_critics(args.in_path, args.n, args.concurrency, args.seed, args.disagreements_path)
