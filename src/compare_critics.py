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

def compare_critics(in_path: str, n: int = 150, concurrency: int = 8) -> list[Comparaison]:
    reports = load_reports(in_path)
    random.shuffle(reports)

    reports = reports[:n]

    strong = Critic(STRONG_MODEL)
    normal = Critic(NORMAL_MODEL)

    def judge_both(report) -> Comparaison:
        return Comparaison(
            report,
            strong.critique(report),
            normal.critique(report),
        )
    
    comparaison = []

    with ThreadPoolExecutor(concurrency) as pool:
        futures = [pool.submit(judge_both, r) for r in reports]
        for future in tqdm(as_completed(futures), total= len(reports), desc='Judging'):
            comparaison.append(future.result())

    print_summary(comparaison)
    return comparaison


def _accuracy(verdicts: list[tuple[LabeledReport, CriticVerdict]]) -> tuple[float, float, float]:
    cat = sum([1 for r, v in verdicts if v.judged_category == r.category]) / len(verdicts)
    sev = sum([1 for r, v in verdicts if v.judged_severity == r.severity]) / len(verdicts)
    plausible = sum([1 for _, v in verdicts if v.plausible]) / len(verdicts)
    return cat, sev, plausible

def print_summary(comparaison: list[Comparaison]) -> None:
    print(f'\nCompared {len(comparaison)} reports.')

    strong_verdicts = [(c.report, c.strong) for c in comparaison]
    normal_verdicts = [(c.report, c.normal) for c in comparaison]

    strong_cat, strong_sev, strong_plausible = _accuracy(strong_verdicts)
    normal_cat, normal_sev, normal_plausible = _accuracy(normal_verdicts)

    print(f'\nStrong Critic:')
    print(f'\nModel: {STRONG_MODEL}')
    print(f'\nJudged accuracy for categories: {strong_cat:6.1%}')
    print(f'\nJudged accuracy for severities: {strong_sev:6.1%}')
    print(f'\nJudged accuracy for plausibility: {strong_plausible:6.1%}')
    print(f'\n\nNormal Critic:')
    print(f'\nModel: {NORMAL_MODEL}')
    print(f'\nJudged accuracy for categories: {normal_cat:6.1%}')
    print(f'\nJudged accuracy for severities: {normal_sev:6.1%}')
    print(f'\nJudged accuracy for plausibility: {normal_plausible:6.1%}')

    cat_agree = sum(1 for c in comparaison if c.strong.judged_category == c.normal.judged_category)
    sev_agree = sum(1 for c in comparaison if c.strong.judged_severity == c.normal.judged_severity)
    plaus_agree = sum(1 for c in comparaison if c.strong.plausible == c.normal.plausible)
    n = len(comparaison)

    print("\nAgreement between strong and normal judge:")
    print(f"  category    {cat_agree / n:6.1%}  ({cat_agree}/{n})")
    print(f"  severity    {sev_agree / n:6.1%}  ({sev_agree}/{n})")
    print(f"  plausible   {plaus_agree / n:6.1%}  ({plaus_agree}/{n})")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compare a strong judge critic against a normal one.")
    parser.add_argument("--in", dest="in_path", type=str, default="data/raw.jsonl", help="labeled reports to judge")
    parser.add_argument("--n", type=int, default=150, help="limit number of reports")
    args = parser.parse_args()

    compare_critics(args.in_path, args.n)
