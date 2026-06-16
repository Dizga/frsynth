"""Generate reports, then compare the strong and normal critic on them."""
from argparse import ArgumentParser, Namespace

from compare_critics import compare_critics
from generate_reports import generate_reports
from pipelines._common import prepare_run

NAME = "critic_compare"


def run(n: int = 100, concurrency: int = 8, seed: int = 0) -> None:
    out_dir = prepare_run(NAME, seed)
    reports_path = out_dir / "reports.jsonl"
    disagreements_path = out_dir / "critic_disagreements.jsonl"

    # generate_reports appends; start clean so re-runs don't pile up.
    reports_path.unlink(missing_ok=True)

    print(f"== Generating {n} reports -> {reports_path} ==")
    generate_reports(n, str(reports_path), concurrency)

    print(f"\n== Comparing critics on {reports_path} ==")
    compare_critics(
        str(reports_path),
        n=n,
        concurrency=concurrency,
        seed=seed,
        disagreements_path=str(disagreements_path),
    )


def add_args(parser: ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=100, help="number of reports to generate and judge")
    parser.add_argument("--concurrency", type=int, default=8, help="number of concurrent requests")
    parser.add_argument("--seed", type=int, default=0, help="seed for generation and the critic subsample")


def main(args: Namespace) -> None:
    run(args.n, args.concurrency, args.seed)
