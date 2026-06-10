from argparse import ArgumentParser

from tqdm import tqdm

from agents import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_reports(n: int, out_path: str, concurrency = 8):
    generator = Generator()

    with(
        open(out_path, 'w', encoding="utf-8") as f,
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
                continue
            f.write(report.model_dump_json(by_alias=True) + "\n") 


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate a batch of labeled reports.")
    parser.add_argument("--n", type=int, default=1000, help="number of reports to generate")
    parser.add_argument("--out", type=str, default="data/raw.jsonl", help="output file path")
    args = parser.parse_args()

    generate_reports(args.n, args.out)