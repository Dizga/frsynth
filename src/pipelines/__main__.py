"""Dispatcher: register pipelines here and run one by name.

    python src/run_pipeline.py <name> [args]      # from the repo root
"""
from argparse import ArgumentParser

from pipelines import critic_compare

# name -> pipeline module (each exposes NAME, add_args, main)
PIPELINES = {
    critic_compare.NAME: critic_compare,
}


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(prog="pipelines", description="Run a named FRSynth pipeline.")
    sub = parser.add_subparsers(dest="pipeline", required=True)
    for name, module in PIPELINES.items():
        p = sub.add_parser(name, help=module.__doc__)
        module.add_args(p)

    args = parser.parse_args(argv)
    PIPELINES[args.pipeline].main(args)


if __name__ == "__main__":
    main()
