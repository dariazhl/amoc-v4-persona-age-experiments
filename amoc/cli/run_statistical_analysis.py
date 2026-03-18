import argparse
from amoc.analysis.statistics import run_statistical_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    run_statistical_analysis(args.model)


if __name__ == "__main__":
    main()
