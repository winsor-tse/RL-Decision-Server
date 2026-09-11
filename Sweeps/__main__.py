"""Plan, run, and summarize local sequential experiments."""

import argparse

from Sweeps.runner import build_plan, read_json, report, run_plan, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("config", help="JSON design configuration")
        command.add_argument("--output", required=True, help="plan JSON file or run directory")
        if name == "run":
            command.add_argument("--automation-config", help="bridge YAML; trainer/restore commands are ignored")
    summary = commands.add_parser("report")
    summary.add_argument("directory")
    summary.add_argument("--output", required=True, help="summary JSON path")
    summary.add_argument("--confidence", type=float, default=0.95)
    summary.add_argument("--resamples", type=int, default=2000)
    summary.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            value = report(args.directory, confidence=args.confidence, resamples=args.resamples, seed=args.seed)
            write_json(args.output, value)
            print(f"Summarized {value['completed_trials']}/{value['planned_trials']} trials: {args.output}")
        else:
            plan = build_plan(read_json(args.config))
            print(f"{plan['method']}: {len(plan['trials'])} sequential trials; "
                  f"{plan['duplicate_configurations_removed']} duplicate configurations removed")
            if args.command == "plan":
                write_json(args.output, plan)
                print(f"Plan written: {args.output} (no game connection)")
            else:
                run_plan(plan, args.output, automation_config=args.automation_config)
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
