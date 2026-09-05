"""CLI for the additive forward study. Running this file never applies DDL."""
import argparse
from datetime import date
import json
import os


def database_url():
    # Explicit study override is useful for an isolated database. No automatic
    # Streamlit secrets import; this CLI must never initialize legacy tables.
    return os.environ.get("FORWARD_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("migrate", "register", "run", "export"))
    parser.add_argument("--study", default="spread-finder-weekly-v1")
    parser.add_argument("--start-week")
    parser.add_argument("--output")
    args = parser.parse_args()
    from range_finder.forward_test.config import utcnow, methodology, UNIVERSE, MODELS, COHORT
    from range_finder.forward_test.store import Store
    from range_finder.trading_week import trading_week
    store = Store.postgres(database_url())
    try:
        if args.command == "migrate":
            store.migrate()
            print("Additive forward-study migrations applied to configured database")
        elif args.command == "register":
            if not args.start_week:
                parser.error("register requires --start-week YYYY-MM-DD (Monday label)")
            start = date.fromisoformat(args.start_week)
            week = trading_week(start)
            if start != week.monday or utcnow() >= week.capture_end:
                parser.error("Choose a Monday label whose capture window has not ended")
            _, config = methodology()
            store.register(args.study, args.start_week, utcnow(),
                           {"universe": UNIVERSE, "models": MODELS, "cohort": COHORT, "activation_methodology": config})
            print(f"Registered {args.study} beginning {args.start_week}; no forecasts reconstructed")
        elif args.command == "run":
            from range_finder.forward_test.provider import TradierProvider
            from range_finder.forward_test.runner import run_study
            provider = TradierProvider(os.environ.get("TRADIER_TOKEN"), clock=utcnow,
                                       declared_delay_seconds=int(os.environ.get("FORWARD_TEST_DATA_DELAY_SECONDS", "0")),
                                       history_loader=store.legacy_weekly)
            try:
                result = run_study(store, provider, args.study, clock=utcnow)
                print(json.dumps(result, indent=2))
                if result["errors"]:
                    raise SystemExit(1)
            finally:
                provider.close()
        else:
            if not args.output:
                parser.error("export requires --output path.xlsx")
            from pathlib import Path
            from range_finder.forward_test.results import load_results, build_workbook
            Path(args.output).write_bytes(build_workbook(load_results(store, args.study)))
            print(f"Exported durable records to {args.output}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
