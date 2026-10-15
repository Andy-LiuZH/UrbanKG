"""Command line interface.

    python -m unified_ukg build  --config configs/setagaya_sample.yaml
    python -m unified_ukg text   --tables ukg_sample/setagaya/tables --out ukg_sample/setagaya/text
    python -m unified_ukg neo4j  --tables ukg_sample/setagaya/tables [--env .env] [--database neo4j] [--dry-run]
    python -m unified_ukg stats  --tables ukg_sample/setagaya/tables
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .config import load_config
from .export.neo4j import load_into_neo4j, plan
from .export.text import export_text
from .graph import UKG
from .pipeline import build_and_save


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unified_ukg", description="Unified Urban Knowledge Graph construction")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build the UKG from a config and save tables + text")
    p_build.add_argument("--config", required=True)
    p_build.add_argument("--no-text", action="store_true", help="skip the text (KGE) export")

    p_text = sub.add_parser("text", help="export saved tables to the text (KGE) format")
    p_text.add_argument("--tables", required=True)
    p_text.add_argument("--out", required=True)
    p_text.add_argument("--typed-relations", action="store_true", help="relation names Head_rel_Tail")

    p_neo = sub.add_parser("neo4j", help="load saved tables into Neo4j (MERGE, idempotent)")
    p_neo.add_argument("--tables", required=True)
    p_neo.add_argument("--env", default=".env", help="file with NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD")
    p_neo.add_argument("--database", default=None, help="defaults to NEO4J_DATABASE or 'neo4j'")
    p_neo.add_argument("--batch-size", type=int, default=5000)
    p_neo.add_argument("--dry-run", action="store_true", help="print the load plan without connecting")

    p_stats = sub.add_parser("stats", help="print node / edge / literal counts of saved tables")
    p_stats.add_argument("--tables", required=True)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "build":
        result = build_and_save(load_config(args.config), write_text=not args.no_text)
        print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    elif args.command == "text":
        print(json.dumps(export_text(UKG.load_tables(args.tables), args.out, typed_relations=args.typed_relations), indent=2))
    elif args.command == "stats":
        ukg = UKG.load_tables(args.tables)
        print(ukg.schema_stats().to_string(index=False))
        print(json.dumps(ukg.summary()["nodes"], indent=2, ensure_ascii=False))
    elif args.command == "neo4j":
        ukg = UKG.load_tables(args.tables)
        if args.dry_run:
            for name, query, rows in plan(ukg):
                print(f"{name:45s} {len(rows):>8d}  {query[:100]}")
            return 0
        try:
            from dotenv import load_dotenv
            load_dotenv(args.env, override=True)
        except ImportError:
            pass
        missing = [k for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD") if not os.getenv(k)]
        if missing:
            print(f"Missing Neo4j settings: {', '.join(missing)} (set them in {args.env})", file=sys.stderr)
            return 2
        written = load_into_neo4j(
            ukg, os.environ["NEO4J_URI"], os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"],
            database=args.database or os.getenv("NEO4J_DATABASE", "neo4j"), batch_size=args.batch_size,
        )
        print(json.dumps(written, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
