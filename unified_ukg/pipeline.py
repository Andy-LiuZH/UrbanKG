"""End-to-end build: raw sources -> UKG -> tables / text."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .builders.activity import build_activity
from .builders.attributes import build_attributes
from .builders.context import BuildContext
from .builders.spatial import build_spatial
from .builders.urban import build_urban
from .config import Config
from .export.text import export_text
from .graph import UKG

log = logging.getLogger(__name__)


def build(config: Config) -> tuple[UKG, dict]:
    ctx = BuildContext(config=config)
    build_spatial(ctx)   # Ward, Area, Region + spatial relations
    build_urban(ctx)     # Road, POI, Block, Cate + urban/functional relations
    build_attributes(ctx)  # Region literals (needs POI-Region-Cate)
    build_activity(ctx)  # Purpose, Group, hasOD, attracts (needs land use)
    problems = ctx.ukg.validate()
    if problems:
        raise RuntimeError("UKG validation failed:\n" + "\n".join(problems))
    return ctx.ukg, ctx.report


def build_and_save(config: Config, write_text: bool = True) -> dict:
    ukg, report = build(config)
    out = Path(config.output_dir)
    ukg.save_tables(out / "tables")
    result = {"config": config.name, "build_report": report, "summary": ukg.summary()}
    if write_text:
        result["text"] = export_text(
            ukg, out / "text", split=config.param("split"),
            typed_relations=bool(config.params.get("text_typed_relations", False)),
        )
    (out / "build_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info("UKG written to %s", out)
    return result
