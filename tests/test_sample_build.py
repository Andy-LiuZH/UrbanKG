"""Build the bundled sample end to end and check the result against the schema."""

import json
from pathlib import Path

import pytest

from unified_ukg import schema
from unified_ukg.config import load_config
from unified_ukg.export.neo4j import plan
from unified_ukg.export.text import export_text
from unified_ukg.graph import UKG
from unified_ukg.pipeline import build

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sample_ukg():
    ukg, report = build(load_config(ROOT / "configs/setagaya_sample.yaml"))
    return ukg, report


def test_every_entity_and_relation_present(sample_ukg):
    ukg, _ = sample_ukg
    assert ukg.validate() == []
    for label in schema.ENTITY_TYPES:
        assert len(ukg.nodes.get(label, ())) > 0, label
    for spec in schema.RELATIONS:
        assert len(ukg.edges.get(spec.key, ())) > 0, spec.key


def test_symmetric_relations_are_written_both_ways(sample_ukg):
    ukg, _ = sample_ukg
    for spec in schema.RELATIONS:
        if spec.symmetric:
            df = ukg.edges[spec.key]
            pairs = set(zip(df["head_id"], df["tail_id"]))
            assert all((t, h) in pairs for h, t in pairs), spec.key


def test_block_has_no_geometry(sample_ukg):
    ukg, _ = sample_ukg
    assert set(ukg.nodes["Block"].columns) == {"id", "address"}


def test_station_merged_across_lines(sample_ukg):
    ukg, _ = sample_ukg
    poi = ukg.nodes["POI"]
    shimokitazawa = poi[poi["name"] == "下北沢"]
    assert len(shimokitazawa) == 1
    on_line = ukg.edges["POI__onLine__Road"]
    assert (on_line["head_id"] == shimokitazawa["id"].iloc[0]).sum() == 2


def test_zenrin_grouping_and_station_filter(sample_ukg):
    ukg, _ = sample_ukg
    zenrin = ukg.nodes["POI"].query("source == 'zenrin'")
    assert len(zenrin) == 40  # 42 rows: one duplicate row grouped, one PTYPE 22 row dropped
    assert zenrin["busc_codes"].map(len).max() == 2


def test_region_literals_complete(sample_ukg):
    ukg, _ = sample_ukg
    per_region = ukg.literals.groupby("id")["attribute"].nunique()
    assert (per_region == len(schema.REGION_ATTRIBUTES)).all()
    landuse = ukg.literals[ukg.literals["attribute"].str.startswith(schema.ATTR_LANDUSE_PREFIX)]
    assert (landuse.groupby("id")["value"].sum() <= 1.0 + 1e-6).all()


def test_tables_roundtrip_and_text(sample_ukg, tmp_path):
    ukg, _ = sample_ukg
    ukg.save_tables(tmp_path / "tables")
    loaded = UKG.load_tables(tmp_path / "tables")
    assert loaded.summary()["edges"] == ukg.summary()["edges"]
    stats = export_text(loaded, tmp_path / "text")
    assert stats["relations"] == len(schema.RELATION_NAMES)
    assert stats["train"] + stats["valid"] + stats["test"] == stats["triples"]
    lines = (tmp_path / "text/UrbanKG.txt").read_text(encoding="utf-8").splitlines()
    assert all(len(line.split()) == 3 for line in lines)


def test_neo4j_plan_covers_everything(sample_ukg):
    ukg, _ = sample_ukg
    steps = {name: rows for name, _, rows in plan(ukg)}
    for label in schema.ENTITY_TYPES:
        assert len(steps[f"node:{label}"]) == len(ukg.nodes[label])
    for spec in schema.RELATIONS:
        assert len(steps[f"edge:{spec.key}"]) == len(ukg.edges[spec.key])
    assert len(steps["literal:Region"]) == len(ukg.nodes["Region"])
    json.dumps(steps)  # every value must be a plain, driver-serialisable Python value
