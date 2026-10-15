"""Text export in the UUKG / DCM-UrbanKG format for KG embedding.

Files written to ``<out>/``::

    entity2id.txt     <entity>\t<id>          entity = <Type>/<node id>
    relation2id.txt   <relation>\t<id>
    UrbanKG.txt       <head> <relation> <tail>  (names)
    triplets.txt      <head id> <relation id> <tail id>
    train.txt / valid.txt / test.txt   (ids, shuffled split)
    literal.txt       <entity> <attribute> <value>
    literal_schema.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .. import schema
from ..graph import UKG


def entity_name(label: str, node_id: str) -> str:
    return f"{label}/{node_id}"


def relation_name(spec: schema.RelationSpec, typed: bool) -> str:
    return f"{spec.head}_{spec.relation}_{spec.tail}" if typed else spec.relation


def collect_triples(ukg: UKG, typed_relations: bool = False) -> list[tuple[str, str, str]]:
    triples = []
    seen = set()
    for spec in schema.RELATIONS:
        df = ukg.edges.get(spec.key)
        if df is None:
            continue
        relation = relation_name(spec, typed_relations)
        for head, tail in zip(df["head_id"], df["tail_id"]):
            triple = (entity_name(spec.head, head), relation, entity_name(spec.tail, tail))
            if triple not in seen:  # parallel edges (e.g. nextTo per line) collapse
                seen.add(triple)
                triples.append(triple)
    return triples


def export_text(ukg: UKG, out_dir: str | Path, split: dict | None = None, typed_relations: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = split or {"train": 0.9, "valid": 0.05, "seed": 2026}

    entities = [
        entity_name(label, node_id)
        for label in schema.ENTITY_TYPES
        for node_id in sorted(ukg.nodes.get(label, {"id": []})["id"])
    ]
    entity2id = {name: i for i, name in enumerate(entities)}
    relations = sorted({relation_name(spec, typed_relations) for spec in schema.RELATIONS})
    relation2id = {name: i for i, name in enumerate(relations)}

    triples = collect_triples(ukg, typed_relations)
    id_triples = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples]

    _write(out / "entity2id.txt", (f"{name}\t{i}" for name, i in entity2id.items()))
    _write(out / "relation2id.txt", (f"{name}\t{i}" for name, i in relation2id.items()))
    _write(out / "UrbanKG.txt", (" ".join(t) for t in triples))
    _write(out / "triplets.txt", (f"{h} {r} {t}" for h, r, t in id_triples))

    shuffled = id_triples[:]
    random.Random(split["seed"]).shuffle(shuffled)
    n_train = int(len(shuffled) * split["train"])
    n_valid = int(len(shuffled) * split["valid"])
    parts = {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_valid],
        "test": shuffled[n_train + n_valid:],
    }
    for name, rows in parts.items():
        _write(out / f"{name}.txt", (f"{h} {r} {t}" for h, r, t in rows))

    literals = ukg.literals
    _write(out / "literal.txt", (
        f"{entity_name(label, node_id)} {attribute} {value}"
        for label, node_id, attribute, value in literals[["label", "id", "attribute", "value"]].itertuples(index=False)
    ))
    present = set(literals["attribute"]) if not literals.empty else set()
    literal_schema = {
        "entity_type": "Region",
        "attribute_order": [a for a in schema.REGION_ATTRIBUTES if a in present],
        "origin_attributes": [a for a in schema.ORIGIN_ATTRIBUTES if a in present],
        "destination_attributes": [a for a in schema.DESTINATION_ATTRIBUTES if a in present],
    }
    (out / "literal_schema.json").write_text(json.dumps(literal_schema, indent=2, ensure_ascii=False))

    return {
        "entities": len(entity2id),
        "relations": len(relation2id),
        "triples": len(id_triples),
        **{name: len(rows) for name, rows in parts.items()},
        "literals": len(literals),
    }


def _write(path: Path, lines) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
