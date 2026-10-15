"""Load a built UKG into Neo4j.

The loader reads the canonical tables (``<output>/tables``) and writes with
``MERGE`` in batches, so re-running it is idempotent. Region literals become
Region node properties, list columns become arrays, ``geometry`` is a WKT
string and nodes with lon/lat also get a ``location`` point.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pandas as pd

from .. import schema
from ..graph import UKG

log = logging.getLogger(__name__)


def _clean(value: Any) -> Any:
    """Plain Python values only: no NaN / pd.NA / numpy scalars."""
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for record in df.to_dict("records"):
        rows.append({k: v for k, v in ((k, _clean(v)) for k, v in record.items()) if v is not None})
    return rows


def _batches(rows: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def constraint_queries() -> list[str]:
    return [
        f"CREATE CONSTRAINT ukg_{label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
        for label in schema.ENTITY_TYPES
    ]


def node_query(label: str, with_location: bool) -> str:
    location = (
        ", n.location = CASE WHEN row.props.lon IS NULL OR row.props.lat IS NULL THEN null "
        "ELSE point({longitude: row.props.lon, latitude: row.props.lat}) END"
        if with_location else ""
    )
    return f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row.props{location}"


def edge_query(spec: schema.RelationSpec) -> str:
    keys = schema.EDGE_KEY_PROPS.get(spec.key, [])
    key_map = ", ".join(f"{k}: row.props.{k}" for k in keys)
    rel = f"[r:{spec.relation} {{{key_map}}}]" if keys else f"[r:{spec.relation}]"
    return (
        f"UNWIND $rows AS row "
        f"MATCH (h:{spec.head} {{id: row.head_id}}) MATCH (t:{spec.tail} {{id: row.tail_id}}) "
        f"MERGE (h)-{rel}->(t) SET r += row.props"
    )


def literal_query(label: str) -> str:
    return f"UNWIND $rows AS row MATCH (n:{label} {{id: row.id}}) SET n += row.props"


def plan(ukg: UKG) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """(step name, query, rows) in execution order, without touching a database."""
    steps: list[tuple[str, str, list[dict[str, Any]]]] = [("constraint", q, []) for q in constraint_queries()]
    for label in schema.ENTITY_TYPES:
        df = ukg.nodes.get(label)
        if df is None or df.empty:
            continue
        rows = [{"id": r.pop("id"), "props": r} for r in _records(df)]
        steps.append((f"node:{label}", node_query(label, {"lon", "lat"} <= set(df.columns)), rows))
    if not ukg.literals.empty:
        for label, group in ukg.literals.groupby("label"):
            wide = group.pivot_table(index="id", columns="attribute", values="value", aggfunc="first")
            rows = [{"id": node_id, "props": {k: v for k, v in values.items() if not pd.isna(v)}}
                    for node_id, values in wide.to_dict("index").items()]
            steps.append((f"literal:{label}", literal_query(label), rows))
    for spec in schema.RELATIONS:
        df = ukg.edges.get(spec.key)
        if df is None or df.empty:
            continue
        rows = []
        for record in _records(df):
            head, tail = record.pop("head_id"), record.pop("tail_id")
            rows.append({"head_id": str(head), "tail_id": str(tail), "props": record})
        steps.append((f"edge:{spec.key}", edge_query(spec), rows))
    return steps


def load_into_neo4j(ukg: UKG, uri: str, user: str, password: str, database: str = "neo4j",
                    batch_size: int = 5000) -> dict[str, int]:
    from neo4j import GraphDatabase

    written: dict[str, int] = {}
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            for name, query, rows in plan(ukg):
                if not rows:
                    session.run(query).consume()
                    continue
                for batch in _batches(rows, batch_size):
                    session.execute_write(lambda tx, q=query, b=batch: tx.run(q, rows=b).consume())
                written[name] = len(rows)
                log.info("neo4j: %-40s %d rows", name, len(rows))
    return written
