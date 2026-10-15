"""In-memory container for a built UKG: node tables, edge tables and literals."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import schema
from .utils.geo import from_json_list, to_json_list

log = logging.getLogger(__name__)


@dataclass
class UKG:
    #: label -> DataFrame with a unique string ``id`` column plus properties.
    nodes: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: relation key (``Head__rel__Tail``) -> DataFrame(head_id, tail_id, *props).
    edges: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: Region literals: DataFrame(label, id, attribute, value).
    literals: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["label", "id", "attribute", "value"])
    )

    # ------------------------------------------------------------------ build
    def add_nodes(self, label: str, df: pd.DataFrame) -> None:
        if label not in schema.ENTITY_LAYERS:
            raise ValueError(f"Unknown entity type {label!r}")
        df = df.copy()
        df["id"] = df["id"].astype(str)
        if label in self.nodes:
            df = pd.concat([self.nodes[label], df], ignore_index=True)
        self.nodes[label] = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)

    def add_edges(self, head: str, relation: str, tail: str, df: pd.DataFrame) -> None:
        spec = schema.RELATION_BY_KEY.get(f"{head}__{relation}__{tail}")
        if spec is None:
            raise ValueError(f"({head})-[{relation}]->({tail}) is not in the unified schema")
        df = df.copy()
        df["head_id"] = df["head_id"].astype(str)
        df["tail_id"] = df["tail_id"].astype(str)
        if spec.symmetric:
            reverse = df.rename(columns={"head_id": "tail_id", "tail_id": "head_id"})
            df = pd.concat([df, reverse[df.columns]], ignore_index=True)
        if spec.key in self.edges:
            df = pd.concat([self.edges[spec.key], df], ignore_index=True)
        df = df[df["head_id"] != df["tail_id"]] if spec.head == spec.tail else df
        identity = ["head_id", "tail_id"] + schema.EDGE_KEY_PROPS.get(spec.key, [])
        self.edges[spec.key] = df.drop_duplicates(subset=identity).reset_index(drop=True)

    def add_literals(self, label: str, df: pd.DataFrame) -> None:
        """``df`` has columns id, attribute, value."""
        df = df.assign(label=label)[["label", "id", "attribute", "value"]]
        df["id"] = df["id"].astype(str)
        frames = [frame for frame in (self.literals, df) if not frame.empty]
        self.literals = pd.concat(frames, ignore_index=True) if frames else df

    # ---------------------------------------------------------------- inspect
    def node_ids(self, label: str) -> set[str]:
        return set(self.nodes[label]["id"]) if label in self.nodes else set()

    def schema_stats(self) -> pd.DataFrame:
        rows = [
            {"head_type": spec.head, "relation": spec.relation, "tail_type": spec.tail,
             "count": len(self.edges.get(spec.key, ())), "symmetric": spec.symmetric}
            for spec in schema.RELATIONS
        ]
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        return {
            "nodes": {label: len(self.nodes.get(label, ())) for label in schema.ENTITY_TYPES},
            "edges": {spec.key: len(self.edges.get(spec.key, ())) for spec in schema.RELATIONS},
            "literals": self.literals.groupby("attribute").size().to_dict() if not self.literals.empty else {},
            "total_nodes": sum(len(df) for df in self.nodes.values()),
            "total_edges": sum(len(df) for df in self.edges.values()),
            "total_literals": len(self.literals),
        }

    def validate(self) -> list[str]:
        """Return a list of problems (empty if the graph is consistent)."""
        problems = []
        for label, df in self.nodes.items():
            if df["id"].duplicated().any():
                problems.append(f"{label}: duplicated ids")
        for key, df in self.edges.items():
            spec = schema.RELATION_BY_KEY[key]
            missing_heads = set(df["head_id"]) - self.node_ids(spec.head)
            missing_tails = set(df["tail_id"]) - self.node_ids(spec.tail)
            if missing_heads:
                problems.append(f"{key}: {len(missing_heads)} head ids are not {spec.head} nodes")
            if missing_tails:
                problems.append(f"{key}: {len(missing_tails)} tail ids are not {spec.tail} nodes")
        if not self.literals.empty:
            for label, df in self.literals.groupby("label"):
                missing = set(df["id"]) - self.node_ids(label)
                if missing:
                    problems.append(f"literals: {len(missing)} ids are not {label} nodes")
        return problems

    # ------------------------------------------------------------------- I/O
    def save_tables(self, directory: str | Path) -> None:
        """Canonical CSV tables (the input of every exporter)."""
        directory = Path(directory)
        (directory / "nodes").mkdir(parents=True, exist_ok=True)
        (directory / "edges").mkdir(parents=True, exist_ok=True)
        for label, df in self.nodes.items():
            out = df.copy()
            for column in schema.LIST_COLUMNS.get(label, []):
                if column in out:
                    out[column] = out[column].apply(to_json_list)
            out.to_csv(directory / "nodes" / f"{label}.csv", index=False)
        for key, df in self.edges.items():
            df.to_csv(directory / "edges" / f"{key}.csv", index=False)
        self.literals.to_csv(directory / "literals.csv", index=False)
        self.schema_stats().to_csv(directory / "schema_stats.csv", index=False)
        (directory / "summary.json").write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False))

    @classmethod
    def load_tables(cls, directory: str | Path) -> "UKG":
        directory = Path(directory)
        ukg = cls()
        for path in sorted((directory / "nodes").glob("*.csv")):
            label = path.stem
            df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False, na_values=[""])
            for column in schema.LIST_COLUMNS.get(label, []):
                if column in df:
                    df[column] = df[column].apply(from_json_list)
            ukg.nodes[label] = df
        for path in sorted((directory / "edges").glob("*.csv")):
            ukg.edges[path.stem] = pd.read_csv(path, dtype={"head_id": str, "tail_id": str})
        literal_path = directory / "literals.csv"
        if literal_path.exists():
            ukg.literals = pd.read_csv(literal_path, dtype={"id": str})
        return ukg
