"""YAML configuration: data sources, study area, parameters and output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PARAMS: dict[str, Any] = {
    "poi_road_max_distance_m": 20.0,
    "region_nearby_max_distance_m": 3000.0,
    "station_merge_distance_m": 500.0,
    "loop_lines": ["山手線"],
    "landuse_attract_threshold": 0.2,
    "survey_reference_year": 2023,
    "split": {"train": 0.9, "valid": 0.05, "seed": 2026},
}


@dataclass
class Config:
    name: str
    root: Path
    study_area: dict[str, Any]
    sources: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)
    output_dir: Path = Path("output")

    def source(self, key: str) -> dict[str, Any] | None:
        """Source entry with its path(s) resolved, or None if not configured."""
        entry = self.sources.get(key)
        if entry is None:
            return None
        return self.resolve_entry(entry)

    def resolve_entry(self, entry: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(entry, str):
            entry = {"path": entry}
        entry = dict(entry)
        for path_key in ("path", "dir"):
            if path_key in entry:
                entry[path_key] = str(self.resolve(entry[path_key]))
        if "paths" in entry:
            entry["paths"] = [str(self.resolve(p)) for p in entry["paths"]]
        return entry

    def resolve(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()

    def param(self, key: str) -> Any:
        return self.params.get(key, DEFAULT_PARAMS[key])


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Relative paths are resolved against the project root (parent of configs/).
    root = Path(raw.get("root", path.parent.parent))
    root = root if root.is_absolute() else (path.parent / root).resolve()
    config = Config(
        name=raw["name"],
        root=root,
        study_area=raw.get("study_area", {}),
        sources=raw.get("sources", {}),
        params={**DEFAULT_PARAMS, **(raw.get("params") or {})},
    )
    config.output_dir = config.resolve(raw.get("output_dir", f"output/{config.name}"))
    return config
