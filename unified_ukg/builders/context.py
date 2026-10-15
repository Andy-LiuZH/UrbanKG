"""State shared between builders during one build."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from ..config import Config
from ..graph import UKG


@dataclass
class BuildContext:
    config: Config
    ukg: UKG = field(default_factory=UKG)
    #: Union of the study area (storage CRS).
    study_area: BaseGeometry | None = None
    #: Working GeoDataFrames (storage CRS) keyed by entity type.
    frames: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)
    #: Every Area key of the loaded town file, used to detect chome towns.
    all_area_keys: list[str] = field(default_factory=list)
    #: Region -> land-use type -> share, filled by the attribute builder.
    landuse_ratio: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Build log: counts and dropped rows per step.
    report: dict[str, Any] = field(default_factory=dict)

    def note(self, step: str, **values: Any) -> None:
        self.report.setdefault(step, {}).update(values)
