"""Japanese standard grid square (JIS X 0410) helpers for 1/2 regional meshes.

A 9-digit 4th-level mesh code (about 500 m) is handled through integer grid
indices so that adjacency and geometry are exact:

* row = latitude  / 15"    (south -> north)
* col = (longitude - 100°) / 22.5"  (west -> east)
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from shapely.geometry import Polygon, box

LAT_STEP = 15.0 / 3600.0  # degrees per 1/2 mesh row
LON_STEP = 22.5 / 3600.0  # degrees per 1/2 mesh column


def mesh_to_rowcol(code: str | int) -> tuple[int, int]:
    text = str(code).strip()
    if len(text) != 9 or not text.isdigit():
        raise ValueError(f"Expected a 9-digit 1/2 mesh code, got {code!r}")
    ab, cd = int(text[0:2]), int(text[2:4])
    e, f = int(text[4]), int(text[5])
    g, h = int(text[6]), int(text[7])
    quadrant = int(text[8])
    if quadrant not in (1, 2, 3, 4) or e > 7 or f > 7:
        raise ValueError(f"Invalid 1/2 mesh code {code!r}")
    north = 1 if quadrant in (3, 4) else 0
    east = 1 if quadrant in (2, 4) else 0
    return ab * 160 + e * 20 + g * 2 + north, cd * 160 + f * 20 + h * 2 + east


def rowcol_to_mesh(row: int, col: int) -> str:
    ab, rest = divmod(row, 160)
    e, rest = divmod(rest, 20)
    g, north = divmod(rest, 2)
    cd, rest_c = divmod(col, 160)
    f, rest_c = divmod(rest_c, 20)
    h, east = divmod(rest_c, 2)
    return f"{ab:02d}{cd:02d}{e}{f}{g}{h}{1 + east + 2 * north}"


def mesh_polygon(code: str | int) -> Polygon:
    row, col = mesh_to_rowcol(code)
    south, west = row * LAT_STEP, 100.0 + col * LON_STEP
    return box(west, south, west + LON_STEP, south + LAT_STEP)


def point_to_mesh(lon: float, lat: float) -> str:
    return rowcol_to_mesh(math.floor(lat / LAT_STEP), math.floor((lon - 100.0) / LON_STEP))


def meshes_covering(bounds: tuple[float, float, float, float]) -> list[str]:
    """All 1/2 meshes whose cell overlaps a lon/lat bounding box."""
    min_lon, min_lat, max_lon, max_lat = bounds
    rows = range(math.floor(min_lat / LAT_STEP), math.floor(max_lat / LAT_STEP) + 1)
    cols = range(math.floor((min_lon - 100.0) / LON_STEP), math.floor((max_lon - 100.0) / LON_STEP) + 1)
    return [rowcol_to_mesh(r, c) for r in rows for c in cols]


def neighbours(code: str | int) -> list[str]:
    """The 8 surrounding meshes (edge and corner neighbours)."""
    row, col = mesh_to_rowcol(code)
    return [
        rowcol_to_mesh(row + dr, col + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr or dc
    ]


def neighbour_pairs(codes: Iterable[str]) -> list[tuple[str, str]]:
    """Unordered pairs (a < b) of 8-neighbouring meshes within ``codes``."""
    code_set = {str(c) for c in codes}
    pairs = set()
    for code in code_set:
        for other in neighbours(code):
            if other in code_set:
                pairs.add(tuple(sorted((code, other))))
    return sorted(pairs)


def fine_100m_to_half_mesh(code: str | int) -> str:
    """Map a 10-digit 100 m subdivision code (L03-b-u) to its 1/2 mesh code.

    The last two digits are the row (south -> north) and column (west -> east)
    inside the 3rd-level mesh, each 0-9.
    """
    text = str(code).strip()
    if len(text) != 10 or not text.isdigit():
        raise ValueError(f"Expected a 10-digit 100 m mesh code, got {code!r}")
    sub_row, sub_col = int(text[8]), int(text[9])
    quadrant = 1 + (1 if sub_col >= 5 else 0) + (2 if sub_row >= 5 else 0)
    return f"{text[:8]}{quadrant}"
