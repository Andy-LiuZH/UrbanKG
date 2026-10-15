import pytest

from unified_ukg import schema
from unified_ukg.utils import address, mesh


def test_mesh_roundtrip_and_geometry():
    code = "533935931"
    row, col = mesh.mesh_to_rowcol(code)
    assert mesh.rowcol_to_mesh(row, col) == code
    poly = mesh.mesh_polygon(code)
    assert poly.bounds[2] - poly.bounds[0] == pytest.approx(22.5 / 3600)
    assert poly.bounds[3] - poly.bounds[1] == pytest.approx(15 / 3600)
    centre = poly.centroid
    assert mesh.point_to_mesh(centre.x, centre.y) == code


def test_mesh_neighbours_cross_parent_cells():
    # 533935934 is the NE quadrant; its east neighbour lies in the next 3rd-level mesh.
    assert "533935943" in mesh.neighbours("533935934")
    assert len(set(mesh.neighbours("533935931"))) == 8


def test_fine_100m_quadrants():
    assert mesh.fine_100m_to_half_mesh("5339359300") == "533935931"  # SW
    assert mesh.fine_100m_to_half_mesh("5339359309") == "533935932"  # SE
    assert mesh.fine_100m_to_half_mesh("5339359390") == "533935933"  # NW
    assert mesh.fine_100m_to_half_mesh("5339359399") == "533935934"  # NE


@pytest.mark.parametrize(
    "raw, block_key",
    [
        ("東京都　世田谷区　赤堤　１丁目　３−１５−１０３", "東京都 世田谷区 赤堤 1丁目 3"),
        ("東京都　板橋区　相生町　２１", "東京都 板橋区 相生町 21"),
        ("東京都世田谷区北沢二丁目24番3号", "東京都 世田谷区 北沢 2丁目 24"),
        ("東京都世田谷区北沢2-24-3", "東京都 世田谷区 北沢 2丁目 24"),  # chome inferred from Area names
        ("静岡県静岡市葵区与一一丁目3-5", "静岡県 静岡市葵区 与一 1丁目 3"),  # numeral inside the town name
        ("静岡県　静岡市清水区　梅ヶ谷　１２３", "静岡県 静岡市清水区 梅ケ谷 123"),  # ヶ/ケ variants unified
    ],
)
def test_block_keys(raw, block_key):
    chome_towns = {"東京都 世田谷区 北沢": {1, 2, 3, 4, 5}}
    assert address.address_block_key(address.parse_address(raw, chome_towns)) == block_key


def test_chome_only_inferred_for_existing_chome():
    chome_towns = address.chome_towns_from_area_keys(["静岡県 静岡市駿河区 大谷 1丁目", "静岡県 静岡市駿河区 大谷"])
    parsed = address.parse_address("静岡県静岡市駿河区大谷3800-1", chome_towns)
    assert address.address_block_key(parsed) == "静岡県 静岡市駿河区 大谷 3800"
    parsed = address.parse_address("静岡県静岡市駿河区大谷1-20-3", chome_towns)
    assert address.address_block_key(parsed) == "静岡県 静岡市駿河区 大谷 1丁目 20"


def test_area_key_matches_block_prefix():
    assert address.area_key("東京都", "世田谷区", "北沢二丁目") == "東京都 世田谷区 北沢 2丁目"
    assert address.standardize_town("与一一丁目") == "与一 1丁目"
    assert address.standardize_town("新町二十三丁目") == "新町 23丁目"


def test_busc_mapping_gufm_version():
    assert schema.map_busc_to_categories([3450000]) == ["Life_related", "Shopping"]  # overlap -> both
    assert schema.map_busc_to_categories([2200300]) == ["Transportation"]
    assert schema.map_busc_to_categories([2201500]) == ["others"]  # DCM would say Transportation
    assert schema.map_busc_to_categories([9000000]) == ["others"]


def test_group_ids_reproduce_dcm_order():
    assert schema.group_id_of("0-19", "女性", "その他") == 0
    assert schema.group_id_of("0-19", "女性", "医療関係者") == 4
    assert schema.group_id_of("60+", "男性", "自営業") == 107
    assert len(schema.all_groups()) == 108
