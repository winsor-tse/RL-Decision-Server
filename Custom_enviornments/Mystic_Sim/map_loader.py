"""Validate Tiled map53 and normalize immutable map/spawn definitions."""
import json
from pathlib import Path

from .state import MapDefinition, SpawnBox

DEFAULT_MAP_PATH = Path(__file__).with_name("data") / "map53.json"


def require(condition, path, message):
    if not condition:
        raise ValueError(f"map53 {path}: {message}")


def properties(records, path):
    require(isinstance(records, list), path, "expected property list")
    result = {}
    for i, record in enumerate(records):
        require(isinstance(record, dict) and isinstance(record.get("name"), str)
                and "value" in record, f"{path}[{i}]", "expected name/value record")
        require(record["name"] not in result, path, "duplicate property name")
        result[record["name"]] = record["value"]
    return result


def validate_map53(data):
    require(isinstance(data, dict), "$", "expected object")
    for key, value in {"width": 100, "height": 100, "tilewidth": 32,
                       "tileheight": 32, "infinite": False,
                       "orientation": "orthogonal", "type": "map"}.items():
        require(type(data.get(key)) is type(value) and data[key] == value,
                key, f"expected {value!r}")
    props = properties(data.get("properties"), "properties")
    for key, value in {"balanceCap": 68, "instanceType": "shared",
                       "name": "Severed Space", "respawnMapId": -1,
                       "respawnX": -1, "respawnY": -1, "zone": "Raid Maps"}.items():
        require(type(props.get(key)) is type(value) and props[key] == value,
                f"properties.{key}", f"expected {value!r}; review changed scenario data")
    layers = data.get("layers")
    require(isinstance(layers, list), "layers", "expected list")
    by_name = {}
    for i, layer in enumerate(layers):
        require(isinstance(layer, dict) and isinstance(layer.get("name"), str),
                f"layers[{i}]", "expected named layer")
        require(layer["name"] not in by_name, "layers", "duplicate layer name")
        by_name[layer["name"]] = layer
    for name in ("layer0", "layer1", "layer2", "blocked", "data", "render"):
        require(name in by_name, f"layers.{name}", "missing required layer")
        layer = by_name[name]
        if name in ("data", "render"):
            require(layer.get("type") == "objectgroup" and isinstance(layer.get("objects"), list),
                    f"layers.{name}", "expected objectgroup with objects")
        else:
            require(layer.get("type") == "tilelayer" and layer.get("width") == 100
                    and layer.get("height") == 100, f"layers.{name}", "expected 100x100 tilelayer")
            tiles = layer.get("data")
            require(isinstance(tiles, list) and len(tiles) == 10000
                    and all(type(t) is int and t >= 0 for t in tiles),
                    f"layers.{name}.data", "expected 10000 nonnegative integer tile IDs")
    require(sum(bool(t) for t in by_name["blocked"]["data"]) == 5088,
            "layers.blocked.data", "expected 5088 marked cells (disabled in baseline)")
    boxes, others, ids = [], [], set()
    for obj in by_name["data"]["objects"]:
        require(isinstance(obj, dict) and type(obj.get("id")) is int,
                "layers.data.objects", "expected object with integer ID")
        require(obj["id"] not in ids, "layers.data.objects", "duplicate object ID")
        ids.add(obj["id"])
        if obj.get("type") != "NPC":
            continue
        path = f"objects[{obj['id']}]"
        p = properties(obj.get("properties"), path + ".properties")
        for key in ("x", "y", "width", "height"):
            v = obj.get(key)
            require(type(v) in (int, float) and v >= 0 and v % 32 == 0,
                    path + "." + key, "expected nonnegative multiple of 32 pixels")
        require(obj.get("rotation") == 0 and obj["x"] + obj["width"] <= 3200
                and obj["y"] + obj["height"] <= 3200, path, "rotated or out-of-bounds NPC")
        if p.get("id") == 5300:
            require(p.get("fixed") is False and type(p.get("quantity")) is int
                    and p["quantity"] == 2, path, "Innie requires fixed=false and quantity=2")
            require(obj["width"] == obj["height"] == 320, path, "Innie box must be 10x10 tiles")
            boxes.append((obj["x"] // 32, obj["y"] // 32))
        else:
            require(p.get("id") == 5399 and p.get("fixed") is True
                    and p.get("quantity") == 1 and (obj["x"], obj["y"], obj["width"], obj["height"])
                    == (1600, 672, 32, 32), path, "unexpected non-Innie NPC")
            others.append(obj)
    expected = {(x, y) for x in range(10, 90, 10) for y in range(31, 72, 10)}
    require(len(boxes) == 40 and set(boxes) == expected,
            "Innie boxes", "expected exactly the 40 unique 8x5 spawn rectangles")
    require(len(others) == 1, "NPCs", "expected one fixed template-5399 NPC")
    return {"map_id": 53, "innie_boxes": 40, "innie_count": 80,
            "blocked_cells": 5088, "parsed_balance_cap": props["balanceCap"]}


def load_map53(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"map53 JSON: {exc}") from exc
    validate_map53(data)
    return data


def normalize_map53(data):
    validate_map53(data)
    layers = {layer["name"]: layer for layer in data["layers"]}
    boxes, excluded = [], []
    for obj in sorted(layers["data"]["objects"], key=lambda obj: obj["id"]):
        if obj.get("type") != "NPC":
            continue
        props = properties(obj["properties"], f"objects[{obj['id']}].properties")
        box = SpawnBox(obj["id"], props["id"], int(obj["x"] / data["tilewidth"]),
                       int(obj["y"] / data["tileheight"]), int(obj["width"] / data["tilewidth"]),
                       int(obj["height"] / data["tileheight"]), props["quantity"], props["fixed"])
        (boxes if box.template_id == 5300 else excluded).append(box)
    return MapDefinition(
        53, data["width"], data["height"], data["tilewidth"], data["tileheight"],
        tuple(sorted(properties(data["properties"], "properties").items())),
        tuple(boxes), tuple(excluded),
        frozenset((i % data["width"], i // data["width"])
                  for i, tile in enumerate(layers["blocked"]["data"]) if tile),
    )


def load_map_definition(path=DEFAULT_MAP_PATH):
    return normalize_map53(load_map53(path))
