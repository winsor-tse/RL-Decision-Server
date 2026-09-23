"""Migrate exported JSON episodes, never edit a Minari dataset in place.

Usage: python -m Offline.remap_mystic_bc legacy.json converted.json
Input: {"dataset_id": "mystic/BC-v0", "actions": [...legacy labels...],
        "episodes": [{"actions": [0, 1, ...], ...}]}
Episode fields are preserved. Removed spells reject the entire conversion.
"""
import argparse
import json
from pathlib import Path

from Custom_enviornments.Mystic_Sim.actions import (
    DATASET_ID, LEGACY_BC_ACTIONS, contract_metadata, remap_legacy_bc,
)


def migrate(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("Destination must be a new file; never overwrite a dataset")
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("dataset_id") != "mystic/BC-v0" or data.get("actions") != list(LEGACY_BC_ACTIONS):
        raise ValueError("Expected explicit mystic/BC-v0 legacy BC action metadata")
    if not isinstance(data.get("episodes"), list):
        raise ValueError("Expected episodes list")
    for episode in data["episodes"]:
        if not isinstance(episode, dict) or not isinstance(episode.get("actions"), list):
            raise ValueError("Each episode requires an actions list")
        episode["actions"] = remap_legacy_bc(episode["actions"])
    data.update(dataset_id=DATASET_ID, source_dataset_id="mystic/BC-v0",
                **contract_metadata())
    with destination.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    migrate(args.source, args.destination)
