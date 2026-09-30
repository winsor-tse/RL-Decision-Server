"""Checkpoint/dataset envelopes for eight-action producers in later phases."""
from .actions import contract_metadata, validate_contract


def checkpoint_envelope(state_dict):
    return {"format_version": 1, "contract": contract_metadata(), "state_dict": state_dict}


def checkpoint_state(envelope):
    """Validate before calling model.load_state_dict; never guess legacy layout."""
    if not isinstance(envelope, dict) or envelope.get("format_version") != 1:
        raise ValueError("Expected versioned Mystic checkpoint; legacy weights require explicit migration")
    validate_contract(envelope.get("contract"))
    if "state_dict" not in envelope:
        raise ValueError("Checkpoint missing state_dict")
    return envelope["state_dict"]
