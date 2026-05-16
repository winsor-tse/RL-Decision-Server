import random
import time
from typing import Any, Dict

import zmq

ZMQ_BIND_URL = "tcp://127.0.0.1:5555"


def decide_move(world_state: Dict[str, Any]) -> str:
    return random.choice(["up", "down", "left", "right", "direction:up", "direction:down", "direction:left", "direction:right", "attack", "castSpell:1", "castSpell:2", "castSpell:3"])

def should_reset(world_state: Dict[str, Any]) -> bool:
    player_hp = world_state.get("playerHp", 100)
    return player_hp <= 0


def handle_ai_tick(message: Dict[str, Any]) -> Dict[str, Any]:
    world_state = message.get("worldState", {})

    move = decide_move(world_state)
    reset = should_reset(world_state)

    return {
        "type": "ai_result",
        "requestId": message.get("requestId"),
        "move": move,
        "reset": reset,
        "serverTime": time.time()
    }


def main():
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(ZMQ_BIND_URL)

    print(f"ZeroMQ AI backend listening at {ZMQ_BIND_URL}")

    while True:
        message = socket.recv_json()
        print("Received:", message)

        message_type = message.get("type")

        if message_type == "ai_tick":
            response = handle_ai_tick(message)
        else:
            response = {
                "type": "error",
                "requestId": message.get("requestId"),
                "error": f"Unknown message type: {message_type}"
            }

        socket.send_json(response)


if __name__ == "__main__":
    main()