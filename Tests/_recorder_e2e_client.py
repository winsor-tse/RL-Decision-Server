import json

from websockets.sync.client import connect


with connect("ws://127.0.0.1:8765") as socket:
    socket.send(json.dumps({
        "type": "ai_tick",
        "requestId": "redo-e2e",
        "worldState": {"player": {"hp": 91}, "entities": []},
    }))
    print(socket.recv())
