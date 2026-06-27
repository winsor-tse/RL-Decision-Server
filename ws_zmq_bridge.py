import asyncio
import json
import logging
from typing import Any, Dict, Optional

import zmq
import zmq.asyncio
import websockets
from websockets.exceptions import ConnectionClosed

WS_HOST = "127.0.0.1"
WS_PORT = 8765

ZMQ_BACKEND_URL = "tcp://127.0.0.1:5555"
ZMQ_TIMEOUT_SECONDS = 5

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)

zmq_context = zmq.asyncio.Context.instance()
zmq_lock = asyncio.Lock()

zmq_socket: Optional[zmq.asyncio.Socket] = None


def create_zmq_socket() -> zmq.asyncio.Socket:
    socket = zmq_context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(ZMQ_BACKEND_URL)

    logging.info("ZeroMQ REQ socket connected to %s", ZMQ_BACKEND_URL)

    return socket


def reset_zmq_socket() -> None:
    global zmq_socket

    if zmq_socket is not None:
        try:
            zmq_socket.close(linger=0)
        except Exception:
            logging.exception("Failed to close old ZeroMQ socket")

    zmq_socket = create_zmq_socket()


async def send_to_zmq_backend(message: Dict[str, Any]) -> Dict[str, Any]:
    global zmq_socket

    request_id = message.get("requestId")

    async with zmq_lock:
        if zmq_socket is None:
            reset_zmq_socket()

        try:
            logging.info("Sending to ZeroMQ requestId=%s", request_id)

            await zmq_socket.send_json(message)

            reply = await asyncio.wait_for(
                zmq_socket.recv_json(),
                timeout=ZMQ_TIMEOUT_SECONDS
            )

            if not isinstance(reply, dict):
                return {
                    "requestId": request_id,
                    "move": None,
                    "reset": False,
                    "error": "ZeroMQ backend returned non-object reply"
                }

            if request_id and "requestId" not in reply:
                reply["requestId"] = request_id

            logging.info("Received ZeroMQ reply requestId=%s", request_id)

            return reply

        except asyncio.TimeoutError:
            logging.error("ZeroMQ timeout waiting for reply requestId=%s", request_id)

            # Important:
            # A REQ socket that timed out while waiting for recv is now poisoned.
            # Reset it before the next request.
            reset_zmq_socket()

            return {
                "requestId": request_id,
                "move": None,
                "reset": False,
                "error": "ZeroMQ backend timeout"
            }

        except zmq.ZMQError as error:
            logging.error(
                "ZeroMQ socket error requestId=%s error=%s",
                request_id,
                error
            )

            # This fixes:
            # Operation cannot be accomplished in current state
            reset_zmq_socket()

            return {
                "requestId": request_id,
                "move": None,
                "reset": False,
                "error": f"ZeroMQ socket error: {error}"
            }

        except Exception as error:
            logging.exception("Unexpected bridge error requestId=%s", request_id)

            reset_zmq_socket()

            return {
                "requestId": request_id,
                "move": None,
                "reset": False,
                "error": str(error)
            }


async def handle_websocket(websocket):
    client = websocket.remote_address
    logging.info("Chrome extension connected: %s", client)

    try:
        async for raw_message in websocket:
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "requestId": None,
                    "move": None,
                    "reset": False,
                    "error": "Invalid JSON received by WebSocket bridge"
                }))
                continue

            request_id = message.get("requestId")
            message_type = message.get("type")

            if message_type == "ping":
                await websocket.send(json.dumps({
                    "type": "pong",
                    "requestId": request_id
                }))
                continue

            if message_type != "ai_tick":
                await websocket.send(json.dumps({
                    "requestId": request_id,
                    "move": None,
                    "reset": False,
                    "error": f"Unknown WebSocket message type: {message_type}"
                }))
                continue

            response = await send_to_zmq_backend(message)

            await websocket.send(json.dumps(response))

    except ConnectionClosed:
        logging.info("Chrome extension disconnected: %s", client)

    except Exception:
        logging.exception("WebSocket handler crashed")


async def main():
    reset_zmq_socket()

    logging.info("WebSocket bridge listening on ws://%s:%s", WS_HOST, WS_PORT)
    logging.info("Forwarding to ZeroMQ backend at %s", ZMQ_BACKEND_URL)

    async with websockets.serve(handle_websocket, WS_HOST, WS_PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())