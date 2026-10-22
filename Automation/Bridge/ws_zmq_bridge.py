import asyncio
import argparse
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

import zmq
import zmq.asyncio
import websockets
from websockets.exceptions import ConnectionClosed
from Automation.Bridge.evidence import JsonlCapture

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
capture: Optional[JsonlCapture] = None


def record_evidence(event, **fields):
    if capture is not None:
        capture.record(event, **fields)


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


async def send_to_zmq_backend(message: Dict[str, Any], *, connection_id=None) -> Dict[str, Any]:
    global zmq_socket

    request_id = message.get("requestId")

    async with zmq_lock:
        if zmq_socket is None:
            reset_zmq_socket()

        try:
            logging.info("Sending to ZeroMQ requestId=%s", request_id)

            await zmq_socket.send_json(message)
            record_evidence('backend_sent', connection_id=connection_id,
                            request_id=request_id, request=message)

            reply = await asyncio.wait_for(
                zmq_socket.recv_json(),
                timeout=ZMQ_TIMEOUT_SECONDS
            )
            record_evidence('backend_reply', connection_id=connection_id,
                            request_id=request_id, response=reply,
                            command=capture.command(reply) if capture else None)

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
            record_evidence('backend_timeout', connection_id=connection_id, request_id=request_id)
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
            record_evidence('backend_error', connection_id=connection_id,
                            request_id=request_id, error=str(error))
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
            record_evidence('backend_error', connection_id=connection_id,
                            request_id=request_id, error=str(error))
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
    connection_id = uuid.uuid4().hex
    record_evidence('client_connected', connection_id=connection_id, remote_address=client)
    logging.info("[JS CONNECTED] Chrome extension client=%s", client)

    try:
        async for raw_message in websocket:
            received_wall_ns, received_monotonic_ns = time.time_ns(), time.monotonic_ns()
            logging.info(
                "[JS MESSAGE] client=%s bytes=%s",
                client,
                len(raw_message),
            )
            try:
                message = json.loads(raw_message)
            except (json.JSONDecodeError, UnicodeDecodeError):
                record_evidence('malformed_message', connection_id=connection_id,
                                raw_message=raw_message if isinstance(raw_message, str) else raw_message.hex(),
                                raw_encoding='text' if isinstance(raw_message, str) else 'hex')
                logging.warning("[JS MESSAGE] Invalid JSON from client=%s", client)
                await websocket.send(json.dumps({
                    "requestId": None,
                    "move": None,
                    "reset": False,
                    "error": "Invalid JSON received by WebSocket bridge"
                }))
                continue

            if not isinstance(message, dict):
                record_evidence('malformed_message', connection_id=connection_id, parsed_message=message)
                await websocket.send(json.dumps({'requestId': None, 'move': None, 'reset': False,
                                                  'error': 'WebSocket message must be a JSON object'}))
                continue
            record_evidence('client_request', connection_id=connection_id,
                            request_id=message.get('requestId'), request=message,
                            bridge_received_wall_ns=received_wall_ns,
                            bridge_received_monotonic_ns=received_monotonic_ns)
            request_id = message.get("requestId")
            message_type = message.get("type")
            logging.info(
                "[JS MESSAGE] type=%r requestId=%r hasWorldState=%s",
                message_type,
                request_id,
                "worldState" in message,
            )

            if message_type == "ping":
                await websocket.send(json.dumps({
                    "type": "pong",
                    "requestId": request_id
                }))
                continue

            if message_type != "ai_tick":
                logging.warning(
                    "[JS MESSAGE REJECTED] expected='ai_tick' received=%r",
                    message_type,
                )
                await websocket.send(json.dumps({
                    "requestId": request_id,
                    "move": None,
                    "reset": False,
                    "error": f"Unknown WebSocket message type: {message_type}"
                }))
                continue

            response = await send_to_zmq_backend(message, connection_id=connection_id)

            await websocket.send(json.dumps(response))
            record_evidence('client_response_sent', connection_id=connection_id,
                            request_id=request_id, response=response,
                            command=capture.command(response) if capture else None)

    except ConnectionClosed:
        logging.info("[JS DISCONNECTED] Chrome extension client=%s", client)

    except Exception:
        logging.exception("WebSocket handler crashed")
    finally:
        record_evidence('client_disconnected', connection_id=connection_id)


async def main(argv=None):
    global capture
    parser = argparse.ArgumentParser(description='WebSocket/ZMQ bridge with optional Phase 8A evidence capture.')
    parser.add_argument('--capture', default=os.environ.get('MYSTIC_ZMQ_CAPTURE'),
                        help='New JSONL file; existing files are never overwritten')
    parser.add_argument('--capture-label', help='Operator-provided scenario/run label (not verified execution)')
    parser.add_argument('--capture-max-records', type=int, default=10000)
    parser.add_argument('--capture-max-bytes', type=int, default=64 * 1024 * 1024)
    parser.add_argument('--capture-actions', choices=('legacy-live', 'sim-seven'), default='legacy-live',
                        help='Explicit action ordering for index annotations; does not change the backend')
    args = parser.parse_args(argv)
    if min(args.capture_max_records, args.capture_max_bytes) <= 0:
        parser.error('Capture limits must be positive')
    if args.capture:
        from Custom_enviornments.Mystic_Sim.actions import ACTIONS, LEGACY_LIVE_ACTIONS
        try:
            capture = JsonlCapture(args.capture, label=args.capture_label,
                                   action_labels=ACTIONS if args.capture_actions == 'sim-seven' else LEGACY_LIVE_ACTIONS,
                                   max_records=args.capture_max_records, max_bytes=args.capture_max_bytes)
        except (OSError, ValueError) as error:
            parser.error(f'Cannot start capture: {error}')
        logging.info('Phase 8A evidence capture: %s (action annotations: %s)', args.capture, args.capture_actions)

    try:
        reset_zmq_socket()
        logging.info("WebSocket bridge listening on ws://%s:%s", WS_HOST, WS_PORT)
        logging.info("Forwarding to ZeroMQ backend at %s", ZMQ_BACKEND_URL)
        logging.info("Waiting for [JS CONNECTED], then [JS MESSAGE] type='ai_tick'.")
        async with websockets.serve(handle_websocket, WS_HOST, WS_PORT):
            await asyncio.Future()
    finally:
        if capture is not None:
            capture.close()
            logging.info('Phase 8A capture summary: %s', capture.status())
            capture = None


if __name__ == "__main__":
    asyncio.run(main())
