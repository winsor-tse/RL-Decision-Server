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
single_run = False
bridge_id = uuid.uuid4().hex
active_connection_id = None
active_run_id = None
terminal_reply = None


def inert_reply(request_id, *, reason=None):
    """An inert transport reply is not a selectable policy action."""
    result = dict(terminal_reply or {})
    result.update(type='ai_result', requestId=request_id, move=None, reset=False,
                  actionId=None, serverTime=time.time())
    if reason:
        result['idleReason'] = reason
    return result


def latch_terminal(response=None, *, reason=None):
    global terminal_reply
    if terminal_reply is None:
        terminal_reply = dict(response or {}, move=None, reset=False, actionId=None,
                              stopInference=True)
        terminal_reply.setdefault('runId', active_run_id)
        if terminal_reply.get('inferenceStatus') not in ('stopped', 'interrupted'):
            terminal_reply['inferenceStatus'] = 'interrupted'
        if not terminal_reply.get('stopReason'):
            terminal_reply['stopReason'] = reason or 'bridge_interrupted'


def single_run_response(response, request_id):
    """Pin one backend run and suppress malformed/legacy/terminal commands."""
    global active_run_id
    from Custom_enviornments.Mystic_Sim.actions import ACTIONS
    if response.get('requestId') != request_id:
        latch_terminal(reason='backend_request_mismatch')
    elif response.get('error'):
        latch_terminal(reason='backend_timeout' if response['error'] == 'ZeroMQ backend timeout' else 'backend_error')
    else:
        run_id = response.get('runId')
        move = response.get('move')
        label = ':'.join(move.split(':')[:2]) if isinstance(move, str) and move.startswith('castSpell:') else move
        valid_move = move is None or (isinstance(label, str) and label in ACTIONS)
        if move is not None:
            valid_move = (valid_move and response.get('inferenceStatus') == 'action_pending'
                          and isinstance(response.get('actionId'), str) and bool(response['actionId']))
        if (not isinstance(run_id, str) or not run_id or not valid_move or response.get('reset') is not False
                or response.get('inferenceStatus') not in ('waiting_for_first_valid_state', 'running',
                                                         'action_pending', 'stopped', 'interrupted')):
            latch_terminal(reason='incompatible_backend')
        elif active_run_id is not None and run_id != active_run_id:
            latch_terminal(dict(runId=active_run_id), reason='backend_run_changed')
        else:
            active_run_id = run_id
            if response.get('stopInference') or response.get('inferenceStatus') in ('stopped', 'interrupted'):
                latch_terminal(response)
    return inert_reply(request_id) if terminal_reply is not None else response


def bridge_context(connection_id):
    return dict(schema='mystic_single_run_bridge_v1', bridgeId=bridge_id,
                connectionId=connection_id, receivedMonotonicNs=time.monotonic_ns(),
                receivedWallNs=time.time_ns())


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
    global active_connection_id
    client = websocket.remote_address
    connection_id = uuid.uuid4().hex
    if single_run and active_connection_id is None:
        active_connection_id = connection_id
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

            if single_run and terminal_reply is not None:
                response = inert_reply(request_id, reason='run_already_ended')
            elif single_run and connection_id != active_connection_id:
                response = inert_reply(request_id, reason='foreign_connection')
            else:
                forwarded = message
                if single_run:
                    # Explicit opt-in only: preserve the legacy bridge envelope
                    # by default, and overwrite client-supplied context here.
                    forwarded = dict(message, bridgeContext=bridge_context(connection_id))
                    if active_run_id is not None:
                        forwarded['runId'] = active_run_id
                response = await send_to_zmq_backend(forwarded, connection_id=connection_id)
                if single_run:
                    response = single_run_response(response, request_id)

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
        if single_run and connection_id == active_connection_id and terminal_reply is None:
            # Latch before yielding: another connection can never take over this
            # run while the backend receives the disconnect control event.
            latch_terminal(reason='client_disconnected')
            event = dict(type='bridge_event', event='client_disconnected',
                         requestId=uuid.uuid4().hex, bridgeContext=bridge_context(connection_id))
            await send_to_zmq_backend(event, connection_id=connection_id)


async def main(argv=None):
    global capture, single_run, active_connection_id, active_run_id, terminal_reply, bridge_id
    parser = argparse.ArgumentParser(description='WebSocket/ZMQ bridge with optional Phase 8A evidence capture.')
    parser.add_argument('--capture', default=os.environ.get('MYSTIC_ZMQ_CAPTURE'),
                        help='New JSONL file; existing files are never overwritten')
    parser.add_argument('--capture-label', help='Operator-provided scenario/run label (not verified execution)')
    parser.add_argument('--capture-max-records', type=int, default=10000)
    parser.add_argument('--capture-max-bytes', type=int, default=64 * 1024 * 1024)
    parser.add_argument('--capture-actions', choices=('legacy-live', 'sim-seven'),
                        help='Annotation order; default legacy-live, or sim-seven with --single-run')
    parser.add_argument('--single-run', action='store_true',
                        help='Phase 8B adapter mode: connection context, disconnect events and permanent terminal latch')
    args = parser.parse_args(argv)
    args.capture_actions = args.capture_actions or ('sim-seven' if args.single_run else 'legacy-live')
    if args.single_run and args.capture_actions != 'sim-seven':
        parser.error('--single-run uses the sim-seven action contract')
    single_run = args.single_run
    active_connection_id, active_run_id, terminal_reply = None, None, None
    bridge_id = uuid.uuid4().hex
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
