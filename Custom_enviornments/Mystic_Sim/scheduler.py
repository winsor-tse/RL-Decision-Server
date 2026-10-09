"""Integer-time event heap with stable ordering and lazy cancellation."""
import heapq


class Scheduler:
    def __init__(self, world, *, max_events_at_timestamp=10000):
        if type(max_events_at_timestamp) is not int or max_events_at_timestamp < 1:
            raise ValueError("max_events_at_timestamp must be positive")
        self.world = world
        heapq.heapify(world.events)
        self.cancelled = set()
        self.max_events_at_timestamp = max_events_at_timestamp

    def schedule(self, due_ms, kind, entity_id, generation=0):
        if type(due_ms) is not int or due_ms < self.world.time_ms:
            raise ValueError("Event deadline must be integer milliseconds at or after current time")
        return self.world.enqueue(due_ms, kind, entity_id, generation)

    def cancel(self, token):
        self.cancelled.add(token)

    def advance(self, end_ms, handler):
        if type(end_ms) is not int or end_ms < self.world.time_ms:
            raise ValueError("Cannot move the clock backwards or to fractional milliseconds")
        last_time, count = None, 0
        while self.world.events and self.world.events[0].due_ms <= end_ms:
            event = heapq.heappop(self.world.events)
            if event.sequence in self.cancelled:
                self.cancelled.remove(event.sequence)
                continue
            if event.due_ms < self.world.time_ms:
                raise ValueError("Queued event is in the past")
            count = count + 1 if last_time == event.due_ms else 1
            last_time = event.due_ms
            if count > self.max_events_at_timestamp:
                raise RuntimeError(f"Too many events at {event.due_ms} ms; possible rescheduling loop")
            self.world.time_ms = event.due_ms
            handler(event)
        self.world.time_ms = end_ms
