import threading
from typing import List, Tuple


class AsyncBatchedEvaluator:
    """Wrap an Evaluator and coalesce many small evaluate() requests into larger batches.

    Usage: create with an existing Evaluator (e.g., BatchedEvaluator) and call
    `evaluate(states, head_ids)` as before. Calls will block until their results
    are available, but the underlying evaluator will be invoked on larger batches.
    """

    def __init__(self, underlying, max_batch_size: int = 32, timeout_ms: int = 10):
        self.underlying = underlying
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._queue = []  # list of Request

        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def evaluate(self, states: List, head_ids: List[int]) -> Tuple[List, List[float]]:
        # Single-call path: if many states provided already, forward directly
        if len(states) >= self.max_batch_size:
            return self.underlying.evaluate(states, head_ids)

        # Otherwise enqueue each state individually and wait for its result
        responses = [None] * len(states)
        events = [threading.Event() for _ in states]

        with self._lock:
            for i, (s, h) in enumerate(zip(states, head_ids)):
                self._queue.append((s, h, i, responses, events[i]))
            # notify worker
            self._cond.notify()

        # Wait for all events
        for ev in events:
            ev.wait()

        # Collect in original order
        policies = [r[0] for r in responses]
        values = [r[1] for r in responses]
        return policies, values

    def _worker_loop(self):
        while self._running:
            with self._lock:
                if not self._queue:
                    # wait until notified or timeout
                    self._cond.wait(timeout=self.timeout_ms / 1000.0)

                # Collect a batch up to max_batch_size
                if not self._queue:
                    continue

                batch = self._queue[:self.max_batch_size]
                self._queue = self._queue[len(batch):]

            # Unpack batch
            states = [item[0] for item in batch]
            head_ids = [item[1] for item in batch]

            try:
                policies, values = self.underlying.evaluate(states, head_ids)
            except Exception:
                # On failure, set None and notify waiters to avoid deadlock
                policies = [None] * len(batch)
                values = [0.0] * len(batch)

            # Dispatch results back to callers
            for (s, h, idx, responses, ev), p, v in zip(batch, policies, values):
                responses[idx] = (p, v)
                ev.set()

    def stop(self):
        self._running = False
        with self._lock:
            self._cond.notify_all()
        self._worker.join()
