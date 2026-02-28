from rclpy.node import Node

class LogThrottler:
    def __init__(self, node: Node, default_ms: int = 1000):
        self._node = node
        self._default_ms = int(default_ms)
        self._last_ns = {}

    def should_log(self, key: str, throttle_ms: int = None):
        if throttle_ms is None:
            throttle_ms = self._default_ms
        throttle_ns = int(throttle_ms) * 1_000_000
        now_ns = self._node.get_clock().now().nanoseconds
        last_ns = self._last_ns.get(key)
        if last_ns is None or (now_ns - last_ns) >= throttle_ns:
            self._last_ns[key] = now_ns
            return True
        return False
