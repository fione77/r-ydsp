# DIRECT GENERATION CODE
# Generated without debate

import threading
import time
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity, ttl):
        self.capacity = capacity
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache.pop(key)
                self.cache[key] = (value, time.time())
                return value
            else:
                return None

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
            self.cache[key] = (value, time.time())
            self._expire()

    def _expire(self):
        now = time.time()
        with self.lock:
            self.cache = {k: v for k, v in self.cache.items() if now - v[1] < self.ttl}