# SOCRATIC METHOD CODE
# Generated after debate synthesis

import threading
import time
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int, ttl: int):
        self.capacity = capacity
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> any:
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache.pop(key)
                if timestamp + self.ttl > time.time():
                    self.cache[key] = (value, timestamp + self.ttl)
                    return value
        return None

    def put(self, key: str, value: any):
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = (value, time.time())

    def expire(self, key: str):
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)

    def reset(self):
        with self.lock:
            self.cache.clear()