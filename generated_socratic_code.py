# SOCRATIC METHOD CODE
# Generated after debate synthesis

import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta

class LRUCache:
    def __init__(self, capacity, ttl):
        self.capacity = capacity
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp > timedelta(seconds=self.ttl):
                del self.cache[key]
                return None
            self.cache.move_to_end(key)
            return value

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            elif len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = (value, datetime.now())
            self.cache.move_to_end(key)

    def clear(self):
        with self.lock:
            self.cache.clear()

    def __len__(self):
        with self.lock:
            return len(self.cache)

# Example usage: