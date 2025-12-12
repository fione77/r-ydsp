# SOCRATIC METHOD CODE
# Generated after debate synthesis

import threading
import time
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity, ttl):
        self.capacity = capacity
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def _expire(self):
        with self.lock:
            now = time.time()
            self.cache = {k: v for k, v in self.cache.items() if now - v['timestamp'] < self.ttl}

    def get(self, key):
        with self.lock:
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = {'value': value['value'], 'timestamp': time.time()}
                self._expire()
                return value['value']
            else:
                return None

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = {'value': value, 'timestamp': time.time()}
            self._expire()