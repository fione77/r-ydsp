# DIRECT GENERATION CODE
# Generated without debate

import threading
from collections import OrderedDict
import time
from datetime import datetime, timedelta

class LRUCache:
    def __init__(self, capacity, ttl):
        self.capacity = capacity
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def _get(self, key):
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache.pop(key)
                self.cache[key] = (value, datetime.now())
                return value
            return None

    def _put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = (value, datetime.now())

    def get(self, key):
        value = self._get(key)
        if value is not None:
            return value
        return None

    def put(self, key, value):
        self._put(key, value)

    def _evict_expired(self):
        with self.lock:
            now = datetime.now()
            self.cache = {k: v for k, v in self.cache.items() if now - v[1] < self.ttl}

    def _run_evict_expired(self):
        while True:
            self._evict_expired()
            time.sleep(self.ttl.total_seconds())

    def start(self):
        threading.Thread(target=self._run_evict_expired).start()

    def stop(self):
        self.lock.acquire()
        self.cache.clear()
        self.lock.release()