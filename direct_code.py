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

    def _expire(self):
        with self.lock:
            now = time.time()
            self.cache = {k: v for k, v in self.cache.items() if v['expires'] > now}

    def _update_expires(self, key):
        with self.lock:
            self.cache[key]['expires'] = time.time() + self.ttl

    def get(self, key):
        with self.lock:
            if key in self.cache:
                item = self.cache.pop(key)
                self._update_expires(key)
                self.cache[key] = item
                return item['value']
            else:
                return None

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = {'value': value, 'expires': time.time() + self.ttl}
            self._update_expires(key)

    def start(self):
        threading.Thread(target=self._expire, args=()).start()

    def stop(self):
        threading.Thread(target=self._expire, args=()).stop()

class LRUCacheManager:
    def __init__(self):
        self.caches = {}

    def get_cache(self, name, capacity, ttl):
        if name not in self.caches:
            self.caches[name] = LRUCache(capacity, ttl)
            self.caches[name].start()
        return self.caches[name]