# Problem: Implement a thread-safe LRU cache with TTL expiration
# Difficulty: Hard
# Generated in: 1.39s

from collections import OrderedDict
import time
import threading

class LRUCache:
    def __init__(self, capacity, ttl):
        """
        Initialize the LRU cache with capacity and TTL (time to live) in seconds.

        :param capacity: Maximum number of items to store in the cache.
        :param ttl: Time to live in seconds for each item in the cache.
        """
        self.capacity = capacity
        self.ttl = ttl
        self.lock = threading.Lock()
        self.cache = OrderedDict()

    def get(self, key):
        """
        Retrieve the value associated with the given key from the cache.

        :param key: The key to look up in the cache.
        :return: The value associated with the key if it exists in the cache, otherwise None.
        """
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache.pop(key)
                self.cache[key] = (value, time.time())  # Move to the end to mark as recently used
                return value
            return None

    def set(self, key, value):
        """
        Add or update a key-value pair in the cache with TTL expiration.

        :param key: The key to add or update in the cache.
        :param value: The value associated with the key.
        """
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Remove the oldest item if the cache is full
                oldest_key = self.cache.popitem(last=False)[0]
                self.cache[oldest_key] = (None, 0)  # Set TTL to 0 to expire immediately
            self.cache[key] = (value, time.time())

    def clean(self):
        """
        Remove expired items from the cache.
        """
        with self.lock:
            current_time = time.time()
            self.cache = {key: (value, timestamp) for key, (value, timestamp) in self.cache.items() if timestamp + self.ttl > current_time}
            if len(self.cache) > self.capacity:
                # Remove the oldest item if the cache is full
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)