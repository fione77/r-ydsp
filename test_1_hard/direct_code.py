import threading
from collections import OrderedDict
import time
from datetime import datetime, timedelta

class LRUTTLCache:
    """
    Thread-safe LRU cache with TTL expiration.
    """

    def __init__(self, capacity, ttl):
        """
        Initialize the LRU cache.

        Args:
            capacity (int): Maximum number of items in the cache.
            ttl (int): Time to live in seconds for each item.
        """
        self.capacity = capacity
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        """
        Get the value associated with the given key.

        Args:
            key (any): Cache key.

        Returns:
            any: Value associated with the key if found, otherwise None.
        """
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache.pop(key)
                self.cache[key] = (value, datetime.now() - timedelta(seconds=self.ttl))
                return value
            return None

    def set(self, key, value):
        """
        Set the value associated with the given key.

        Args:
            key (any): Cache key.
            value (any): Value associated with the key.
        """
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = (value, datetime.now())

    def delete(self, key):
        """
        Delete the item associated with the given key.

        Args:
            key (any): Cache key.
        """
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)

# Example usage
cache = LRUTTLCache(capacity=100, ttl=60)  # 1 minute TTL

# Set some values
cache.set('key1', 'value1')
cache.set('key2', 'value2')

# Get values
print(cache.get('key1'))  # Output: value1
print(cache.get('key2'))  # Output: value2

# Delete an item
cache.delete