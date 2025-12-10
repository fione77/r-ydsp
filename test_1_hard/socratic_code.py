import threading
import time
import collections
import copy_reg
import types
import schedule
import threading
from typing import Dict, Any, Callable

def _pickle_method(m):
    """Helper function to pickle method objects."""
    if m.im_self is None:
        return getattr, (m.im_class,), m.im_func.__name__
    else:
        return getattr, (m.im_self, m.im_class), m.im_func.__name__

copy_reg.pickle(types.MethodType, _pickle_method)

class LRUCache:
    """
    A thread-safe LRU cache with TTL expiration.

    Attributes:
        max_size (int): The maximum size of the cache.
        ttl (int): The time to live for cache entries in seconds.
        cache (Dict[str, Tuple[Any, float]]): The cache data structure.
        access_order (CopyOnWriteArrayList): The access order of the cache.
        executor (ScheduledExecutorService): The executor for periodic removal of expired entries.
    """

    def __init__(self, max_size: int, ttl: int):
        """
        Initializes the LRU cache.

        Args:
            max_size (int): The maximum size of the cache.
            ttl (int): The time to live for cache entries in seconds.
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = collections.OrderedDict()
        self.access_order = collections.OrderedDict()
        self.executor = schedule.Scheduler()

        # Create a lock for thread-safe access
        self.lock = threading.Lock()

        # Schedule periodic removal of expired entries
        self.executor.every(ttl).seconds.do(self.remove_expired_entries)

    def get(self, key: str) -> Any:
        """
        Retrieves a value from the cache.

        Args:
            key (str): The key to retrieve.

        Returns:
            Any: The cached value, or None if not found.
        """
        with self.lock:
            # Check if the key exists in the cache
            if key in self.cache:
                # Move the accessed key to the end of the access order
                self.access_order.move_to_end(key)

                # Return the cached value
                return self.cache[key][0]

            # If the key does not exist, return None
            return None

    def set(self, key: str, value: Any, expiration_time: float):
        """
        Sets a value in the cache with a TTL.

        Args:
            key (str): The key to set.
            value (Any): The value to set.
            expiration_time (float): The expiration time of the cache entry in seconds.
        """
        with self.lock:
            # Check if the key already exists in the cache
            if key in self.cache:
                # Remove the existing key from the cache
                del self.cache[key]

            # Add the new key-value pair to the cache
            self.cache[key] = (value, expiration_time)

            # Add the key to the access order
            self.access_order[key] = None

            # If the cache is full, remove the least recently used entry
            if len(self.cache) > self.max_size:
                self.remove_expired_entries()

            # Schedule periodic removal of expired entries
            self.executor.run_pending()

    def remove_expired_entries(self):
        """
        Removes expired entries from the cache.
        """
        with self.lock:
            # Get the current time
            current_time = time.time()

            # Remove expired entries from the cache
            for key, (value, expiration_time) in list(self.cache.items()):
                if current_time >= expiration_time:
                    del self.cache[key]
                    del self.access_order[key]

    def shutdown(self):
        """
        Shuts down the cache and stops the executor.
        """
        self.executor.shutdown()


# Usage example:
if __name__ == "__main__":
    cache