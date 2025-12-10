# Problem: Implement a concurrent web crawler with rate limiting
# Difficulty: Very Hard
# Generated in: 1.17s

import concurrent.futures
import requests
from bs4 import BeautifulSoup
import time
from collections import deque
import threading

class WebCrawler:
    def __init__(self, start_url, max_workers=10, max_depth=5, max_requests=100):
        self.start_url = start_url
        self.max_workers = max_workers
        self.max_depth = max_depth
        self.max_requests = max_requests
        self.visited = set()
        self.queue = deque([self.start_url])
        self.lock = threading.Lock()

    def fetch(self, url):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def parse(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        return links

    def crawl(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            while self.queue and len(futures) < self.max_requests:
                url = self.queue.popleft()
                if url not in self.visited:
                    self.visited.add(url)
                    futures[url] = executor.submit(self._crawl, url)
                if len(futures) >= self.max_requests:
                    with self.lock:
                        for url in list(futures.keys()):
                            if futures[url].done():
                                del futures[url]
                                self.queue.extend(self._parse(futures[url].result()))
            for future in concurrent.futures.as_completed(futures):
                self._parse(future.result())

    def _crawl(self, url):
        html = self.fetch(url)
        if html:
            return html
        else:
            return None

    def _parse(self, html):
        if html:
            links = self.parse(html)
            for link in links:
                if link and link.startswith('http'):
                    self.queue.append(link)
                    if self.max_depth:
                        self.max_depth -= 1
        return links

    def run(self):
        start_time = time.time()
        self.crawl()
        print(f" Crawling finished in {time.time() - start_time} seconds")
        print(f" Visited {len(self.visited)} URLs")

# Example usage
crawler = WebCrawler("https://www.example.com")
crawler.run()