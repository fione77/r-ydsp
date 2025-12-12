"""
LeetCode Hard Problems Database - FIXED VERSION
5 hard problems for testing code generators
"""
import json
from typing import Dict, List, Any

LEETCODE_HARD_PROBLEMS = [
    {
        "id": 42,
        "title": "Trapping Rain Water",
        "difficulty": "Hard",
        "description": """
Given n non-negative integers representing an elevation map where the width of each bar is 1,
compute how much water it can trap after raining.

Example 1:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

Example 2:
Input: height = [4,2,0,3,2,5]
Output: 9

Constraints:
- n == height.length
- 1 <= n <= 2 * 10^4
- 0 <= height[i] <= 10^5

REQUIREMENTS:
- Time complexity: O(n)
- Space complexity: O(1) ideally, O(n) acceptable
- Must handle edge cases: empty array, single element, descending heights
""",
        "function_signature": "def trap(height: List[int]) -> int:",
        "optimal_time_complexity": "O(n)",
        "optimal_space_complexity": "O(1)",
        "optimal_solution": """
def trap(height):
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water
""",
        "test_cases": [
            {
                "input": {"height": [0,1,0,2,1,0,1,3,2,1,2,1]},
                "expected": 6,
                "name": "Example 1"
            },
            {
                "input": {"height": [4,2,0,3,2,5]},
                "expected": 9,
                "name": "Example 2"
            },
            {
                "input": {"height": []},
                "expected": 0,
                "name": "Empty array"
            },
            {
                "input": {"height": [5]},
                "expected": 0,
                "name": "Single element"
            },
            {
                "input": {"height": [5,4,3,2,1]},
                "expected": 0,
                "name": "Descending heights"
            },
            {
                "input": {"height": [1,2,3,4,5]},
                "expected": 0,
                "name": "Ascending heights"
            },
            {
                "input": {"height": [100000, 0, 100000]},
                "expected": 100000,
                "name": "Large numbers"
            }
        ],
        "key_edge_cases": [
            "Empty array should return 0",
            "Single element array should return 0",
            "All descending heights should return 0",
            "All ascending heights should return 0",
            "Symmetrical valleys should work",
            "Handle maximum constraint values"
        ]
    },
    {
        "id": 10,
        "title": "Regular Expression Matching",
        "difficulty": "Hard",
        "description": """
Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*'.

'.' Matches any single character.
'*' Matches zero or more of the preceding element.

The matching should cover the entire input string (not partial).

Example 1:
Input: s = "aa", p = "a"
Output: false

Example 2:
Input: s = "aa", p = "a*"
Output: true

Example 3:
Input: s = "ab", p = ".*"
Output: true

Constraints:
- 1 <= s.length <= 20
- 1 <= p.length <= 30
- s contains only lowercase English letters.
- p contains only lowercase English letters, '.', and '*'.
- It is guaranteed for each appearance of the character '*', there will be a previous valid character to match.

REQUIREMENTS:
- Time complexity: O(m*n) where m = len(s), n = len(p)
- Space complexity: O(m*n) for DP solution
""",
        "function_signature": "def isMatch(s: str, p: str) -> bool:",
        "optimal_time_complexity": "O(m*n)",
        "optimal_space_complexity": "O(m*n)",
        "optimal_solution": """
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Handle patterns like a*, a*b*, a*b*c*
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == s[i-1] or p[j-1] == '.':
                dp[i][j] = dp[i-1][j-1]
            elif p[j-1] == '*':
                dp[i][j] = dp[i][j-2]  # zero occurrences
                if p[j-2] == s[i-1] or p[j-2] == '.':
                    dp[i][j] = dp[i][j] or dp[i-1][j]  # one or more occurrences
    
    return dp[m][n]
""",
        "test_cases": [
            {"input": {"s": "aa", "p": "a"}, "expected": False, "name": "Simple mismatch"},
            {"input": {"s": "aa", "p": "a*"}, "expected": True, "name": "Star match"},
            {"input": {"s": "ab", "p": ".*"}, "expected": True, "name": "Dot star"},
            {"input": {"s": "aab", "p": "c*a*b"}, "expected": True, "name": "Complex pattern"},
            {"input": {"s": "mississippi", "p": "mis*is*p*."}, "expected": False, "name": "LeetCode example"},
            {"input": {"s": "", "p": "a*"}, "expected": True, "name": "Empty string with star"},
            {"input": {"s": "abc", "p": "a.c"}, "expected": True, "name": "Dot in middle"}
        ],
        "key_edge_cases": [
            "Empty string with various patterns",
            "Multiple stars in pattern",
            ".* at beginning, middle, end",
            "Pattern longer than string",
            "String longer than pattern with stars"
        ]
    },
    {
        "id": 23,
        "title": "Merge k Sorted Lists",
        "difficulty": "Hard",
        "description": """
You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
Merge all the linked-lists into one sorted linked-list and return it.

Example 1:
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]

Example 2:
Input: lists = []
Output: []

Example 3:
Input: lists = [[]]
Output: []

Constraints:
- k == lists.length
- 0 <= k <= 10^4
- 0 <= lists[i].length <= 500
- -10^4 <= lists[i].val <= 10^4
- lists[i] is sorted in ascending order.

REQUIREMENTS:
- Time complexity: O(n log k) where n is total number of nodes
- Space complexity: O(k) for heap solution
""",
        "function_signature": "def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:",
        "optimal_time_complexity": "O(n log k)",
        "optimal_space_complexity": "O(k)",
        "optimal_solution": """
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists(lists):
    heap = []
    
    # Add first node of each list to heap
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return dummy.next
""",
        "test_cases": [
            {
                "input": {"lists": [[1,4,5],[1,3,4],[2,6]]},
                "expected": [1,1,2,3,4,4,5,6],
                "name": "Example 1"
            },
            {
                "input": {"lists": []},
                "expected": [],
                "name": "Empty array"
            },
            {
                "input": {"lists": [[]]},
                "expected": [],
                "name": "Empty list"
            },
            {
                "input": {"lists": [[1],[0]]},
                "expected": [0,1],
                "name": "Two lists"
            },
            {
                "input": {"lists": [[-10,-9,-9,-3,-1,-1,0],[-5],[4],[-8],[],[-9,-6,-5,-4,-2,2,3],[-3,-3,-2,-1,0]]},
                "expected": [-10,-9,-9,-9,-8,-6,-5,-5,-4,-3,-3,-3,-2,-2,-1,-1,-1,0,0,2,3,4],
                "name": "Complex lists"
            }
        ],
        "key_edge_cases": [
            "Empty lists array",
            "Lists containing empty lists",
            "Lists with negative numbers",
            "Very large k (up to 10^4)",
            "Lists of varying lengths"
        ]
    },
    {
        "id": 295,
        "title": "Find Median from Data Stream",
        "difficulty": "Hard",
        "description": """
The median is the middle value in an ordered integer list. If the size of the list is even,
there is no middle value and the median is the mean of the two middle values.

Implement the MedianFinder class:
- MedianFinder() initializes the MedianFinder object.
- void addNum(int num) adds the integer num from the data stream to the data structure.
- double findMedian() returns the median of all elements so far.

Example 1:
Input:
["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]
[[],[1],[2],[],[3],[]]
Output: [null,null,null,1.5,null,2.0]

Constraints:
- -10^5 <= num <= 10^5
- There will be at least one element in the data structure before calling findMedian.
- At most 5 * 10^4 calls will be made to addNum and findMedian.

REQUIREMENTS:
- Time complexity: O(log n) for addNum, O(1) for findMedian
- Space complexity: O(n)
""",
        "class_signature": """
class MedianFinder:
    def __init__(self):
        # Initialize your data structure here
    
    def addNum(self, num: int) -> None:
        # Add number to data structure
    
    def findMedian(self) -> float:
        # Return median of all numbers
""",
        "optimal_time_complexity": "O(log n) add, O(1) find",
        "optimal_space_complexity": "O(n)",
        "optimal_solution": """
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # max heap (store negatives for max heap behavior)
        self.large = []  # min heap
    
    def addNum(self, num):
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
        else:
            heapq.heappush(self.large, num)
        
        # Balance heaps
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        elif len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0
""",
        "test_cases": [
            {
                "operations": ["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"],
                "args": [[],[1],[2],[],[3],[]],
                "expected": [None,None,None,1.5,None,2.0],
                "name": "Example 1"
            },
            {
                "operations": ["MedianFinder","addNum","findMedian","addNum","findMedian","addNum","findMedian"],
                "args": [[],[1],[],[2],[],[3],[]],
                "expected": [None,None,1.0,None,1.5,None,2.0],
                "name": "Gradual addition"
            },
            {
                "operations": ["MedianFinder","addNum","addNum","addNum","addNum","addNum","findMedian"],
                "args": [[],[5],[3],[4],[2],[1],[]],
                "expected": [None,None,None,None,None,None,3.0],
                "name": "Five numbers"
            }
        ],
        "key_edge_cases": [
            "Single number median",
            "Negative numbers",
            "Large number of operations (50k)",
            "Extreme values (-10^5 to 10^5)",
            "Alternating add and find operations"
        ]
    },
    {
        "id": 297,
        "title": "Serialize and Deserialize Binary Tree",
        "difficulty": "Hard",
        "description": """
Serialization is the process of converting a data structure or object into a sequence of bits
so that it can be stored in a file or memory buffer, or transmitted across a network connection
link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how
your serialization/deserialization algorithm should work. You just need to ensure that a binary
tree can be serialized to a string and this string can be deserialized to the original tree structure.

Example 1:
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]

Constraints:
- The number of nodes in the tree is in the range [0, 10^4].
- -1000 <= Node.val <= 1000

REQUIREMENTS:
- Both serialize and deserialize should work in O(n) time
- Serialized string should be as compact as possible
""",
        "class_signature": """
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Codec:
    def serialize(self, root):
        '''Encodes a tree to a single string.'''
    
    def deserialize(self, data):
        '''Decodes your encoded data to tree.'''
""",
        "optimal_time_complexity": "O(n)",
        "optimal_space_complexity": "O(n)",
        "optimal_solution": """
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Codec:
    def serialize(self, root):
        def dfs(node):
            if not node:
                return "None,"
            return str(node.val) + "," + dfs(node.left) + dfs(node.right)
        return dfs(root)
    
    def deserialize(self, data):
        def dfs(nodes):
            val = next(nodes)
            if val == "None":
                return None
            node = TreeNode(int(val))
            node.left = dfs(nodes)
            node.right = dfs(nodes)
            return node
        
        return dfs(iter(data.split(',')))
""",
        "test_cases": [
            {
                "input": [1,2,3,None,None,4,5],
                "expected": [1,2,3,None,None,4,5],
                "name": "Example 1"
            },
            {
                "input": [],
                "expected": [],
                "name": "Empty tree"
            },
            {
                "input": [1],
                "expected": [1],
                "name": "Single node"
            },
            {
                "input": [1,2,None,3,None,4,None,5],
                "expected": [1,2,None,3,None,4,None,5],
                "name": "Left skewed tree"
            },
            {
                "input": [-10,9,20,None,None,15,7],
                "expected": [-10,9,20,None,None,15,7],
                "name": "Tree with negatives"
            }
        ],
        "key_edge_cases": [
            "Empty tree",
            "Single node tree",
            "Left/right skewed trees",
            "Full binary tree",
            "Tree with duplicate values",
            "Large tree (10^4 nodes)",
            "Negative and zero values"
        ]
    }
]

def save_problems_to_file(filename="leetcode_hard_problems.json"):
    """Save problems to JSON file"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(LEETCODE_HARD_PROBLEMS, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(LEETCODE_HARD_PROBLEMS)} problems to {filename}")

if __name__ == "__main__":
    save_problems_to_file()