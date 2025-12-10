# ARCHITECT: "Let's use a simple solution..."

def binary_search(arr, target):
    """
    Performs a binary search on a sorted array.

    Args:
    arr (list): A sorted list of elements.
    target: The element to search for.

    Returns:
    int: The index of the target element if found, -1 otherwise.
    """
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1

# TESTER: "But what about edge cases like..."

def test_binary_search():
    """
    Tests the binary_search function with various edge cases.
    """
    # Test case 1: Target element is in the middle of the array
    arr = [1, 2, 3, 4, 5]
    target = 3
    assert binary_search(arr, target) == 2

    # Test case 2: Target element is at the beginning of the array
    arr = [1, 2, 3, 4, 5]
    target = 1
    assert binary_search(arr, target) == 0

    # Test case 3: Target element is at the end of the array
    arr = [1, 2, 3, 4, 5]
    target = 5
    assert binary_search(arr, target) == 4

    # Test case 4: Target element is not in the array
    arr = [1, 2, 3, 4, 5]
    target = 6
    assert binary_search(arr, target) == -1

    # Test case 5: Empty array
    arr = []
    target = 1
    assert binary_search(arr, target) == -1

# OPTIMIZER: "Maybe we should consider..."

def optimize_binary_search(arr, target):
    """
    Optimizes the binary search function by handling edge cases and adding error checking.
    """
    if not arr:  # Check if the array is empty
        return -1

    low = 0
    high = len(arr