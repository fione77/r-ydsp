def find_duplicates(lst):
    """
    Find duplicates in a list.

    Args:
        lst (list): The list to search for duplicates.

    Returns:
        list: A list of duplicates found in the input list.
    """
    seen = set()
    duplicates = set()

    for item in lst:
        if item in seen:
            duplicates.add(item)
        seen.add(item)

    return list(duplicates)

# Example usage:
numbers = [1, 2, 2, 3, 4, 4, 5, 6, 6]
print(find_duplicates(numbers))  # Output: [2, 4, 6]

# Edge case: Empty list
empty_list = []
print(find_duplicates(empty_list))  # Output: []

# Edge case: List with single unique element
single_element_list = [1]
print(find_duplicates(single_element_list))  # Output: []