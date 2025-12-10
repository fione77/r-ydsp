class TreeNode:
    """Represents a node in a binary tree."""
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def serialize(root: TreeNode) -> str:
    """
    Serialize a binary tree into a string.

    Args:
    root: The root node of the binary tree.

    Returns:
    A string representing the serialized binary tree.
    """
    def dfs(node: TreeNode) -> str:
        # Handle edge case: Empty tree
        if node is None:
            return "X,"
        
        # Recursively serialize left and right subtrees
        left = dfs(node.left)
        right = dfs(node.right)
        
        # Append node's value and serialized subtrees to the output string
        return str(node.val) + "," + left + right
    
    # Call the dfs function to start the recursive traversal
    return dfs(root).rstrip(",")

def deserialize(data: str) -> TreeNode:
    """
    Deserialize a binary tree from a string.

    Args:
    data: A string representing the serialized binary tree.

    Returns:
    The root node of the deserialized binary tree.
    """
    def dfs(values: list[str]) -> TreeNode:
        # Handle edge case: Empty string
        if not values:
            return None
        
        # Get the next value from the list
        val = values.pop(0)
        
        # Handle edge case: Empty tree
        if val == "X":
            return None
        
        # Create a new node with the current value
        node = TreeNode(int(val))
        
        # Recursively construct the left and right subtrees
        node.left = dfs(values)
        node.right = dfs(values)
        
        # Return the constructed node
        return node
    
    # Split the input string into a list of values
    values = data.split(",")
    
    # Call the dfs function to start the recursive traversal
    return dfs(values)

# Testing the implementation
if __name__ == "__main__":
    # Create a sample binary tree
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.right.left = TreeNode(4)
    root.right.right = TreeNode(5)

    # Serialize the binary tree
    serialized = serialize(root)
    print("Serialized:", serialized)

    # Deserialize the binary tree
    deserialized = deserialize(serialized)
    print("Deserialized Root:", deserialized.val if deserialized else None)