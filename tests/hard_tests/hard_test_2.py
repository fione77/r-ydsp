# Problem: Write a function to serialize and deserialize a binary tree
# Difficulty: Hard
# Generated in: 1.07s

# Problem: Write a function to serialize and deserialize a binary tree

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        def dfs(node):
            if node is None:
                vals.append('#')
                return
            vals.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        
        vals = []
        dfs(root)
        return ' '.join(vals)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        def dfs():
            val = next(vals)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node
        
        vals = iter(data.split())
        return dfs()

# Test the Codec class
if __name__ == "__main__":
    codec = Codec()
    
    # Create a binary tree:
    #          1
    #         / \
    #        2   3
    #       / \
    #      4   5
    
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    
    serialized_tree = codec.serialize(root)
    print("Serialized tree:", serialized_tree)
    
    deserialized_tree = codec.deserialize(serialized_tree)
    
    # Print the deserialized tree
    def print_tree(node, level=0):
        if node is not None:
            print_tree(node.right, level + 1)
            print('  ' * level + str(node.val))
            print_tree(node.left, level + 1)
    
    print_tree(deserialized_tree)