import collections

# Definition for a binary tree node.
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
        def preorder(node):
            if node:
                vals.append(str(node.val))
                preorder(node.left)
                preorder(node.right)
            else:
                vals.append("#")
        
        vals = []
        preorder(root)
        return " ".join(vals)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        def postorder():
            val = next(vals)
            if val == "#":
                return None
            node = TreeNode(int(val))
            node.right = postorder()
            node.left = postorder()
            return node
        
        vals = iter(data.split())
        return postorder()

# Example Usage
if __name__ == "__main__":
    # Create a sample binary tree:
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)

    codec = Codec()
    serialized_tree = codec.serialize(root)
    print("Serialized Tree:", serialized_tree)

    deserialized_root = codec.deserialize(serialized_tree)
    print("Deserialized Root:", deserialized_root.val)

    # Test the deserialized tree
    def print_tree(node, level=0):
        if node is not None:
            print_tree(node.right, level + 1)
            print("  " * level + str(node.val))
            print_tree(node.left, level + 1)

    print("Deserialized Tree:")
    print_tree(deserialized_root)