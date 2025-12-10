# Initial discussion between the programmers

ARCHITECT: "Let's use a simple solution, we can define a Stack class with push and pop methods."

TESTER: "But what about edge cases like an empty stack? What happens when we try to pop from an empty stack?"

OPTIMIZER: "Maybe we should consider using a list to implement the stack, it's efficient and easy to use."

ARCHITECT: "OK, revised approach... Let's use a list to implement the stack."

# Revised code

class SimpleStack:
    """
    A simple implementation of a stack using a list.
    """

    def __init__(self):
        """
        Initializes an empty stack.
        """
        self.stack = []

    def push(self, item):
        """
        Adds an item to the top of the stack.
        
        Args:
            item: The item to add to the stack.
        """
        self.stack.append(item)

    def pop(self):
        """
        Removes and returns the top item from the stack. If the stack is empty, raises an IndexError.
        
        Returns:
            The top item from the stack.
        
        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot pop from an empty stack")
        return self.stack.pop()

    def peek(self):
        """
        Returns the top item from the stack without removing it. If the stack is empty, raises an IndexError.
        
        Returns:
            The top item from the stack.
        
        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot peek an empty stack")
        return self.stack[-1]

    def is_empty(self):
        """
        Checks if the stack is empty.
        
        Returns:
            True if the stack is empty, False otherwise.
        """
        return len(self.stack) == 0

    def size(self):
        """
        Returns the number of items in the stack.
        
        Returns:
            The number of items in the stack.
        """
        return len(self.stack)