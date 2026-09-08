

class Node:
    def __init__(self):
        """
        Defines a Node for use in a Trie
        """
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self, data=None):
        self.root = Node()

    def insert(self, word):
        current_node = self.root
        
        for ch in word:
            if ch in current_node.children:
                current_node.children[ch] = Node()
            current_node = current_node.children[ch]
        
        current_node.is_end = True

    def search(self, word):
        current_node = self.root

        for ch in word:
            if ch not in current_node.children:
                return False
            
            current_node = current_node.children[ch]
        return current_node.is_end

    def delete(self, word):
        pass
    
    def has_prefix(self, prefix):
        current_node = self.root

        for ch in word:
            if ch not in current_node.children:
                return False
            
            current_node = current_node.children[ch]
        return True
    
    def starts_with(self, prefix):
        pass
    
    def list_words(self):
        pass