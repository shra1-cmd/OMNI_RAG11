from collections import deque

class ShortTermMemory:
    def __init__(self, max_turns: int = 10):
        self.buffer = deque(maxlen=max_turns)

    def add(self, role: str, content: str):
        self.buffer.append({
            "role": role,
            "content": content
        })

    def get(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()
