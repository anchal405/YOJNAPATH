from langgraph.types import StreamWriter


class TypedLivekit:
    writer: StreamWriter

    def __init__(self, writer: StreamWriter):
        self.writer = writer

    def say(self, content: str):
        self.writer({"type": "say", "data": {"content": content}})

    def flush(self):
        self.writer({"type": "flush", "data": None})
