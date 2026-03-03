from typing import Dict, List

from langgraph.graph import MessagesState


class State(MessagesState):
    next: str
    task_description: str
    retrieved_information: str
    memory: str
    worker_input_file: str
    worker_output_files: Dict[str, List[str]]
