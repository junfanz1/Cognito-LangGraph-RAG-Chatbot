from typing import List, TypedDict

class GraphState(TypedDict):
    """
    include all the states we need for graph execution.
    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]