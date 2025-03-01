from typing import TypedDict, Any, Dict
from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---Retrieving---")
    question = state["question"]
    # do semantic search and get all relevant docs
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
