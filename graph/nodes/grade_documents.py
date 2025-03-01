from typing import Any, Dict
from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determine whether retrieved docs are relevant to the question.
    If any doc is irrelevant, set a flag to run web search.
    Args:
        state (dict): Current graph state.
    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state.
    """
    print("---check document relevance to question---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = False
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score.binary_score
        if grade.lower() == 'yes':
            print("---grade: document relevant---")
            filtered_docs.append(doc)
        else:
            print("---grade: document not relevant---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}