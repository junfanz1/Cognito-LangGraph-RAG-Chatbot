from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from graph.state import GraphState

web_search_tool = TavilySearchResults(k=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---web search---")
    print("State keys:", state.keys())

    question = state.get("question", "")
    documents = state.get("documents", [])

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}