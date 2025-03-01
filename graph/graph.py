from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import END, StateGraph
from graph.consts import *
from graph.nodes import *
from graph.state import GraphState

def decide_to_generate(state):
    print("---assess granted documents---")
    if state["web_search"]:
        # we find a doc not relevant to user query
        print("---decision: not all docs are relevant to question ---")
        return WEBSEARCH
    else:
        print("---decision: generate---")
        return GENERATE

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(GENERATE, generate)
workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    # path map
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    }
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")