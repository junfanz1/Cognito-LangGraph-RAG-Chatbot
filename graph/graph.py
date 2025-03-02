from dotenv import load_dotenv

from graph.chains.answer_grader import answer_grader

from langgraph.graph import END, StateGraph
from graph.consts import *
from graph.nodes import *
from graph.state import GraphState
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery

load_dotenv()

def decide_to_generate(state):
    print("---assess granted documents---")
    if state["web_search"]:
        # we find a doc not relevant to user query
        print("---decision: not all docs are relevant to question ---")
        return WEBSEARCH
    else:
        print("---decision: generate---")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: StateGraph) -> str:
    print("---check hallucination---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    if hallucination_grade := score.binary_score:
        print("---decision: generation is grounded in documents---")
        print("---grade generation vs question---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---decision: generation addresses question---")
            return "useful"
        else:
            print("---decision: generation does not address question---")
            return "not useful" # vector store info not sufficient to answer question, need external search
    else:
        print("---decision: generation is not grounded in documents, retry---")
        return "not supported"

def route_question(state: GraphState) -> str:
    print("---route question---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---route question to websearch---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---route question to RAG---")
        return RETRIEVE


workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    }
)

#workflow.set_entry_point(RETRIEVE)
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
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE, # regenerate after answer not grounded
        "useful": END, # return answer to user
        "not useful": WEBSEARCH, # vector store doesn't have enough info, go to websearch
    }
)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")