from dotenv import load_dotenv
from pprint import pprint

load_dotenv()
from graph.chains.router import question_router, RouteQuery
from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucination

def test_retrieval_grader_answer_yes() -> None:
    question = "agent"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": question}
    )
    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = "agent"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": "How to make pizza"}
    )
    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)

def test_hallucination_grader_answer_yes() -> None:
    question = "agent"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucination = retrieval_grader.invoke(
        {"document": docs, "question": question}
    )
    assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
    question = "agent"
    docs = retriever.invoke(question)
    res: GradeHallucination = retrieval_grader.invoke(
        {"document": docs, "question": question,
         "generation": "To make a pizza we need to buy some sausage"}
    )
    assert not res.binary_score  # should assert not...

def test_router_to_vectorstore() -> None:
    question = "agent"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"

def test_router_to_websearch() -> None:
    question = "how to make pizza"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"