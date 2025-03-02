from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

from graph.chains.retrieval_grader import structured_llm_grader

llm = ChatOpenAI(model="gpt-4", temperature=0)

class GradeHallucination(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: bool = Field(
        description="Answer is grounded in facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucination)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of facts.\n
Give a binary score 'yes' or 'no'. 'yes' means the answer is grounded in /supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader