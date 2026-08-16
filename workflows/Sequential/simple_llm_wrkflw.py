'''
START -> LLM -> END
'''

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm = init_chat_model(
    model='gemini-3.1-flash-lite',
    model_provider='google_genai'
)

# State
class LLMState(TypedDict):
    question: str
    answer: str

# define nodes
def llm_qa(state: LLMState) -> LLMState:
    question = state['question']
    prompt = f'Answer the following question: {question}'
    state['answer'] = llm.invoke(prompt).content
    return state

# create graph
graph = StateGraph(state_schema= LLMState)

# add nodes
graph.add_node(node = 'llm_qa', action = llm_qa)

# add edges
graph.add_edge(start_key = START, end_key = 'llm_qa')
graph.add_edge('llm_qa', END)

# compile graph
workflow = graph.compile()

# execute graph
init_state: LLMState = {'question': 'How far is moon from earth?'}
final_state: LLMState = workflow.invoke(init_state)

print(final_state)