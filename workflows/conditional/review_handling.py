from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

llm = init_chat_model('google_genai:gemini-3.1-flash-lite')

# define schema
class SentimentSchema(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description='Sentiment of the review')

class DiagnosisSchema(BaseModel):
    issue_type: Literal['UX', 'Performance', 'Bug', 'Support', 'Other'] = Field(description='The category of issue mentioned in the review')
    tone: Literal['angry', 'frustrated', 'disappointed', 'calm'] = Field(description='The emotional tone expressed by the user')
    urgency: Literal['low', 'medium', 'high'] = Field(description='How urgent or critical the issue appears to be')

structured_model = llm.with_structured_output(SentimentSchema)
structured_model2 = llm.with_structured_output(DiagnosisSchema)

# define state
class ReviewState(TypedDict):
    review: str
    sentiment: Literal['positive', 'negative']
    diagnosis: dict
    response: str

# define graph
graph = StateGraph(ReviewState)

# define nodes
def find_sentiment(state: ReviewState) -> ReviewState:
    prompt = f'For the following review find out the sentiment \n {state["review"]}'
    result = structured_model.invoke(prompt).sentiment
    return {'sentiment': result}

def positive_response(state: ReviewState) -> ReviewState:
    prompt = f'Write a warm thank you message in response to this review: \n {state["review"]}'
    result = llm.invoke(prompt).content
    return {'response': result}

def negative_response(state: ReviewState) -> ReviewState:
    diagnosis = state['diagnosis']
    prompt = f'You are a support assistant. The user had a {diagnosis['issue_type']}, sounded {diagnosis['tone']} and marked urgency as {diagnosis['urgency']}. Write an empathetic, helpful resolution message'
    result = llm.invoke(prompt).content
    return {'response': result}

def run_diagnosis(state: ReviewState) -> ReviewState:
    prompt = f'Diagnosis this negative review \n {state["review"]} return issue_type, tone and urgency'
    result = structured_model2.invoke(prompt)
    print(f'diagnosis result without model_dump(): {result}')
    print(f'diagnosis result with dump: {result.model_dump()}')
    return {'diagnosis': result.model_dump()}

def check_sentiment(state: ReviewState) -> Literal['positive_response', 'run_diagnosis']:
    if state['sentiment'] == 'Positive': return 'positive_response'
    else: return 'run_diagnosis'

# add nodes
graph.add_node('find_sentiment', find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('negative_response', negative_response)
graph.add_node('run_diagnosis', run_diagnosis)

# add edges
graph.add_edge(START, 'find_sentiment')
graph.add_conditional_edges('find_sentiment', check_sentiment)
graph.add_edge('positive_response', END)
graph.add_edge('run_diagnosis', 'negative_response')
graph.add_edge('negative_response', END )

# compile graph
workflow = graph.compile()

# execute graph
init_state: ReviewState = {'review': 'I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality.'}
final_state: ReviewState = workflow.invoke(init_state)

# print result
print(final_state)