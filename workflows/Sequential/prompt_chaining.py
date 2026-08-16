'''
topic -> llm -> outline for blog -> llm -> topic + outline -> output blog
'''

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model('google_genai:gemini-3.1-flash-lite')

# define state
class BlogState(TypedDict):
    title: str
    outline: str
    blog_content: str
    evaluation_score: int

# create graph
graph = StateGraph(BlogState)

# define nodes
def create_outline(state: BlogState) -> BlogState:
    title = state['title']
    prompt = f'Generate a blog outline on topic: {title}'
    state['outline'] = llm.invoke(prompt).content
    return state

def create_blog(state: BlogState) -> BlogState:
    title = state['title']
    outline = state['outline']
    prompt = f'Generate a blog using the blog outline: {outline}, on topic: {title}'
    state['blog_content'] = llm.invoke(prompt).content
    return state

def evaluate_blog(state: BlogState) -> BlogState:
    outline = state['outline']
    blog_content = state['blog_content']
    prompt = f'Based on the this {outline}, rate my blog: {blog_content} and generate a score'
    state['evaluation_score'] = llm.invoke(prompt).content
    return state

# add nodes
graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)
graph.add_node('evaluate_blog', evaluate_blog)

# add edges
graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog', 'evaluate_blog')
graph.add_edge('evaluate_blog', END)

# compile graph
workflow = graph.compile()

# execute graph
init_state: BlogState = {'title': 'AI Agents'}
final_state: BlogState = workflow.invoke(init_state)

print(final_state)
print('\n')
print(f'Evaluation Score: {final_state["evaluation_score"]}')