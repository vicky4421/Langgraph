from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
# from nodes.constants import Node      # leave it for now, we've to set path in sys, will deal it later

# define state
class QuadState(TypedDict):
    a: int
    b: int
    c: int
    equation: str
    discriminant: float
    result: str

# define graph
graph = StateGraph(QuadState)

# define nodes
def show_equation(state: QuadState) -> QuadState:
    # equation = ax^2 + bx + c
    return {'equation': f'{state["a"]}x^2 + {state['b']}x + {state['c']}'}

def calculate_discriminant(state: QuadState) -> QuadState:
    # discriminant = b^2 + 4ac
    return {'discriminant': (state['b']**2 - 4 * state['a'] * state['c'])}


# add nodes
graph.add_node('show_equation', show_equation)
graph.add_node('calculate_discriminant', calculate_discriminant)

# add edges
graph.add_edge(START, 'show_equation')
graph.add_edge('show_equation', 'calculate_discriminant')
graph.add_edge('calculate_discriminant', END)

# compile graph
workflow = graph.compile()

# execute graph
init_state: QuadState = {'a': 4, 'b': -5, 'c': -4}
final_state: QuadState = workflow.invoke(init_state)

# print result
print(final_state)