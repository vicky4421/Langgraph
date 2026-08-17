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
    return {'equation': f'{state["a"]}x^2 + ({state['b']}x) + ({state['c']})'}

def calculate_discriminant(state: QuadState) -> QuadState:
    # discriminant = b^2 + 4ac
    return {'discriminant': (state['b']**2 - 4 * state['a'] * state['c'])}

def real_roots(state: QuadState) -> QuadState:
    root1 = (-state['b'] + state['discriminant'] ** 0.5)/(2 * state['a'])
    root2 = (-state['b'] - state['discriminant'] ** 0.5)/(2 * state['a'])
    return {'result': f'The roots are {root1} and {root2}'}

def repeated_roots(state: QuadState) -> QuadState:
    root = (-state['b'])/(2 * state['a'])
    return {'result': f'Only repeating root is {root}'}

def no_real_roots(state: QuadState) -> QuadState:
    return {'result': 'No real roots'}

# check condition (This is not node, it returns a literal string of node name)
def check_condition(state: QuadState) -> Literal['real_roots', 'repeated_roots', 'no_real_roots']:
    if state['discriminant'] > 0: return 'real_roots'
    elif state['discriminant'] == 0: return 'repeated_roots'
    else: return 'no_real_roots'

# add nodes
graph.add_node('show_equation', show_equation)
graph.add_node('calculate_discriminant', calculate_discriminant)
graph.add_node('real_roots', real_roots)
graph.add_node('repeated_roots', repeated_roots)
graph.add_node('no_real_roots', no_real_roots)

# add edges
graph.add_edge(START, 'show_equation')
graph.add_edge('show_equation', 'calculate_discriminant')
graph.add_conditional_edges('calculate_discriminant', check_condition)
graph.add_edge('real_roots', END)
graph.add_edge('repeated_roots', END)
graph.add_edge('no_real_roots', END)

# compile graph
workflow = graph.compile()

# execute graph
init_state: QuadState = {'a': 4, 'b': -5, 'c': -4}
final_state: QuadState = workflow.invoke(init_state)

# print result
print(final_state)

'''
Output:
    {'a': 4, 'b': -5, 'c': -4, 'equation': '4x^2 + (-5x) + (-4)', 'discriminant': 89, 'result': 'The roots are 1.8042476415070754 and -0.5542476415070754'}
'''