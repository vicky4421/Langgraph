from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# State
class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    str_rate: float
    boundary_percentage: float
    balls_per_boundary: float
    summary: str

# define graph
graph = StateGraph(BatsmanState)

# define nodes
def calculate_strike_rate(state: BatsmanState) -> BatsmanState:
    sr_rate = (state['runs'] / state['balls']) * 100
    # return state
    return {'str_rate': sr_rate}

def calculate_balls_per_boundary(state: BatsmanState) -> BatsmanState:
    bpb = state['balls'] / (state['fours'] + state['sixes'])
    # return state
    return {'balls_per_boundary': bpb}

def calculate_boundary_percentage(state: BatsmanState) -> BatsmanState:
    bp = (((state['fours'] * 4) + (state['sixes'] * 6))/state['runs']) * 100
    # return state
    return {'boundary_percentage': bp}

def summary(state: BatsmanState) -> BatsmanState:
    state['summary'] = f'''
    Strike Rate: {state['str_rate']}
    Balls Per Boundary: {state["balls_per_boundary"]}
    Boundary Percentage: {state["boundary_percentage"]}
'''
    return {'summary': summary}

# add nodes
graph.add_node('calculate_strike_rate', calculate_strike_rate)
graph.add_node('calculate_balls_per_boundary', calculate_balls_per_boundary)
graph.add_node('calculate_boundary_percentage', calculate_boundary_percentage)
graph.add_node('summary', summary)

# add edges
graph.add_edge(START, 'calculate_strike_rate')
graph.add_edge(START, 'calculate_balls_per_boundary')
graph.add_edge(START, 'calculate_boundary_percentage')

graph.add_edge('calculate_strike_rate', 'summary')
graph.add_edge('calculate_balls_per_boundary', 'summary')
graph.add_edge('calculate_boundary_percentage', 'summary')

graph.add_edge('summary', END)

# compile graph
workflow = graph.compile()

# execute graph
init_state: BatsmanState = {'runs': 100, 'balls': 50, 'fours': 6, 'sixes': 4}
final_state: BatsmanState = workflow.invoke(init_state)

# print state
print(final_state)

'''
Output: langgraph.errors.InvalidUpdateError: At key 'runs': Can receive only one value per step. Use an Annotated key to handle multiple values.

Because every node is expecting a State and returning a State, due to this langgraph is thining that every node is making changes in state and returning the state, and making changes symulteneously in one state is not allowed, it would create ambiguity.
So we need to return partial state from nodes.
Why? langgraph will see that every node is making changes in different attribute of state.
At the end the State is just a dictionary so nodes are expecting a dict and returning a dict.
'''