from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image

# Define State
class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    category: str

# Define functions for nodes
def calculate_bmi(state: BMIState) -> BMIState:

    # bmi = weight / (height square): round off to 2 decimal places using round()
    state['bmi'] =  round(state['weight_kg'] / (state['height_m']**2), 2)

    return state

def label_bmi(state: BMIState) -> BMIState:
    bmi = state['bmi']

    if bmi < 18.5:
        state['category'] = 'Underweight'
    elif 18.5 <= bmi < 25:
        state['category'] = 'Normal'
    elif 25 <= bmi < 30:
        state['category'] = 'Overweight'
    else:
        state['category'] = 'Obese'

    return state

# Define graph
graph = StateGraph(state_schema= BMIState)

# Add nodes to the graph
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('label_bmi', label_bmi)

# Add edges to the graph
graph.add_edge(start_key= START, end_key= 'calculate_bmi')
graph.add_edge('calculate_bmi', 'label_bmi')
graph.add_edge('label_bmi', END)

# Compile the graph
workflow = graph.compile()

# Execute the graph
initial_state: BMIState = {'weight_kg': 78, 'height_m': 1.74}
final_state: BMIState = workflow.invoke(initial_state)

print(final_state)
Image(workflow.get_graph().draw_mermaid_png())   # This works in jupyter notebook

# Output: {'weight_kg': 78, 'height_m': 1.74, 'bmi': 25.76, 'category': 'Overweight'}