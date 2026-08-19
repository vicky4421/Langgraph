from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = init_chat_model('google_genai:gemini-3.1-flash-lite')


# state
class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str


# define graph
graph = StateGraph(state_schema=JokeState)

# define nodes
def generate_joke(state: JokeState) -> JokeState:
    prompt = f'generate joke on the topic: {state['topic']}'
    response = llm.invoke(prompt).content
    return {'joke': response}

def explain_joke(state: JokeState) -> JokeState:
    prompt = f'write a short explanation for the joke: {state['joke']}'
    response = llm.invoke(prompt).content
    return {'explanation': response}

# add nodes
graph.add_node('generate_joke', generate_joke)
graph.add_node('explain_joke', explain_joke)

# add edges
graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'explain_joke')
graph.add_edge('explain_joke', END)

# define checkpointer [here we're saving state in RAM]
checkpointer = InMemorySaver()

# compile graph
workflow = graph.compile(checkpointer=checkpointer)

config1 = {'configurable': {'thread_id': '1'}}

# execute graph
init_state: JokeState = {'topic': 'Coding'}
final_state: JokeState = workflow.invoke(init_state, config=config1)

# print result
print(final_state)
print(list(workflow.get_state_history(config=config1)))

# config2
config2 = {'configurable': {'thread_id': '2'}}
final_state2: JokeState = workflow.invoke({'topic': 'Cricket'}, config=config2)

print(final_state2)
print(list(workflow.get_state_history(config=config2)))

'''
Output:
    thread 1 final state / workflow.get_state(config1):
        {'topic': 'Coding', 'joke': [{'type': 'text', 'text': 'Why do programmers prefer dark mode?\n\nBecause light attracts bugs.', 'extras': {'signature': 'EnEKbwERTTIPV3HpPOPbbHvyYBd+yVn1e3YFxxvUJfwuGJY5ZzqoL/LI3WsZjfJoNbGS/lNys3+BxaE4Go6W261s1jAXPTb6T4gzkmgskVDj6sOLJvOZtVGZTtUKvkAiD3fLU9LJCs/sj33WyhRh139eyw=='}}], 'explanation': [{'type': 'text', 'text': 'This joke plays on a double meaning of the word **"bugs."**\n\n1.  **In nature:** Insects (bugs) are naturally drawn to light sources, like lamps or porch lights at night.\n2.  **In programming:** A "bug" is a common term for an error, flaw, or glitch in computer code. \n\nThe joke suggests that by choosing "dark mode" (a dark background for their screens), programmers are metaphorically avoiding the "bugs" that would otherwise be attracted to the "light."', 'extras': {'signature': 'EnEKbwERTTIP/n3ahOg/ZUTkP8Hbq95Njo6NMffALfKVoepHKbUdcNzlbVGUTtwABuF9aNoukwLZwNgpbGlP2r4QQLRs2qh++LEXPdkX3yLNugc15UJ448nHTHaP99Avf4EdkrOjw9KFLe+kZnan8RblYw=='}}]}

    get_state_history: snapshots
        [StateSnapshot(values={'topic': 'Coding', 'joke': [{'type': 'text', 'text': 'Why do programmers prefer dark mode?\n\nBecause light attracts bugs.', 'extras': {'signature': 'EnEKbwERTTIPtVTcB+biQ81/ou8j/TsApPcl42t4d4cQg3SzHIcqd+uCNRElqPiHBhCafEGKVQOpBaF5itk+DnSUr2/jmZF/ECTBlv/oelJ/KVDluBi3sjO5g5Aaw4FlAvQG6bUsMi/EV+Z2JH3aBgQAnw=='}}], 'explanation': [{'type': 'text', 'text': 'This joke is a play on two different meanings of the word **"bugs"**:\n\n1.  **In nature:** Insects are naturally drawn to bright lights (like a porch light or a lamp).\n2.  **In programming:** A "bug" is an error or flaw in computer code that prevents it from working correctly.\n\nThe joke relies on the programmer\'s perspective: by using "dark mode" (a dark background on their screen), they jokingly imply that they are avoiding the "bugs" that would otherwise be attracted to the light.', 'extras': {'signature': 'EnEKbwERTTIPeaTr8FNU9ztvjS26fuHSWvowfDPC18wJ94A2noSXe0LyNt2gUDcZM6dVvOt+SV9nXRHq5Ij1bzCdIZQVr84KMrjLnQkCszCW7Kquh0IJxrcNca/W5sIWbivUW72ZucgIOcgskXde9vroFg=='}}]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f19bacd-d060-674f-8002-601f4eda7cbd'}}, metadata={'source': 'loop', 'step': 2, 'parents': {}}, created_at='2026-08-19T09:03:37.003591+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f19bacd-c3e8-667f-8001-809a8e5ab6a6'}}, tasks=(), interrupts=()), StateSnapshot(values={'topic': 'Coding', 'joke': [{'type': 'text', 'text': 'Why do programmers prefer dark mode?\n\nBecause light attracts bugs.', 'extras': {'signature': 'EnEKbwERTTIPtVTcB+biQ81/ou8j/TsApPcl42t4d4cQg3SzHIcqd+uCNRElqPiHBhCafEGKVQOpBaF5itk+DnSUr2/jmZF/ECTBlv/oelJ/KVDluBi3sjO5g5Aaw4FlAvQG6bUsMi/EV+Z2JH3aBgQAnw=='}}]}, next=('explain_joke',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f19bacd-c3e8-667f-8001-809a8e5ab6a6'}}, metadata={'source': 'loop', 'step': 1, 'parents': {}}, created_at='2026-08-19T09:03:35.696126+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f19bacd-ac41-62c2-8000-093f2bbb5936'}}, tasks=(PregelTask(id='7598629c-d162-eaba-b1bf-87f52b0db59a', name='explain_joke', path=('__pregel_pull', 'explain_joke'), error=None, interrupts=(), state=None, result={'explanation': [{'type': 'text', 'text': 'This joke is a play on two different meanings of the word **"bugs"**:\n\n1.  **In nature:** Insects are naturally drawn to bright lights (like a porch light or a lamp).\n2.  **In programming:** A "bug" is an error or flaw in computer code that prevents it from working correctly.\n\nThe joke relies on the programmer\'s perspective: by using "dark mode" (a dark background on their screen), they jokingly imply that they are avoiding the "bugs" that would otherwise be attracted to the light.', 'extras': {'signature': 'EnEKbwERTTIPeaTr8FNU9ztvjS26fuHSWvowfDPC18wJ94A2noSXe0LyNt2gUDcZM6dVvOt+SV9nXRHq5Ij1bzCdIZQVr84KMrjLnQkCszCW7Kquh0IJxrcNca/W5sIWbivUW72ZucgIOcgskXde9vroFg=='}}]}),), interrupts=()), StateSnapshot(values={'topic': 'Coding'}, next=('generate_joke',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f19bacd-ac41-62c2-8000-093f2bbb5936'}}, metadata={'source': 'loop', 'step': 0, 'parents': {}}, created_at='2026-08-19T09:03:33.215899+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f19bacd-ac31-662e-bfff-cdf51b29b807'}}, tasks=(PregelTask(id='41e629d9-1b6c-1fb5-1a93-06ff32d3f96f', name='generate_joke', path=('__pregel_pull', 'generate_joke'), error=None, interrupts=(), state=None, result={'joke': [{'type': 'text', 'text': 'Why do programmers prefer dark mode?\n\nBecause light attracts bugs.', 'extras': {'signature': 'EnEKbwERTTIPtVTcB+biQ81/ou8j/TsApPcl42t4d4cQg3SzHIcqd+uCNRElqPiHBhCafEGKVQOpBaF5itk+DnSUr2/jmZF/ECTBlv/oelJ/KVDluBi3sjO5g5Aaw4FlAvQG6bUsMi/EV+Z2JH3aBgQAnw=='}}]}),), interrupts=()), StateSnapshot(values={}, next=('__start__',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f19bacd-ac31-662e-bfff-cdf51b29b807'}}, metadata={'source': 'input', 'step': -1, 'parents': {}}, created_at='2026-08-19T09:03:33.209440+00:00', parent_config=None, tasks=(PregelTask(id='6ec00ad2-e1da-0418-b543-533d295c20f1', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'topic': 'Coding'}),), interrupts=())]
    
'''