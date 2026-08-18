from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages

load_dotenv()

llm = init_chat_model('google_genai:gemini-3.1-flash-lite')


# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    # BaseMessage: parent class of all Human,System,Ai and Tool messages

    # Merges two lists of messages, updating existing messages by ID.
    # By default, this ensures the state is "append-only", unless the new message has the same ID as an existing message.
    # Optimized to work with BaseMessages


# graph
graph = StateGraph(state_schema=ChatState)


# define nodes
def chat_node(state: ChatState) -> ChatState:
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


# add nodes
graph.add_node('chat_node', chat_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile graph
chatbot = graph.compile()

# execute graph
# init_state: ChatState = {
#     'messages': [HumanMessage(content='What is the capital of India')]
# }
# final_state: ChatState = chatbot.invoke(init_state)

while True:
    user_message = input('You:')
    # print(user_message)
    if user_message.strip().lower() in ['exit', 'quit', 'bye']: break
    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_message)]})
    print('AI: ', response['messages'][-1].content[0]['text'])

# print result
# print(final_state)
'''
NOTES:
    while True:
        user_message = input('You:')
        # print(user_message)
        if user_message.strip().lower() in ['exit', 'quit', 'bye']: break
        response = chatbot.invoke(
            {'messages': [HumanMessage(content=user_message)]})
        print('AI: ', response['messages'][-1].content[0]['text'])

    At this point AI is unable to remember our context, i.e. the list of chat messages.
    The problem is in the loop we're using invoke() of chatbot, that is why the execution of the graph is started and ended in one loop. So when we type new query it again starting new state and ends it.
    Meaning, the list of messages is adding message at start of loop and resetting it when same loop ends, at new loop the list of message is empty.
'''
