from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = init_chat_model('google_genai:gemini-3.1-flash-lite')


# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# create checkpoint
checkpointer = MemorySaver()

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
chatbot = graph.compile(checkpointer=checkpointer)

thread_id = 1
config = {'configurable': {'thread_id': thread_id}}

# execute graph
init_state: ChatState = {
    'messages':
    [HumanMessage(content='Write 200 word essay on capital of india')]
}

# stream = chatbot.stream(init_state, config=config, stream_mode='messages') -> stream is a generator object which generates values on the fly

# stream generator contains two objects, message_chunk and metadata
for message_chunk, metadata in chatbot.stream(init_state,
                                              config=config,
                                              stream_mode='messages'):
    if message_chunk.content:
        print(message_chunk.content, end=' ',
              flush=True)  # for end we used space
