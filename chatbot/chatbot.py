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

    # BaseMessage: parent class of all Human,System,Ai and Tool messages

    # Merges two lists of messages, updating existing messages by ID.
    # By default, this ensures the state is "append-only", unless the new message has the same ID as an existing message.
    # Optimized to work with BaseMessages


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

# execute graph
# init_state: ChatState = {
#     'messages': [HumanMessage(content='What is the capital of India')]
# }
# final_state: ChatState = chatbot.invoke(init_state)

thread_id = '1'
while True:
    user_message = input('You:')
    # print(user_message)
    if user_message.strip().lower() in ['exit', 'quit', 'bye']: break
    config = {'configurable': {'thread_id': thread_id}}
    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_message)]}, config=config)
    print('AI: ', response['messages'][-1].content[0]['text'])

print(chatbot.get_state(config=config))

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
    Meaning, the list of messages is adding message at start of loop and resetting it when same loop ends, at new loop  the list of message is empty.

    Output with persistance:
        You:i'm vivek
        Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to  use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead,    we recommend to use AFC in Chat.send_message_stream.
        AI:  Hello, Vivek! It's nice to meet you. How are you doing today? Is there anything I can help you with?
        You:what's my name?
        AI:  Your name is Vivek!
        You:whats 10 + 3541
        AI:  10 + 3541 = 3551
        You:multiply it with 10
        AI:  3551 multiplied by 10 is **35,510**.
        You:bye
        StateSnapshot(values={'messages': [HumanMessage(content="i'm vivek", additional_kwargs={}, response_metadata={}, id='326f6fcc-736b-422f-b2e3-95b4e291a6e3'), AIMessage(content=[{'type': 'text', 'text': "Hello, Vivek! It's nice to meet you. How are you doing today? Is there anything I can help you with?", 'extras': {'signature': 'EnEKbwERTTIPcnUBq2eAPgg1Uf41F/hwHSI5Ih5qTVTTYLAHK47l6XT9l04nO4ljjXCpK1dJofyDyhqrVGyWTm3PkJe0BIYzBRq28Exkic9uwrOv878HoI83mFFqQ6nyb7oHQmrQkAoXo5VWIaZAznsgqw=='}}], additional_kwargs={}, response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-3.1-flash-lite', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--01a015d2-d16f-7e32-a16d-61b1e6e3e3db-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 6, 'output_tokens': 27, 'total_tokens': 33, 'input_token_details': {'cache_read': 0}}), HumanMessage(content="what's my name?", additional_kwargs={}, response_metadata={}, id='539026a3-56ed-493d-8691-d08a26620dac'), AIMessage(content=[{'type': 'text', 'text': 'Your name is Vivek!', 'extras': {'signature': 'EnEKbwERTTIPsJ7/TNNhTq1g+pCuVCdxtYT3MXglRKhi9S4FNnjPWUtnDnJiQb+wYny4poF6wBwEcIpbn6z9K0fOgkcOdlgKpg18x+TrizPxxWekdUa/qcGUG2xh1rZ6aROKRcnYPV2Ak2PV0hqBqVYeUg=='}}], additional_kwargs={}, response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-3.1-flash-lite', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--01a015d2-e58c-7083-91dd-d0e3067276b1-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 41, 'output_tokens': 5, 'total_tokens': 46, 'input_token_details': {'cache_read': 0}}), HumanMessage(content='whats 10 + 3541', additional_kwargs={}, response_metadata={}, id='78912d52-da0c-4d18-aa44-2f5eb4a6e275'), AIMessage(content=[{'type': 'text', 'text': '10 + 3541 = 3551', 'extras': {'signature': 'EnEKbwERTTIPWFofl/SOoxkiYn1zUdEfY9klvOV3JCnWR3tdF/42XNbhQ/2nAL/+XWhGp+vJV6dwYWmHSWWPiinTU4nzpzzBwEE2ckTzdXxHjZwxwQ5s3SiWq+rSBNPBYC5uAGCd0U8PcEkY5MYMXeSyQA=='}}], additional_kwargs={}, response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-3.1-flash-lite', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--01a015d3-0abf-7520-b74a-872f3d164761-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 58, 'output_tokens': 14, 'total_tokens': 72, 'input_token_details': {'cache_read': 0}}), HumanMessage(content='multiply it with 10', additional_kwargs={}, response_metadata={}, id='374ea002-0662-4f57-8ae3-9498dca312a8'), AIMessage(content=[{'type': 'text', 'text': '3551 multiplied by 10 is **35,510**.', 'extras': {'signature': 'EnEKbwERTTIPPqc7oPqrJNz1KHZJqViBGPSFN4AZ1wb+OaXYZ+V8j/AOfLI3bc05qW6SqaBJvi1GqTDUQbw7lu7r1n+Q3yCeAed/Sc+OdVU8rLq2uXDbi4aTKgyrJqcs4C3b728MOSHDSp6YM7FjRvycnA=='}}], additional_kwargs={}, response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-3.1-flash-lite', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--01a015d3-890b-73d3-b20c-73ce5599c326-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 80, 'output_tokens': 18, 'total_tokens': 98, 'input_token_details': {'cache_read': 0}})]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '',    'checkpoint_id': '1f19b26a-ca96-681a-800a-d1128c605e55'}}, metadata={'source': 'loop', 'step': 10, 'parents': {}},     created_at='2026-08-18T17:03:03.304157+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f19b26a-a9a5-6858-8009-69370598dea8'}}, tasks=(), interrupts=())
'''
