from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import operator

load_dotenv()

generator_llm = init_chat_model("google_genai:gemini-3.1-flash-lite")
evaluator_llm = init_chat_model("google_genai:gemini-3.5-flash")
optimizer_llm = init_chat_model("google_genai:gemini-3.6-flash")


# Tweet evaluation schema
class TweetEvaluationSchema(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        description="Final evaluation result")
    feedback: str = Field(description="Feedback for the tweet")


structured_evaluator_llm = evaluator_llm.with_structured_output(
    schema=TweetEvaluationSchema)


# state
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]


# create graph
graph = StateGraph(state_schema=TweetState)


# create nodes
def generate_tweet(state: TweetState) -> TweetState:
    messages = [
        SystemMessage(
            content="You are a funny and clever Twitter/X influncer."),
        HumanMessage(content=f"""
            Write a short, original and hilarious tweet on the topic: {state["topic"]}
            Rules:
            - Do not use question-answer format.
            - Max characters: 200.
            - User observational humor, irony, sarcasm, or cultural references.
            - Think in meme logic, punchlines or relatable takes.
            - Use simple english.
            """),
    ]

    result = generator_llm.invoke(messages).content
    return {"tweet": result, "tweet_history": [result]}


def evaluate_tweet(state: TweetState) -> TweetState:
    messages = [
        SystemMessage(
            content=
            "You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."
        ),
        HumanMessage(content=f"""
            Evaluate the following tweet:

            Tweet: "{state['tweet']}"

            Use the criteria below to evaluate the tweet:

            1. Originality – Is this fresh, or have you seen it a hundred times before?  
            2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
            3. Punchiness – Is it short, sharp, and scroll-stopping?  
            4. Virality Potential – Would people retweet or share it?  
            5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

            Auto-reject if:
            - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
            - It exceeds 280 characters
            - It reads like a traditional setup-punchline joke
            - Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

            ### Respond ONLY in structured format:
            - evaluation: "approved" or "needs_improvement"  
            - feedback: One paragraph explaining the strengths and weaknesses 
        """),
    ]

    result = structured_evaluator_llm.invoke(messages)
    return {
        "evaluation": result.evaluation,
        "feedback": result.feedback,
        "feedback_history": [result.feedback]
    }


def optimize_tweet(state: TweetState) -> TweetState:
    messages = [
        SystemMessage(
            content=
            "You punch up tweets for virality and humor based on given feedback."
        ),
        HumanMessage(content=f"""
            Improve the tweet based on this feedback:
            "{state['feedback']}"

            Topic: "{state['topic']}"
            Original Tweet:
            {state['tweet']}

            Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
        """),
    ]

    result = optimizer_llm.invoke(messages).content
    return {
        "tweet": result,
        "iteration": state["iteration"] + 1,
        "tweet_history": [result]
    }


def route_evaluation(state: TweetState):
    if (state["evaluation"] == "approved"
            or state["iteration"] >= state["max_iteration"]):
        return "approved"
    else:
        return "needs_improvement"


# add nodes
graph.add_node("generate_tweet", generate_tweet)
graph.add_node("evaluate_tweet", evaluate_tweet)
graph.add_node("optimize_tweet", optimize_tweet)

# add edges
graph.add_edge(START, "generate_tweet")
graph.add_edge("generate_tweet", "evaluate_tweet")
graph.add_conditional_edges('evaluate_tweet', route_evaluation, {
    'approved': END,
    'needs_improvement': 'optimize_tweet'
})
graph.add_edge('optimize_tweet', 'evaluate_tweet')

# compile graph
workflow = graph.compile()

# execute graph
init_state: TweetState = {
    'topic': 'Cricket',
    'iteration': 1,
    'max_iteration': 5
}
final_state: TweetState = workflow.invoke(init_state)

# print result
print(final_state)
