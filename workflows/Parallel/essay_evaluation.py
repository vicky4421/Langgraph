'''
Parallel Workflow
Structured Output: in text and int
reducer: merging outputs from parallel nodes
'''

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator # python module

load_dotenv()

llm = init_chat_model('google_genai:gemini-3.1-flash-lite')

# schema
class EvaluationScheme(BaseModel):
    feedback: str = Field(description='Detailed feedback for essay')
    score: int = Field(description='Evaluation score out of 10', ge=0, le=10)

# structured model
structured_model = llm.with_structured_output(schema=EvaluationScheme)

# essay
essay = """ India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.

One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India’s doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.

To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

India’s demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India’s AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society."""

# define state
class EvaluationState(TypedDict):
    essay: str
    lang_feedbck: str
    analysis_feedbck: str
    clarity_feedbck: str
    summary_feedbck: str
    # individual scores for lang, analysis and clarity of thought for essay given by llm
    # these scores are generated parallely and can be overwrite when stored in one var
    # so we use Reducer func (add) to not overwrite
    # suppose lang returns 8, analysis return 7 and clarity returns 9 they'll return it in list like [8],[7],[9]
    # to merge list into one we use + operation so it will look like [8] + [7] + [9 = [8, 7, 9]]
    # but we can't use + operator as param, for that python provides us operator module
    individual_score: Annotated[list[int], operator.add]
    avg_score: float

# create graph
graph = StateGraph(EvaluationState)

# create nodes
def evaluate_lang(state: EvaluationState) -> EvaluationState:
    prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}'
    result = structured_model.invoke(prompt)
    return {'lang_feedbck': result.feedback, 'individual_score': [result.score]}  # score should be in list

def evaluate_analysis(state: EvaluationState) -> EvaluationState:
    prompt = f'Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}'
    result = structured_model.invoke(prompt)
    return {'analysis_feedbck': result.feedback, 'individual_score': [result.score]}

def evaluate_clarity(state: EvaluationState) -> EvaluationState:
    prompt = f'Evaluate the thaught of clarity of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}'
    result = structured_model.invoke(prompt)
    return {'clarity_feedbck': result.feedback, 'individual_score': [result.score]}

def final_evaluation(state: EvaluationState) -> EvaluationState:
    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state["lang_feedbck"]} \n depth of analysis feedback - {state["analysis_feedbck"]} \n clarity of thought feedback - {state["clarity_feedbck"]}'
    summary_feedback = llm.invoke(prompt).content

    avg_score = sum(state['individual_score'])/len(state['individual_score'])

    return {'summary_feedbck': summary_feedback, 'avg_score': avg_score}

# add nodes
graph.add_node('evaluate_lang', evaluate_lang)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_clarity', evaluate_clarity)
graph.add_node('final_evaluation', final_evaluation)

# add edges
graph.add_edge(START, 'evaluate_clarity')
graph.add_edge(START, 'evaluate_lang')
graph.add_edge(START, 'evaluate_analysis')

graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_lang', 'final_evaluation')
graph.add_edge('evaluate_clarity', 'final_evaluation')

graph.add_edge('final_evaluation', END)

# compile graph
workflow = graph.compile()

# execute graph
init_state: EvaluationState = {'essay': essay}
final_state: EvaluationState = workflow.invoke(init_state)

# print result
print(final_state)