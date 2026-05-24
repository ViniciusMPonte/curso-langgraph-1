from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph import graph
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('API_KEY')

llm_model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=API_KEY
)

class StateSchema(BaseModel):
    input: str
    output: str

def responder(state: StateSchema):
    response: AIMessage = llm_model.invoke([HumanMessage(content=state.input)])
    return StateSchema(input=state.input, output=str(response.content))

graph: StateGraph = StateGraph(StateSchema)
graph.add_node('responder', responder)
graph.set_entry_point('responder')
graph.set_finish_point('responder')

export_graph: CompiledStateGraph = graph.compile()

if __name__ == "__main__":
    result = export_graph.invoke(StateSchema(input='Quem descobriu o Brasil?', output=''))
    print(result)