import os
from typing import Sequence

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

load_dotenv()
API_KEY = os.getenv('API_KEY')

llm_model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=API_KEY
)

class StateSchema(BaseModel):
    input: str
    output: str
    tipo: str = None

def realizar_calculo(state: StateSchema) -> StateSchema:
    return StateSchema(input=state.input, output="Resposta fictícia: 42")

def responder_curiosidades(state: StateSchema) -> StateSchema:
    response = llm_model.invoke([HumanMessage(content=state.input)])
    return StateSchema(input=state.input, output=response.content)

def responder_erro(state: StateSchema) -> StateSchema:
    return StateSchema(input=state.input, output="Desculpa, não entendi a pergunta.")

def classificar(state: StateSchema) -> StateSchema:
    pergunta: str = state.input.lower()
    tipo: str = 'desconhecido'
    if any(palavra in pergunta for palavra in ['somar', 'quanto é', '+', 'calcular']):
        tipo = 'calculo'
    elif any(palavra in pergunta for palavra in ['quem', 'onde', 'quando', 'por que', 'qual']):
        tipo = 'curiosidade'

    state.tipo = tipo
    return state

graph: StateGraph = StateGraph(StateSchema)
graph.add_node('classificar', classificar)
graph.add_node('realizar_calculo', realizar_calculo)
graph.add_node('responder_curiosidades', responder_curiosidades)
graph.add_node('responder_erro', responder_erro)

graph.add_conditional_edges(
    'classificar',
    lambda state: {
        'calculo': 'realizar_calculo',
        'curiosidade': 'responder_curiosidades',
        'desconhecido': 'responder_erro',
    }[state.tipo]
)

graph.set_entry_point('classificar')
graph.set_finish_point(['realizar_calculo', 'responder_curiosidades', 'responder_erro'])

export_graph: CompiledStateGraph = graph.compile()

if __name__ == '__main__':
    exemplos: Sequence[str] = [
        'Quando é 5 + 8?',
        'Quem inventou a lampada?',
        'Me diga um comando especial'
    ]

    for exemplo in exemplos:
        response = export_graph.invoke(StateSchema(input=exemplo, output=''))
        print(f'Pergunta: {exemplo}\nResposta: {response["output"]}\n\n')