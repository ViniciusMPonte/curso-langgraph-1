import os
from typing import Sequence, Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.tools import tool, BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph

load_dotenv()
API_KEY: str = os.getenv('API_KEY')

llm_model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=API_KEY
)

system_message: SystemMessage = SystemMessage(content="""
Você é um assistente. Se o usuário pedir resultado de contas de matemática,
use a ferramenta 'somar'. Caso contrário, apenas responda normalmente.
""")

@tool
def somar(valores: str) -> str:
    """somar dois números separados por vírgula"""
    try:
        a, b = map(float, valores.split(','))
        return str(a+b)
    except Exception as e:
        return f"Erro ao somar: {str(e)}"

tools: Sequence[BaseTool] = [somar]

graph: CompiledStateGraph = create_agent(
    model=llm_model,
    tools=tools,
    system_prompt=system_message
)

def extrair_resposta_final(result: dict[str, Any]) -> str:
    ai_messages: Sequence[AIMessage] = [m for m in result['messages'] if isinstance(m, AIMessage) and m.content]
    if ai_messages:
        content = ai_messages[-1].content
        if isinstance(content, list):
            return " ".join(block["text"] for block in content if block.get("type") == "text")
        return content
    return 'Nenhuma mensagem final encontrada'

if __name__ == '__main__':
    entrada1: HumanMessage = HumanMessage(content='Quanto é 8 + 5?')
    resposta1 = graph.invoke({'messages': [entrada1]})
    for mensagem in resposta1['messages']:
        print(mensagem)
    resposta_texto_1: str = extrair_resposta_final(resposta1)
    print('Resposta 1:', resposta_texto_1)

    entrada2: HumanMessage = HumanMessage(content='Quem pintou a Monalisa?')
    resposta2 = graph.invoke({'messages': [entrada2]})
    for mensagem in resposta2['messages']:
        print(mensagem)
    resposta_texto_2: str = extrair_resposta_final(resposta2)
    print('Resposta 2:', resposta_texto_2)
