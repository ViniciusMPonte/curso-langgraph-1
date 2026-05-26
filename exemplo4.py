import os
from typing import Sequence

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool, BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.state import CompiledStateGraph

load_dotenv()
API_KEY: str = os.getenv('API_KEY')

llm_model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=API_KEY
)

system_message: SystemMessage = SystemMessage(content="""
Você é um pesquisador muito sarcástico e irônico.
Use ferramenta 'search' sempre que necessário, especialmente
para perguntas que exigem informações da web
""")

@tool('search')
def search_web(query: str = '') -> str:
    """
    Busca informações na web baseada na consulta fornecida.

    Args:
        query: Termos para buscar dados na web

    Returns:
        As informações encontradas na web ou uma mensagem indicando
        que nenhuma informação foi encontrada.
    """
    tavily_search: TavilySearchResults = TavilySearchResults(max_results=3)
    tavily_docs = tavily_search.invoke(query)
    return tavily_docs

tools: Sequence[BaseTool] = [search_web]

graph: CompiledStateGraph = create_agent(
    model=llm_model,
    tools=tools,
    system_prompt=system_message
)
