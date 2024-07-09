from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.agents.format_scratchpad.openai_tools import (format_to_openai_tool_messages,)
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.messages import AIMessage, HumanMessage
import uvicorn
from langchain.pydantic_v1 import BaseModel as LangChainBaseModel
from pydantic import BaseModel
from ai_with_memory import chain_with_history
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_community.drive import GoogleDriveLoader
import os
from test import fetch_and_process_pdf
#from retrieve_from_google_drive import docs
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
loader=fetch_and_process_pdf('1PI1pAOriyWQPOpLvUJcgKpCHTrDe12c7')
docs=loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()


logging.basicConfig(level=logging.INFO)
store={}
def accessing_history(session_id:str):
    if session_id not in store:
        store[session_id] = []
    return store[session_id]
@tool
def tavilysearch(search: str) ->str:
    """Returns a search outcome from tavily"""
    Tavily=TavilySearchResults(max_results=3)
    return Tavily.invoke(search)
tools = [tavilysearch]
"""retriever_tool = create_retriever_tool(
    retriever,
    "search_state_of_union",
    "Searches and returns excerpts from the 2022 State of the Union.",
)
retriever_tools=[retriever_tool]
two_tools=[retriever_tool,tavilysearch]"""




my_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that will answer, response to the user's input."),
    ("user", "{question}"),
])

agent_prompt=ChatPromptTemplate.from_messages(    [
        (
            "system",
            "You are a helpful assistant that will answer, and response to the user's input."
        ),
        ("user", "{question}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
agent_history_prompt=ChatPromptTemplate.from_messages(    [
        (
            "system",
            "You are a helpful assistant that will answer, response to the user's input, accroding to the user's request and the chat {history} and utilize the tools that you are given, make sure that before answering to have tried all the tools."
        ),
        ("user", "{question}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])



llm = ChatOpenAI(temperature=1.2,openai_api_key=OPENAI_API_KEY)
chain = my_prompt | llm
llm_with_tavily_tools = llm.bind(tools=[convert_to_openai_tool(tool)for tool in tools])
#llm_with_retriever_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in retriever_tools])
#llm_with_tavily_and_retriever_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in two_tools])

tavily_agent = (
    {
        "question": lambda x: x["question"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | agent_prompt
    | llm_with_tavily_tools
    | OpenAIToolsAgentOutputParser()
)
tavily_agent_history = (
    {
        "question": lambda x: x["question"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        "history": lambda x: x["history"]
    }
    | agent_history_prompt
    | llm_with_tavily_tools
    | OpenAIToolsAgentOutputParser()
)
"""retriever_agent = (
    {
        "question": lambda x: x["question"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | agent_prompt
    | llm_with_retriever_tools
    | OpenAIToolsAgentOutputParser()
)
retriever_agent_history = (
    {
        "question": lambda x: x["question"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        "history": lambda x: x["history"]
    }
    | agent_history_prompt
    | llm_with_retriever_tools
    | OpenAIToolsAgentOutputParser()
)

both_tool_agent = ({
        "question": lambda x: x["question"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        "history": lambda x: x["history"]
    }
    | agent_history_prompt
    | llm_with_tavily_and_retriever_tools
    | OpenAIToolsAgentOutputParser()
)

"""



tavily_agent_executor = AgentExecutor(agent=tavily_agent, tools=tools, verbose=True)
tavily_agent_executor_history =  AgentExecutor(agent=tavily_agent_history, tools=tools, verbose=True)

"""retriever_agent_executor = AgentExecutor(agent=retriever_agent, tools=retriever_tools, verbose=True)
retriever_agent_executor_history =  AgentExecutor(agent=retriever_agent_history, tools=retriever_tools, verbose=True)

retriever_tavily_agent_executor_history =AgentExecutor(agent=both_tool_agent, tools=two_tools, verbose=True)"""


tavily_agent_with_history = RunnableWithMessageHistory(
    tavily_agent_executor_history,
    input_messages_key="question",
    history_messages_key="history",
    get_session_history=accessing_history
)
"""retriever_agent_with_history=RunnableWithMessageHistory(
    retriever_agent_executor_history,
    input_messages_key="question",
    history_messages_key="history",
    get_session_history=accessing_history
)
"""





app = FastAPI(title="LangChain APP")
class QueryRequest(BaseModel):
    question: str
    session: str = None
class Input(LangChainBaseModel):
    question: str
class Output(LangChainBaseModel):
    output: str


add_routes(app,chain.with_types(input_type=Input),playground_type="default", path="/Xassistant")
add_routes(app, tavily_agent_executor.with_types(input_type=Input, output_type=Output).with_config({"run_name": "Sagent"}), path="/Sagent")
#add_routes(app, retriever_agent_executor.with_types(input_type=Input, output_type=Output).with_config({"run_name": "Ragent"}), path="/Ragent")




@app.post("/query/Xassitant")
async def query_model(request: QueryRequest):
    input_data = request.question
    response = chain.invoke({"input": input_data})
    return {response.content}


@app.post("/query/Xassistant-with-memory")
async def query_model(request: QueryRequest):
    input_data = request.question
    session_id =  request.session
    response = chain_with_history.invoke({"question": input_data}, config={"configurable": {"session_id": session_id}})
    return {response.content}


@app.post("/query/Sagent")
async def agent_model(request: QueryRequest):
    input_data=request.question
    response=tavily_agent_executor.stream({"question": input_data})
    response_list=list(response)
    final_content=response_list[-1]
    print(final_content.get('messages')[0].content)
    return {final_content.get('messages')[0].content}
"""@app.post("/google-login")
async def login():
    loader = GoogleDriveLoader(
    folder_id="1PI1pAOriyWQPOpLvUJcgKpCHTrDe12c7",
    file_types=["pdf"],
    recursive=True,
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    return{"Success!"}
@app.post("/query/Ragent")
async def agent_model(request: QueryRequest):
    input_data=request.question
    response=retriever_agent_executor.stream({"question": input_data})
    response_list=list(response)
    final_content=response_list[-1]
    print(final_content.get('messages')[0].content)
    return {final_content.get('messages')[0].content}
"""
@app.post("/query/Sagent-memory")
async def query_model(request: QueryRequest):
    input_data = request.question
    session_id =  request.session
    history = accessing_history(session_id)
    response = tavily_agent_executor_history.invoke({"question": input_data, "history":history }, config={"configurable": {"session_id": session_id}})
    history.extend([
        HumanMessage(content=input_data),
        AIMessage(content=response["output"]),
    ])
    print(response)
    return {response["output"]}
"""
@app.post("/query/Ragent-memory")
async def query_model(request: QueryRequest):
    input_data = request.question
    session_id =  request.session
    history = accessing_history(session_id)
    response = retriever_agent_executor_history.invoke({"question": input_data,"history":history }, config={"configurable": {"session_id": session_id}})
    history.extend([
        HumanMessage(content=input_data),
        AIMessage(content=response["output"]),
    ])
    print(response)
    return {response["output"]}

@app.post("/query/RSagent-memory")
async def query_model(request: QueryRequest):
    input_data = request.question
    session_id =  request.session
    history = accessing_history(session_id)
    response = retriever_tavily_agent_executor_history.invoke({"question": input_data,"history":history }, config={"configurable": {"session_id": session_id}})
    history.extend([
        HumanMessage(content=input_data),
        AIMessage(content=response["output"]),
    ])
    print(response)
    return {response["output"]}
"""
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)