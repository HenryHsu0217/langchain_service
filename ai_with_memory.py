from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import os
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that will answer, response to the user's input"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])
chain = prompt | ChatOpenAI(api_key=OPENAI_API_KEY)
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
)
