from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage

from dabini_engine.db import get_db_engine

# For backwards compatibility or as a singleton instance
_default_service = None


class MessageService:
    def __init__(self, model_name="gpt-4o"):
        model = ChatOpenAI(model=model_name)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "너의 이름은 다빈이야. "
                    "You will converse naturally with speakers marked as '(Speaker: name)'. "
                    "Answer directly without adding any name tags or prefixes.",
                ),
                MessagesPlaceholder(variable_name="history", n_messages=100),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.runnable = prompt | model

    def _get_message_history(
        self, session_id: str, connection_provider=None
    ) -> SQLChatMessageHistory:
        # Get connection from provider or use default
        if connection_provider is None:
            connection_provider = get_db_engine

        return SQLChatMessageHistory(
            session_id=session_id,
            table_name="message_store",
            connection=connection_provider(),
        )

    def process_messages(self, messages, session_id, speaker_name=None):
        # Format messages with speaker information
        formatted_messages = []
        for msg in messages:
            formatted_msg = f"(Speaker: {speaker_name or 'Anonymous'})\n{msg}"
            formatted_messages.append(formatted_msg)

        # Create RunnableWithMessageHistory using the chat history
        with_message_history = RunnableWithMessageHistory(
            self.runnable,
            self._get_message_history,
            input_messages_key="messages",
            history_messages_key="history",
        )

        # Invoke the runnable with the formatted messages
        response = with_message_history.invoke(
            {"messages": [HumanMessage(content=msg) for msg in formatted_messages]},
            config={"configurable": {"session_id": session_id}},
        )

        return response


def get_message_service():
    global _default_service
    if _default_service is None:
        _default_service = MessageService()
    return _default_service


# Legacy functions that use the default service
def setup_services():
    get_message_service()  # Initialize the default service


def process_messages(messages, session_id, speaker_name=None):
    return get_message_service().process_messages(messages, session_id, speaker_name)
