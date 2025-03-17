import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from google.cloud.sql.connector import Connector, IPTypes
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import sqlalchemy
import uvicorn

# Load environment variables
load_dotenv()

# DB Configuration
INSTANCE_CONNECTION_NAME = os.environ["INSTANCE_CONNECTION_NAME"]
USER = "test"
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB = "test"

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

db_connector = Connector()


def getconn():
    conn = db_connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pymysql",
        user=USER,
        password=DB_PASSWORD,
        db=DB,
        ip_type=IPTypes.PRIVATE,
    )
    return conn


# create connection pool
engine = sqlalchemy.create_engine(
    "mysql+pymysql://",
    creator=getconn,
)

# Create messages table if not exists - this will be managed by SQLChatMessageHistory
with engine.connect() as conn:
    conn.execute(
        sqlalchemy.text(
            """
        CREATE TABLE IF NOT EXISTS message_store (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            message JSON NOT NULL,
            INDEX idx_session_id (session_id)
        )
    """
        )
    )
    conn.commit()

# Initialize model and prompt
model = ChatOpenAI(model="gpt-4o")

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
runnable = prompt | model

chat_message_history = {}


# Function to get history manager for a thread
def get_message_history(thread_id: str) -> SQLChatMessageHistory:
    if thread_id not in chat_message_history:
        chat_message_history[thread_id] = SQLChatMessageHistory(
            session_id=thread_id,
            table_name="message_store",
            connection=engine,
        )
    return chat_message_history[thread_id]


# FastAPI setup
api = FastAPI()


class MessageRequest(BaseModel):
    messages: list
    thread_id: str
    speaker_name: str | None = None


@api.post("/messages")
async def post_messages(request: MessageRequest):
    logger.info(
        f"Received message request - Thread ID: {request.thread_id}, Speaker: {request.speaker_name}"
    )

    # Format messages with speaker information
    formatted_messages = []
    for msg in request.messages:
        formatted_msg = f"(Speaker: {request.speaker_name or 'Anonymous'})\n{msg}"
        formatted_messages.append(formatted_msg)

    # Create RunnableWithMessageHistory using the chat history
    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_message_history,
        input_messages_key="messages",
        history_messages_key="history",
    )

    # Invoke the runnable with the formatted messages
    response = with_message_history.invoke(
        {"messages": [HumanMessage(content=msg) for msg in formatted_messages]},
        config={"configurable": {"session_id": request.thread_id}},
    )

    return {"response": response}


if __name__ == "__main__":
    logger.info("Starting Dabini Engine server on port 8080...")
    uvicorn.run(api, host="0.0.0.0", port=8080)
