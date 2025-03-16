import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from google.cloud.sql.connector import Connector, IPTypes
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import BaseModel
import sqlalchemy
import uvicorn

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize model and workflow
model = ChatOpenAI(model="gpt-4o")
workflow = StateGraph(state_schema=MessagesState)

# DB Configuration
INSTANCE_CONNECTION_NAME = "dabini:asia-northeast3:dabini-dev"
USER = "test"
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB = "test"
PROJECT_ID = "dabini"

connector = Connector()

def getconn():
    conn = connector.connect(
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

# Create messages table if not exists
with engine.connect() as conn:
    conn.execute(sqlalchemy.text("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id SERIAL PRIMARY KEY,
            thread_id VARCHAR(255) NOT NULL,
            speaker_name VARCHAR(255),
            content TEXT NOT NULL,
            role VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_thread_id (thread_id),
            INDEX idx_created_at (created_at)
        )
    """))
    conn.commit()

# Add after DB Configuration section
thread_last_timestamps = {}

def get_chat_history(thread_id: str) -> list:
    with engine.connect() as conn:
        last_timestamp = thread_last_timestamps.get(thread_id)
        if last_timestamp:
            result = conn.execute(
                sqlalchemy.text(
                    """SELECT speaker_name, content, role, created_at 
                       FROM chat_messages 
                       WHERE thread_id = :thread_id 
                       AND created_at > :last_timestamp 
                       ORDER BY created_at"""
                ),
                {"thread_id": thread_id, "last_timestamp": last_timestamp}
            )
        else:
            result = conn.execute(
                sqlalchemy.text(
                    """SELECT speaker_name, content, role, created_at 
                       FROM chat_messages 
                       WHERE thread_id = :thread_id 
                       ORDER BY created_at DESC LIMIT 100"""
                ),
                {"thread_id": thread_id}
            )
        
        messages = []
        latest_timestamp = None
        rows = list(result)

        # Reverse the rows if we fetched with DESC order
        if not last_timestamp:
            rows = rows[::-1]
            
        for row in rows:
            if row.role == "system":
                messages.append(SystemMessage(content=row.content))
            else:
                content = f"(Speaker: {row.speaker_name})\n{row.content}" if row.speaker_name else row.content
                messages.append(HumanMessage(content=content))
            latest_timestamp = row.created_at
        
        if latest_timestamp:
            thread_last_timestamps[thread_id] = latest_timestamp
            
        return messages

def save_message(thread_id: str, content: str, role: str, speaker_name: str = None):
    with engine.connect() as conn:
        current_time = datetime.now()
        conn.execute(
            sqlalchemy.text(
                """INSERT INTO chat_messages (thread_id, speaker_name, content, role, created_at) 
                   VALUES (:thread_id, :speaker_name, :content, :role, :created_at)"""
            ),
            {
                "thread_id": thread_id,
                "speaker_name": speaker_name,
                "content": content,
                "role": role,
                "created_at": current_time
            }
        )
        thread_last_timestamps[thread_id] = current_time
        conn.commit()

# Define the function that calls the model
def call_model(state: MessagesState, config: dict) -> dict:
    system_prompt = (
        "너의 이름은 다빈이야. "
        "You will converse naturally with speakers marked as '(Speaker: name)'. "
        "Answer directly without adding any name tags or prefixes."
    )
    thread_id = config["metadata"]["thread_id"]
    messages = [SystemMessage(content=system_prompt)] + get_chat_history(thread_id) + state["messages"]

    response = model.invoke(messages)
    return {"messages": response}

# Define the node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# FastAPI setup
api = FastAPI()


class MessageRequest(BaseModel):
    messages: list
    thread_id: str
    speaker_name: str | None = None

# Initialize message template
human_template = PromptTemplate.from_template("(Speaker: {speaker})\n{message}")

@api.post("/messages")
async def post_messages(request: MessageRequest):
    logger.info(f"Received message request - Thread ID: {request.thread_id}, Speaker: {request.speaker_name}")
    
    input_messages = [
        HumanMessage(
            content=human_template.format(
                speaker=request.speaker_name or "Anonymous",
                message=msg
            )
        )
        for msg in request.messages
    ]

    output = app.invoke(
        {"messages": input_messages},
        config={
            "configurable": {
                "thread_id": request.thread_id,
            }
        },
    )
    last_message = output["messages"][-1]

    # Save incoming messages to DB
    for msg in request.messages:
        save_message(
            thread_id=request.thread_id,
            content=msg,
            role="human",
            speaker_name=request.speaker_name
        )
    
    # Save assistant's response to DB
    save_message(
        thread_id=request.thread_id,
        content=last_message.content,
        role="assistant",
        speaker_name="Dabini"
    )
    
    logger.info(f"Generated response for thread {request.thread_id}")
    return {"response": last_message}


if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8080)
