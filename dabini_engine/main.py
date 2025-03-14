import getpass
import os
import logging

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import BaseModel

# Load environment variables
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize model and workflow
model = ChatOpenAI(model="gpt-4o")
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    system_prompt = (
        "너의 이름은 다빈이야. "
        "You will converse naturally with speakers marked as '(Speaker: name)'. "
        "Answer directly without adding any name tags or prefixes."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
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
    logger.info(f"Generated response for thread {request.thread_id}")
    return {"response": last_message}


if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8080)
