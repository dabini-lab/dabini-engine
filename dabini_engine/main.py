import getpass
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import BaseModel

# Load environment variables
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# Initialize model and workflow
model = ChatOpenAI(model="gpt-4o-mini")
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    system_prompt = (
        "너의 이름은 다빈이야. " "Answer all questions to the best of your ability."
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


@api.post("/messages")
async def post_messages(request: MessageRequest):
    input_messages = [HumanMessage(content=msg) for msg in request.messages]
    output = app.invoke(
        {"messages": input_messages},
        config={
            "configurable": {
                "thread_id": request.thread_id,
                "user_id": request.speaker_name,
            }
        },
    )
    last_message = output["messages"][-1]
    return {"response": last_message}


if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8080)
