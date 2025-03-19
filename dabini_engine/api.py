from fastapi import FastAPI
from pydantic import BaseModel

from dabini_engine.services import process_messages


# Class definitions
class MessageRequest(BaseModel):
    messages: list
    session_id: str
    speaker_name: str | None = None


# FastAPI setup
api = FastAPI()


@api.post("/messages")
async def post_messages(request: MessageRequest):
    response = process_messages(
        messages=request.messages,
        session_id=request.session_id,
        speaker_name=request.speaker_name,
    )

    return {"response": response}
