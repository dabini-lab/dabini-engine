from fastapi import APIRouter
from pydantic import BaseModel

from dabini_engine.services import process_messages

router = APIRouter(prefix="/messages")


# Class definitions
class MessageRequest(BaseModel):
    messages: list
    session_id: str
    speaker_name: str | None = None


@router.post("")
async def post_messages(request: MessageRequest):
    response = process_messages(
        messages=request.messages,
        session_id=request.session_id,
        speaker_name=request.speaker_name,
    )

    return {"response": response}
