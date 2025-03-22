from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

from dabini_engine.db import initialize_db
from dabini_engine.routers.messages import router as messages_router
from dabini_engine.services import setup_services

HOST = "0.0.0.0"
PORT = 8080

app = FastAPI()
app.include_router(messages_router)


def init():
    # Load environment variables and setup
    load_dotenv()

    # Initialize database
    initialize_db()

    # Setup model and history handlers
    setup_services()


def dev():
    init()
    uvicorn.run("dabini_engine.main:app", host=HOST, port=PORT, reload=True)


def main():
    init()
    uvicorn.run("dabini_engine.main:app", host=HOST, port=PORT)
