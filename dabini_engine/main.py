from dotenv import load_dotenv
import uvicorn

from dabini_engine.db import initialize_db
from dabini_engine.api import api
from dabini_engine.services import setup_services

# Load environment variables and setup
load_dotenv()

# Initialize database
initialize_db()

# Setup model and history handlers
setup_services()

# Main execution
if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8080)
