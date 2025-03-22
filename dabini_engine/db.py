import os
import logging
from google.cloud.sql.connector import Connector
import sqlalchemy

# Set up logging
logger = logging.getLogger(__name__)

# DB Configuration
INSTANCE_CONNECTION_NAME = os.environ["INSTANCE_CONNECTION_NAME"]
USER = "test"
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB = "test"

# Initialize connector
db_connector = Connector()


def getconn():
    conn = db_connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pg8000",
        user=USER,
        password=DB_PASSWORD,
        db=DB,
    )
    return conn


# Create connection pool
db_engine = sqlalchemy.create_engine(
    "postgresql+pg8000://",
    creator=getconn,
)


def get_db_engine():
    return db_engine


# Create messages table if not exists
def initialize_db():
    with db_engine.connect() as conn:
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
    logger.info("Database initialized successfully")
