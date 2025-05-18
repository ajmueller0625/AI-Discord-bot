import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from models import Base

load_dotenv()

# Create the engine
engine = create_engine(os.getenv('DB_URL'), echo=True)

# Initialize the database
def init_db():
    Base.metadata.create_all(bind=engine)

# Get the database
def get_db():
    with Session(engine, expire_on_commit=False) as session:
        yield session
