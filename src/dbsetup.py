import traceback
from os import getenv
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from models import Base
from logger import get_logger
from sqlalchemy.exc import SQLAlchemyError


# Initialize the logger
logger = get_logger()

# Load environment variables
load_dotenv()

db_url = getenv('DB_URL')
if not db_url:
    logger.critical('DB_URL is not set in the environment variables')
    raise EnvironmentError('DB_URL environment variable is required')
try:
    # Create the engine
    logger.info('Creating the engine')
    engine = create_engine(db_url, echo=True)
    logger.debug(f'Database engine created with URL: {db_url.split("://")[0]}')
except Exception as e:
    logger.critical(f'Failed to create the engine: {str(e)}')
    raise


# Initialize the database
def init_db():
    '''
    This function initializes the database.
    It creates all the tables in the database if they don't exist.
    '''
    try:
        logger.info('Initializing the database')
        Base.metadata.create_all(bind=engine)
        logger.info('Database initialized successfully')
    except SQLAlchemyError as e:
        logger.error(f'SQLAlchemy error during database initialization: {str(e)}')
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.critical(f'Unexpected error during database initialization: {str(e)}')
        logger.critical(traceback.format_exc())
        raise

# Get the database
def get_db():
    '''
    This function provides a database session.
    It ensures that the session is properly closed after use.
    '''
    session = None
    try:
        logger.info('Creating a new database session')
        session = Session(engine, expire_on_commit=False)
        yield session
        logger.debug('Database session yielded successfully')
    except SQLAlchemyError as e:
        logger.error(f'SQLAlchemy error during database session: {str(e)}')
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.critical(f'Unexpected error during database session: {str(e)}')
        logger.critical(traceback.format_exc())
        raise
    finally:
        if session:
            logger.info('Closing the database session')
            session.close()
            logger.info('Database session closed successfully')
