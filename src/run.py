import traceback
import sys
from os import getenv
from dotenv import load_dotenv
from discordaibot import DiscordAIBot
from sqlalchemy.orm import Session
from dbsetup import init_db, get_db
from logger import get_logger

def main():
    # Initialize the logger
    logger = get_logger()
    logger.info('Starting the AI Tutor Bot...')
    
    try:
        # Load environment variables
        load_dotenv()
        logger.info('Environment variables loaded successfully')

        required_env_vars = [
            'DISCORD_AI_BOT_TOKEN',
            'HUGGING_FACE_HUB_TOKEN',
            'EMBEDDING_MODEL',
            'LLM_MODEL',
            'SIMILARITY_THRESHOLD'
        ]

        missing_vars = [var for var in required_env_vars if not getenv(var)]
        if missing_vars:
            logger.error(f'Missing required environment variables: {", ".join(missing_vars)}')
            raise ValueError(f'Missing required environment variables: {", ".join(missing_vars)}')
        
        try:
            # Initialize the database
            logger.info('Initializing the database...')
            init_db()

            # Get the database session
            logger.info('Getting the database session...')
            db_session: Session = next(get_db())

            # Get necessary parameters from environment variables
            discord_token = getenv('DISCORD_AI_BOT_TOKEN')
            huggingface_token = getenv('HUGGING_FACE_HUB_TOKEN')
            embedding_model = getenv('EMBEDDING_MODEL')
            llm_model = getenv('LLM_MODEL')

            # Get the similarity threshold
            threshold = float(getenv('SIMILARITY_THRESHOLD', '0.90'))
            try:
                threshold = float(threshold)
                if threshold < 0.0 or threshold > 1.0:
                    raise ValueError(f'Similarity threshold must be between 0.0 and 1.0, got {threshold}')
            except ValueError as e:
                logger.error(f'Invalid similarity threshold: {e}, using default value 0.90')
                threshold = 0.90

            # Log configuration (without sensitive information)
            logger.info(f'Configuration: embedding_model={embedding_model}, llm_model={llm_model}, threshold={threshold}')

            # Initialize the bot
            logger.info('Initializing the bot...')
            bot = DiscordAIBot(discord_token, huggingface_token, embedding_model, llm_model, threshold, db_session)

            # Run the bot
            logger.info('Starting the bot...')
            bot.run()

        except Exception as e:
            logger.error(f'Error initializing the bot: {str(e)}')
            logger.error(traceback.format_exc())
            sys.exit(1)

    except Exception as e:
        if 'logger' in locals():
            logger.critical(f'Critical error in main function: {str(e)}')
            logger.critical(traceback.format_exc())
        else:
            print(f'Critical error before logger initialization: {str(e)}')
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()