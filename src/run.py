from os import getenv
from dotenv import load_dotenv
from discordaibot import DiscordAIBot
from sqlalchemy.orm import Session
from dbsetup import init_db, get_db

def main():
    # Load environment variables
    load_dotenv()

    # Initialize the database
    init_db()

    # Get the database session
    db_session: Session = next(get_db())

    # Get necessary parameters from environment variables
    discord_token = getenv('DISCORD_AI_BOT_TOKEN')
    huggingface_token = getenv('HUGGING_FACE_HUB_TOKEN')
    embedding_model = getenv('EMBEDDING_MODEL')
    llm_model = getenv('LLM_MODEL')
    threshold = float(getenv('SIMILARITY_THRESHOLD', '0.85'))

    # Initialize the bot
    bot = DiscordAIBot(discord_token, huggingface_token, embedding_model, llm_model, threshold, db_session)

    # Run the bot
    bot.run()

if __name__ == "__main__":
    main()