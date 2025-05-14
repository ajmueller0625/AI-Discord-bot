import os
import sys
from discord import Intents, Message
from discord.ext import commands
from discord.ext.commands import Bot
from dotenv import load_dotenv
from os import getenv

class DiscordAIBot:
    def __init__(self, discord_token: str = None):
        if discord_token is None:
            raise ValueError("Discord token is required")
        self.discord_token = discord_token

    def run(self):
        # Set up intents for the bot
        intents = Intents.default()
        intents.message_content = True
        bot = commands.Bot(command_prefix="!", intents=intents, help_command=None, case_insensitive=True)

        @bot.event
        async def on_ready():
            # Print a message when the bot is ready
            print(f"{bot.user} is ready")

            # Loop through all guilds (servers) the bot is in
            for guild in bot.guilds:
                # Loop through all text channels in the guild
                for channel in guild.text_channels:
                    # Check if the bot has permission to send messages in the channel
                    if channel.permissions_for(guild.me).send_messages:
                        await channel.send("Hello, how can I assist you today? Please type !help to see the available commands.")
                        break

        @bot.event
        async def on_message(message: Message):
            # Ignore messages from the bot itself
            if message.author == bot.user:
                return

            if message.content.startswith("!"):
                message_content = message.content.split(" ")[0][1:]
                try:
                    # Attempt to process the command received in the message
                    if message_content in bot.all_commands:
                        await bot.process_commands(message)
                    else:
                        # If the command is not found, send a message indicating it doesn't exist
                        await message.channel.send("This command does not exist, please try again!")
                except KeyError:
                    # Handle any errors that may occur while fetching data
                    await message.channel.send("An error occurred while fetching the menu.")
            else:
                await bot.process_commands(message)

        # Define the help command
        @bot.command()
        async def help(ctx):
            # Send a help message to the user
            help_message = (
                "Here are the available commands:\n"
                "- **!aihelp [question]**: Ask the AI a question.\n"
                "- **!faq [topic]**: Show a list of frequently asked questions.\n"
                "- **!topic [name]**: Get information about a specific AI course topic.\n"
                "- **!help**: Show this help message.\n"
            )
            await ctx.send(help_message)

        # Define the aihelp command
        @bot.command()
        async def aihelp(ctx, *, question: str):
            # Placeholder for AI help functionality
            await ctx.send(f"Processing your question: {question}")

        # Define the faq command
        @bot.command()
        async def faq(ctx, *, topic: str):
            # Placeholder for FAQ functionality
            await ctx.send(f"Here are FAQs about {topic}")

        # Define the topic command
        @bot.command()
        async def topic(ctx, *, name: str):
            # Placeholder for topic functionality
            await ctx.send(f"Information about {name}")

        # Define the shutdown command
        @bot.command()
        @commands.has_permissions(administrator=True)
        async def shutdown(ctx):
            # Shut down the bot
            await ctx.send("Shutting down the bot...")
            await bot.close()
            
        # Define the restart command
        @bot.command()
        @commands.has_permissions(administrator=True)
        async def restart(ctx):
            # Restart the bot by executing the script again
            await ctx.send("Restarting the bot...")
            os.execv(sys.executable, ['python'] + sys.argv)

        # Run the bot
        bot.run(self.discord_token)
                
                
if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()

    # Get the Discord token from the environment variable
    discord_token = getenv("DISCORD_AI_BOT_TOKEN") 

    if discord_token:
        bot = DiscordAIBot(discord_token)
        bot.run()
    else:
        print("No Discord token found!")