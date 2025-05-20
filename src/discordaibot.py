import os
import sys
import textwrap
from discord import Intents, Message, Embed, Color
from discord.ext import commands
from sqlalchemy.orm import Session
from ragsys import RAGSystem
from aiagent import AIAgent

class DiscordAIBot:
    def __init__(self, discord_token: str, huggingface_token: str, embedding_model: str, llm_model: str, threshold: float, db_session: Session):
        ''' Initialize the Discord AI Bot '''
        self.discord_token = discord_token

        # Initialize RAG system
        self.rag_system = RAGSystem(db_session, embedding_model, threshold)

        # Initialize AI agent
        self.ai_agent = AIAgent(huggingface_token, llm_model)
        
        # Bot color for embeds
        self.bot_color = Color.blue()

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
                        # Create a welcome embed
                        welcome_embed = Embed(
                            title="AI Tutor Bot Online",
                            description="Hello, how can I assist you today? Please type `!help` to see the available commands.",
                            color=self.bot_color
                        )
                        welcome_embed.set_footer(text="AI Tutor | Ready to help with AI topics")
                        
                        await channel.send(embed=welcome_embed)
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
                        # If the command is not found, send an embed indicating it doesn't exist
                        error_embed = Embed(
                            title="Command Not Found",
                            description="This command does not exist, please try again!",
                            color=Color.red()
                        )
                        await message.channel.send(embed=error_embed)
                except KeyError:
                    # Handle any errors that may occur while fetching data
                    error_embed = Embed(
                        title="Error",
                        description="An error occurred while processing your request.",
                        color=Color.red()
                    )
                    await message.channel.send(embed=error_embed)
            else:
                await bot.process_commands(message)

        # Define the help command
        @bot.command()
        async def help(ctx):
            # Create a help embed
            help_embed = Embed(
                title="AI Tutor Bot Commands",
                description="Here are the available commands for the AI Tutor Bot:",
                color=self.bot_color
            )
            
            # Add fields for each command
            help_embed.add_field(
                name="!aihelp [question]",
                value="Ask the AI a question about artificial intelligence topics.",
                inline=False
            )
            
            help_embed.add_field(
                name="!help",
                value="Show this help message with available commands.",
                inline=False
            )
            
            help_embed.set_footer(text="AI Tutor Bot | Helping you learn about AI")
            
            await ctx.send(embed=help_embed)

        async def send_embed_message(ctx, title, text, is_similar=False, similarity=None):
            ''' Send a message as an embed with proper formatting '''
            # Set the max length for embed description (Discord limit is 4096)
            max_length = 4000
            
            # Create base embed with appropriate color
            embed = Embed(
                title=title,
                color=self.bot_color
            )
            
            # If the message is short enough, send it directly in one embed
            if len(text) <= max_length:
                # First, add the similarity notice at the top if applicable
                if is_similar and similarity is not None:
                    # Create a separate embed for the similarity notice that will appear first
                    similarity_embed = Embed(
                        title="Similar Question Found",
                        description=f"We found a similar question in our database with {similarity*100:.2f}% similarity.",
                        color=Color.green()
                    )
                    await ctx.send(embed=similarity_embed)
                
                # Then send the main content
                embed.description = text
                await ctx.send(embed=embed)
                return
                
            # For longer messages, split into multiple embeds
            processing_embed = Embed(
                title="Processing Long Answer",
                description="The answer is quite detailed, so I'll split it into multiple parts:",
                color=self.bot_color
            )
            await ctx.send(embed=processing_embed)
            
            # First, add the similarity notice at the top if applicable
            if is_similar and similarity is not None:
                # Create a separate embed for the similarity notice that will appear first
                similarity_embed = Embed(
                    title="Similar Question Found",
                    description=f"We found a similar question in our database with {similarity*100:.2f}% similarity.",
                    color=Color.green()
                )
                await ctx.send(embed=similarity_embed)
            
            # Split the text into chunks of max_length characters
            chunks = textwrap.wrap(text, max_length, break_long_words=False, replace_whitespace=False)
            
            # Send each chunk as a separate embed
            for i, chunk in enumerate(chunks):
                part_embed = Embed(
                    title=f"{title} (Part {i+1}/{len(chunks)})",
                    description=chunk,
                    color=self.bot_color
                )
                part_embed.set_footer(text=f"Part {i+1} of {len(chunks)}")
                await ctx.send(embed=part_embed)

        # Define the aihelp command
        @bot.command()
        async def aihelp(ctx, *, question: str):
            # Send a processing message
            processing_embed = Embed(
                title="Processing Your Question",
                description="Please wait while I find the best answer for you...",
                color=self.bot_color
            )
            await ctx.send(embed=processing_embed)
            
            # Get the answer from the RAG system
            similar_question = self.rag_system.find_similar_question(question)

            if similar_question:
                # Send the answer as an embed
                answer = self.rag_system.get_answer(similar_question['id'])
                await send_embed_message(
                    ctx, 
                    "AI Tutor Answer", 
                    answer, 
                    is_similar=True, 
                    similarity=similar_question['similarity']
                )
            else:
                answer = self.ai_agent.generate_response(question)
                if answer:
                    self.rag_system.save_question_answer(question, answer)
                    await send_embed_message(ctx, "AI Tutor Answer", answer)
                else:
                    off_topic_embed = Embed(
                        title="Off-Topic Question",
                        description="Please ask a question that is related to AI topics only.",
                        color=Color.orange()
                    )
                    await ctx.send(embed=off_topic_embed)

        # Define the shutdown command
        @bot.command()
        @commands.has_permissions(administrator=True)
        async def shutdown(ctx):
            # Create a shutdown embed
            shutdown_embed = Embed(
                title="Bot Shutdown",
                description="Shutting down the bot...",
                color=Color.dark_red()
            )
            await ctx.send(embed=shutdown_embed)
            await bot.close()
            
        # Define the restart command
        @bot.command()
        @commands.has_permissions(administrator=True)
        async def restart(ctx):
            # Create a restart embed
            restart_embed = Embed(
                title="Bot Restart",
                description="Restarting the bot...",
                color=Color.dark_green()
            )
            await ctx.send(embed=restart_embed)
            os.execv(sys.executable, ['python'] + sys.argv)

        # Run the bot
        bot.run(self.discord_token)