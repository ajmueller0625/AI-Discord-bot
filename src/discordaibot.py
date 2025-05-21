import os
import sys
import textwrap
import traceback
from discord import Intents, Message, Embed, Color, errors
from discord.ext import commands
from sqlalchemy.orm import Session
from ragsys import RAGSystem
from aiagent import AIAgent
from logger import get_logger

class DiscordAIBot:
    def __init__(self, discord_token: str, huggingface_token: str, embedding_model: str, llm_model: str, threshold: float, db_session: Session):
        ''' Initialize the Discord AI Bot '''
        # Initialize logger
        self.logger = get_logger()
        self.logger.info("Initializing Discord AI Bot")
        
        self.discord_token = discord_token

        try:
            # Initialize RAG system
            self.logger.info("Initializing RAG system")
            self.rag_system = RAGSystem(db_session, embedding_model, threshold)
            self.logger.info("RAG system initialized successfully")

            # Initialize AI agent
            self.logger.info("Initializing AI agent")
            self.ai_agent = AIAgent(huggingface_token, llm_model)
            self.logger.info("AI agent initialized successfully")
            
            # Bot color for embeds
            self.bot_color = Color.blue()
            
        except Exception as e:
            self.logger.critical(f"Critical error during Discord AI Bot initialization: {str(e)}")
            self.logger.critical(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize Discord bot: {str(e)}") from e

    async def _handle_off_topic(self, ctx):
        """Handle off-topic questions"""
        self.logger.info("Question filtered as not AI-related")
        off_topic_embed = Embed(
            title="Off-Topic Question",
            description="Please ask a question that is related to AI topics only.",
            color=Color.orange()
        )
        await ctx.send(embed=off_topic_embed)

    def run(self):
        """Run the Discord bot"""
        self.logger.info("Setting up Discord bot")
        
        try:
            # Set up intents for the bot
            intents = Intents.default()
            intents.message_content = True
            bot = commands.Bot(command_prefix="!", intents=intents, help_command=None, case_insensitive=True)

            @bot.event
            async def on_ready():
                # Print a message when the bot is ready
                self.logger.info(f"Bot {bot.user} is ready and connected to Discord")

                try:
                    # Loop through all guilds (servers) the bot is in
                    for guild in bot.guilds:
                        self.logger.info(f"Bot is active in guild: {guild.name} (ID: {guild.id})")
                        
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
                                
                                try:
                                    await channel.send(embed=welcome_embed)
                                    self.logger.info(f"Sent welcome message to channel: {channel.name} (ID: {channel.id})")
                                    break
                                except errors.Forbidden:
                                    self.logger.warning(f"Permission error sending welcome message to channel: {channel.name}")
                                except Exception as e:
                                    self.logger.error(f"Error sending welcome message: {str(e)}")
                except Exception as e:
                    self.logger.error(f"Error during on_ready event: {str(e)}")
                    self.logger.error(traceback.format_exc())

            @bot.event
            async def on_message(message: Message):
                # Ignore messages from the bot itself
                if message.author == bot.user:
                    return

                try:
                    if message.content.startswith("!"):
                        self.logger.info(f"Command received from {message.author}: {message.content}")
                        message_content = message.content.split(" ")[0][1:]
                        try:
                            # Attempt to process the command received in the message
                            if message_content in bot.all_commands:
                                await bot.process_commands(message)
                            else:
                                # If the command is not found, send an embed indicating it doesn't exist
                                self.logger.warning(f"Unknown command received: {message_content}")
                                error_embed = Embed(
                                    title="Command Not Found",
                                    description="This command does not exist, please try again!",
                                    color=Color.red()
                                )
                                await message.channel.send(embed=error_embed)
                        except KeyError as e:
                            self.logger.error(f"KeyError processing command: {str(e)}")
                            # Handle any errors that may occur while fetching data
                            error_embed = Embed(
                                title="Error",
                                description="An error occurred while processing your request.",
                                color=Color.red()
                            )
                            await message.channel.send(embed=error_embed)
                        except Exception as e:
                            self.logger.error(f"Unexpected error processing command: {str(e)}")
                            self.logger.error(traceback.format_exc())
                            error_embed = Embed(
                                title="Error",
                                description="An unexpected error occurred. Please try again later.",
                                color=Color.red()
                            )
                            await message.channel.send(embed=error_embed)
                    else:
                        await bot.process_commands(message)
                except Exception as e:
                    self.logger.error(f"Error in on_message event: {str(e)}")
                    self.logger.error(traceback.format_exc())

            # Error handler for commands
            @bot.event
            async def on_command_error(ctx, error):
                self.logger.error(f"Command error: {str(error)}")
                
                if isinstance(error, commands.MissingRequiredArgument):
                    error_embed = Embed(
                        title="Missing Argument",
                        description=f"The command is missing a required argument: {error.param}",
                        color=Color.red()
                    )
                    await ctx.send(embed=error_embed)
                elif isinstance(error, commands.CommandNotFound):
                    error_embed = Embed(
                        title="Command Not Found",
                        description="This command does not exist. Type !help for available commands.",
                        color=Color.red()
                    )
                    await ctx.send(embed=error_embed)
                elif isinstance(error, commands.MissingPermissions):
                    error_embed = Embed(
                        title="Permission Denied",
                        description="You don't have the required permissions to use this command.",
                        color=Color.red()
                    )
                    await ctx.send(embed=error_embed)
                else:
                    self.logger.error(traceback.format_exc())
                    error_embed = Embed(
                        title="Error",
                        description="An unexpected error occurred. Please try again later.",
                        color=Color.red()
                    )
                    await ctx.send(embed=error_embed)

            # Define the help command
            @bot.command()
            async def help(ctx):
                self.logger.info(f"Help command requested by {ctx.author}")
                # Create a help embed
                try:
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
                        name="!restart",
                        value="Restart the bot. (admin only)",
                        inline=False
                    )

                    help_embed.add_field(
                        name="!shutdown",
                        value="Shutdown the bot. (admin only)",
                        inline=False
                    )
                    
                    help_embed.add_field(
                        name="!help",
                        value="Show this help message with available commands.",
                        inline=False
                    )
                    
                    help_embed.set_footer(text="AI Tutor Bot | Helping you learn about AI")
                    
                    await ctx.send(embed=help_embed)
                    self.logger.debug("Help command executed successfully")
                except Exception as e:
                    self.logger.error(f"Error executing help command: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    await ctx.send("An error occurred while processing your request. Please try again later.")

            async def send_embed_message(ctx, title, text, is_similar=False, similarity=None):
                ''' Send a message as an embed with proper formatting '''
                self.logger.debug(f"Sending embed message with title: {title}, text length: {len(text) if text else 0}")
                
                try:
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
                        self.logger.debug("Embed message sent successfully")
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
                        
                    self.logger.debug(f"Split message sent successfully in {len(chunks)} parts")
                
                except errors.Forbidden as e:
                    self.logger.error(f"Permission error sending embed: {str(e)}")
                    try:
                        await ctx.send("I don't have permission to send embeds in this channel.")
                    except:
                        pass
                except Exception as e:
                    self.logger.error(f"Error sending embed message: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    try:
                        await ctx.send("An error occurred while sending the message. Please try again later.")
                    except:
                        pass

            # Define the aihelp command
            @bot.command()
            async def aihelp(ctx, *, question: str = None):
                self.logger.info(f"AI help command requested by {ctx.author}")
                
                # Check if the question is provided
                if not question:
                    error_embed = Embed(
                        title="Missing Question",
                        description="Please provide a question after the !aihelp command. For example: !aihelp What is machine learning?",
                        color=Color.red()
                    )
                    await ctx.send(embed=error_embed)
                    self.logger.warning(f"AI help command called without a question by {ctx.author}")
                    return
                
                try:
                    # Send a processing message
                    processing_embed = Embed(
                        title="Processing Your Question",
                        description="Please wait while I find the best answer for you...",
                        color=self.bot_color
                    )
                    await ctx.send(embed=processing_embed)
                    
                    # Get the answer from the RAG system
                    self.logger.debug(f"Finding similar question for: {question[:50]}...")
                    similar_question = self.rag_system.find_similar_question(question)

                    if similar_question:
                        # Get answer for similar question
                        self.logger.info(f"Similar question found with ID: {similar_question['id']}")
                        answer = self.rag_system.get_answer(similar_question['id'])
                        
                        if answer:
                            await send_embed_message(
                                ctx, 
                                "AI Tutor Answer", 
                                answer, 
                                is_similar=True, 
                                similarity=similar_question['similarity']
                            )
                        else:
                            self.logger.warning(f"Similar question found but no answer retrieved for ID: {similar_question['id']}")
                            error_embed = Embed(
                                title="Error Retrieving Answer",
                                description="I found a similar question in my database, but couldn't retrieve the answer. Let me generate a new response.",
                                color=Color.orange()
                            )
                            await ctx.send(embed=error_embed)
                            
                            # Generate a new answer
                            answer = self.ai_agent.generate_response(question)
                            if answer:
                                self.rag_system.save_question_answer(question, answer)
                                await send_embed_message(ctx, "AI Tutor Answer", answer)
                            else:
                                await self._handle_off_topic(ctx)
                    else:
                        # Generate a new answer
                        self.logger.info("No similar question found, generating new answer")
                        answer = self.ai_agent.generate_response(question)
                        
                        if answer:
                            # Save the question and answer
                            self.logger.debug("Saving new question and answer to database")
                            self.rag_system.save_question_answer(question, answer)
                            await send_embed_message(ctx, "AI Tutor Answer", answer)
                        else:
                            await self._handle_off_topic(ctx)
                
                except Exception as e:
                    self.logger.error(f"Error processing AI help command: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    error_embed = Embed(
                        title="Error",
                        description="An error occurred while processing your question. Please try again later.",
                        color=Color.red()
                    )
                    await ctx.send(embed=error_embed)

            # Define the shutdown command
            @bot.command()
            @commands.has_permissions(administrator=True)
            async def shutdown(ctx):
                self.logger.info(f"Shutdown command requested by {ctx.author}")
                try:
                    # Create a shutdown embed
                    shutdown_embed = Embed(
                        title="Bot Shutdown",
                        description="Shutting down the bot...",
                        color=Color.dark_red()
                    )
                    await ctx.send(embed=shutdown_embed)
                    self.logger.info("Bot shutting down via command")
                    await bot.close()
                except Exception as e:
                    self.logger.error(f"Error during shutdown: {str(e)}")
                    self.logger.error(traceback.format_exc())
            
            # Define the restart command
            @bot.command()
            @commands.has_permissions(administrator=True)
            async def restart(ctx):
                self.logger.info(f"Restart command requested by {ctx.author}")
                try:
                    # Create a restart embed
                    restart_embed = Embed(
                        title="Bot Restart",
                        description="Restarting the bot...",
                        color=Color.dark_green()
                    )
                    await ctx.send(embed=restart_embed)
                    self.logger.info("Bot restarting via command")
                    os.execv(sys.executable, ['python'] + sys.argv)
                except Exception as e:
                    self.logger.error(f"Error during restart: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    error_embed = Embed(
                        title="Restart Failed",
                        description="An error occurred while trying to restart the bot.",
                        color=Color.red()
                    )
                    await ctx.send(embed=error_embed)

            # Run the bot
            try:
                self.logger.info("Starting Discord bot")
                bot.run(self.discord_token)
            except errors.LoginFailure:
                self.logger.critical("Invalid Discord token. Please check your token and try again.")
                raise ValueError("Invalid Discord token")
            except Exception as e:
                self.logger.critical(f"Unhandled exception in Discord bot setup: {str(e)}")
                self.logger.critical(traceback.format_exc())
                raise RuntimeError(f"Failed to run Discord bot: {str(e)}") from e
        except Exception as e:
            self.logger.critical(f"Unhandled exception in Discord bot setup: {str(e)}")
            self.logger.critical(traceback.format_exc())
            raise RuntimeError(f"Failed to set up Discord bot: {str(e)}") from e