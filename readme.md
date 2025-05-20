# AI Tutor Discord Bot

A specialized Discord bot designed to function as a virtual teaching assistant for students taking artificial intelligence courses. The bot provides expert AI-powered tutoring on artificial intelligence topics, answering questions about AI concepts, algorithms, techniques, and theories that students encounter in their coursework. It uses a Retrieval-Augmented Generation (RAG) system to deliver accurate, educational responses tailored to academic learning.

## What This Bot Does

### Educational Support
- **Virtual AI Teaching Assistant**: Helps students understand complex AI concepts, just like a course TA
- **Course-Relevant Answers**: Provides explanations on topics typically covered in AI, machine learning, and deep learning courses
- **Academic Focus**: Designed to support the educational journey of AI students at various levels
- **Concept Clarification**: Explains difficult AI concepts in clear, approachable language

### Core Functionality
- **AI-Focused Q&A**: Answers questions specifically about artificial intelligence topics
- **Topic Filtering**: Automatically detects and politely declines to answer off-topic questions
- **Knowledge Retention**: Saves all Q&A pairs to improve future responses
- **Similar Question Detection**: Identifies when a new question is similar to one previously answered
- **Multiple Part Responses**: Handles long explanations by breaking them into digestible chunks

### Commands
- `!aihelp [question]`: Ask any AI-related question about your coursework
- `!help`: Display available commands
- `!shutdown`: Gracefully shut down the bot (admin only)
- `!restart`: Restart the bot (admin only)

## Technical Architecture

### Language Model
- **Model**: Meta's Llama 2 (7B parameters chat model)
- **Quantization**: 4-bit quantization using bitsandbytes for efficient resource usage
- **Interface**: Accessed through Hugging Face's Transformers library

### Embedding & Vector Search
- **Embedding Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: PostgreSQL with pgvector extension (0.8.0)
- **Note**: Uses raw SQL for pgvector operations due to compatibility issues between pgvector 0.8.0 extension and pgvector 0.4.1 Python library

### RAG System
1. User asks a question via Discord
2. Question is transformed into vector embeddings
3. System searches for similar questions in the PostgreSQL database
4. If a similar question is found (above similarity threshold), returns the existing answer
5. If no similar question exists, Llama 2 generates a new answer
6. New question-answer pairs are saved for future retrieval

### Discord Integration
- Built with discord.py library
- Provides rich embed responses with proper formatting
- Handles various edge cases and error scenarios