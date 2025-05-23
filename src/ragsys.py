import numpy as np
import traceback
from sqlalchemy.orm import Session
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sentence_transformers import SentenceTransformer
from models import Question, Answer
from logger import get_logger

class RAGSystem:
    def __init__(self, db_session: Session, embedding_model: str, threshold: float):
        ''' initialize the RAG system with the necessary models and parameters '''

        # Initialize the logger
        self.logger = get_logger()
        self.logger.info('Initializing RAG system')

        # Initialize database connection
        self.db_session = db_session

        # Initialize embedding model
        try:
            self.logger.info(f'Loading embedding model: {embedding_model}')
            self.embedding_model = SentenceTransformer(embedding_model)
            self.logger.info('Embedding model loaded successfully')
        except Exception as e:
            self.logger.error(f'Error initializing embedding model: {str(e)}')
            self.logger.error(traceback.format_exc())
            raise

        # Initialize threshold for similarity
        self.similarity_threshold = threshold
        self.logger.info(f'Threshold similarity is set to: {self.similarity_threshold}')

        
    def generate_embedding(self, text: str) -> list[float]:
        ''' Generate an embedding for a given text '''
        # Encode the text to get embeddings
        self.logger.info(f'Generating embedding for text: {text[:50]}...')

        try:
            embedding = self.embedding_model.encode(text)

            # Ensure we have a 1D array by flattening if needed
            if isinstance(embedding, np.ndarray) and embedding.ndim > 1:
                embedding = embedding.flatten()

            # Convert numpy array to a standard Python list
            result = embedding.tolist()
            self.logger.info(f'Embedding generated successfully with dimension: {len(result)}')
            return result
        
        except Exception as e:
            self.logger.error(f'Error generating embedding: {str(e)}')
            self.logger.error(traceback.format_exc())
            raise

        

    def find_similar_question(self, question: str) -> dict:
        ''' Find the most similar question in the database '''
        self.logger.info(f'Finding similar question to: {question[:50]}...')

        try:
            # Generate embedding for the question
            question_embedding = self.generate_embedding(question)
            
            # Convert the embedding to a string
            embedding_str = '[' + ','.join(str(x) for x in question_embedding) + ']'

            # Query the database for similar questions as raw SQL for compatibility with pgvector postgres extension
            self.logger.info('Executing similarity search query')
            query = f'''
                SELECT q.id, q.question, 1 - (q.question_embeddings <=> '{embedding_str}'::vector) as similarity
                FROM questions q
                ORDER BY q.question_embeddings <=> '{embedding_str}'::vector ASC
                LIMIT 1
            '''
            try:
                result = self.db_session.execute(text(query)).first()
            except SQLAlchemyError as e:
                self.logger.error(f'Database error during similarity search: {str(e)}')
                self.logger.error(traceback.format_exc())
                raise

            # if result similarity is below threshold or no result, return None
            if result and result[2] >= self.similarity_threshold:
                self.logger.info(f'Found similar question with id: {result[0]} and similarity: {result[2]}')
                return {'id': result[0], 'similarity': result[2]}
            else:
                self.logger.info('No similar question found or similarity below threshold')
                return None
            
        except Exception as e:
            self.logger.error(f"Error in find_similar_question: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def get_answer(self, question_id: int) -> str:
        ''' Get the answer for a given question '''
        self.logger.info(f'Getting answer for question with id: {question_id}')

        try:
            query = (
                select(Answer)
                .where(Answer.question_id == question_id)
                .order_by(Answer.created_at.desc())
                .limit(1)
            )
            
            try:
                result = self.db_session.execute(query).one_or_none()
            except SQLAlchemyError as e:
                self.logger.error(f'Database error during answer retrieval: {str(e)}')
                self.logger.error(traceback.format_exc())
                raise
            
            # Return the answer text directly instead of the Answer object
            if result:
                self.logger.info(f'Answer retrieved successfully, length: {len(result[0].answer)}')
                return result[0].answer
            else:
                self.logger.info('No answer found for question id: {question_id}')
                return None
            
        except Exception as e:
            self.logger.error(f"Error in get_answer: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def save_question_answer(self, question: str, answer: str, is_verified: bool = True) -> tuple[Question, Answer]:
        ''' Save a question and answer to the database '''
        self.logger.info(f'Saving new question and answer: {question[:50]}...')

        try:
            # Generate embedding for the question
            question_embedding = self.generate_embedding(question)

            # Create a new question
            new_question = Question(question=question, question_embeddings=question_embedding)
            
            # Create a new answer
            new_answer = Answer(question=new_question, answer=answer, is_verified=is_verified)

            try:
                # Add to the database
                self.db_session.add(new_question)
                self.db_session.add(new_answer)
                self.db_session.commit()
                self.logger.info(f'Question and answer saved successfully with question id: {new_question.id}')
                return new_question, new_answer
            except SQLAlchemyError as e:
                self.db_session.rollback()
                self.logger.error(f'Database error during saving question and answer: {str(e)}')
                self.logger.error(traceback.format_exc())
                return None, None
            
        except Exception as e:
            if 'session.db_session' in locals():
                self.db_session.rollback()
            self.logger.error(f"Error in save_question_answer: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, None