from sqlalchemy.orm import Session
from sqlalchemy import select, text
from sentence_transformers import SentenceTransformer
import numpy as np
from models import Question, Answer

class RAGSystem:
    def __init__(self, db_session: Session, embedding_model: str, threshold: float):
        ''' initialize the RAG system with the necessary models and parameters '''
        # Initialize database connection
        self.db_session = db_session

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize threshold for similarity
        self.similarity_threshold = threshold
        
    def generate_embedding(self, text: str) -> list[float]:
        ''' Generate an embedding for a given text '''
        # Encode the text to get embeddings
        embedding = self.embedding_model.encode(text)
        
        # Ensure we have a 1D array by flattening if needed
        if isinstance(embedding, np.ndarray) and embedding.ndim > 1:
            embedding = embedding.flatten()

        # Convert numpy array to a standard Python list
        return embedding.tolist()

    def find_similar_question(self, question: str) -> dict:
        ''' Find the most similar question in the database '''
        try:
            # Generate embedding for the question
            question_embedding = self.generate_embedding(question)
            
            # Convert the embedding to a string
            embedding_str = '[' + ','.join(str(x) for x in question_embedding) + ']'

            # Query the database for similar questions as raw SQL for compatibility with pgvector postgres extension
            query = f'''
                SELECT q.id, q.question, 1 - (q.question_embeddings <=> '{embedding_str}'::vector) as similarity
                FROM questions q
                ORDER BY q.question_embeddings <=> '{embedding_str}'::vector ASC
                LIMIT 1
            '''

            result = self.db_session.execute(text(query)).first()

            # if result similarity is below threshold or no result, return None
            if result and result[2] >= self.similarity_threshold:
                return {'id': result[0], 'similarity': result[2]}
            else:
                return None
            
        except Exception as e:
            print(f"Error in find_similar_question: {e}")
            return None
    
    def get_answer(self, question_id: int) -> str:
        ''' Get the answer for a given question '''
        try:
            query = (
                select(Answer)
                .where(Answer.question_id == question_id)
                .order_by(Answer.created_at.desc())
                .limit(1)
            )
            
            result = self.db_session.execute(query).one_or_none()
            
            # Return the answer text directly instead of the Answer object
            return result[0].answer if result else None
            
        except Exception as e:
            print(f"Error in get_answer: {e}")
            return None

    def save_question_answer(self, question: str, answer: str, is_verified: bool = True) -> tuple[Question, Answer]:
        ''' Save a question and answer to the database '''
        try:
            # Generate embedding for the question
            question_embedding = self.generate_embedding(question)

            # Create a new question
            new_question = Question(question=question, question_embeddings=question_embedding)
            
            # Create a new answer
            new_answer = Answer(question=new_question, answer=answer, is_verified=is_verified)

            # Add to the database
            self.db_session.add(new_question)
            self.db_session.add(new_answer)
            self.db_session.commit()

            return new_question, new_answer
            
        except Exception as e:
            self.db_session.rollback()
            print(f"Error in save_question_answer: {e}")
            return None, None