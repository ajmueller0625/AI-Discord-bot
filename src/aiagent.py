import os
import torch
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import select
from pgvector.sqlalchemy import cosine_distance
from transformers import AutoTokenizer, AutoModel, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from models import Question, Answer

load_dotenv()

class AIAgent:
    def __init__(self, db_session: Session):
        ''' initialize the AIAgent with the necessary models and parameters '''
        # Initialize database connection
        self.db_session = db_session

        # Login to Hugging Face
        login(os.getenv('HUGGING_FACE_HUB_TOKEN'))

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(os.getenv('EMBEDDING_MODEL'))

        # Initialize the model
        self.model_name = os.getenv('LLM_MODEL')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)        
        
        # 4-bit quantization
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        # Update model loading
        self.llm = pipeline(
            'text-generation',
            model=self.model_name,
            tokenizer=self.tokenizer,
            model_kwargs={'quantization_config': self.quantization_config},
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

        # initialize threshold for similarity
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', 0.85))

    def generate_embedding(self, text: str) -> list[float]:
        ''' Generate an embedding for a given text '''
        return self.embedding_model.encode(text).tolist()
    
    
