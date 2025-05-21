import torch
import traceback
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
from huggingface_hub import login
from logger import get_logger

class AIAgent:
    def __init__(self, huggingface_token: str, llm_model: str):
        ''' initialize the AIAgent with the necessary models and parameters '''
        
        # Initialize the logger
        self.logger = get_logger()
        

        # Initialize the persona and filtering instructions
        self.persona = '''
        You are an AI tutor specializing in artificial intelligence courses. 
        Provide clear, concise, and accurate answers to questions about AI topics.
        Base your answers on established AI knowledge.
        Be educational and informative.
        '''

        self.filtering_instructions = '''
        IMPORTANT INSTRUCTION: You are a specialized AI tutor that ONLY answers questions about artificial intelligence, 
        machine learning, deep learning, neural networks, and closely related technical AI topics.
        
        Follow these rules exactly:
        1. First, evaluate if the question is specifically about AI technology, methods, concepts, or applications.
        2. If the question is not clearly about AI, respond ONLY with the exact text: "NOT_AI_TOPIC"
        3. If you're unsure if a topic is AI-related, err on the side of caution and respond with "NOT_AI_TOPIC"
        
        AI-related topics include but are not limited to:
        - Machine learning algorithms and techniques
        - Neural network architectures and training
        - Natural language processing and computer vision
        - AI programming frameworks and libraries
        - AI ethics and responsible AI development
        - AI research and recent advances
        - Technical implementation of AI systems
        
        Examples of questions that are NOT about AI and should receive "NOT_AI_TOPIC":
        - "What's the weather like today?"
        - "Can you help with my math homework?"
        - "Write me a poem about love"
        - "Who won the World Cup?"
        - "Give me a recipe for chocolate cake"
        - "What's your opinion on politics?"
        
        This filtering is critical. ONLY provide substantive answers to AI-specific questions.
        '''

        try:
            # Login to Hugging Face
            self.logger.info('Logging in to Hugging Face')
            login(huggingface_token)
            self.logger.info('Logged in to Hugging Face successfully')

            # Initialize the model
            self.model_name = llm_model
            self.logger.info(f'Loading tokenizer for {self.model_name}')
            
            # Initialize the tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)        
                self.logger.info('Tokenizer initialized successfully')
            except Exception as e:
                self.logger.error(f'Error loading tokenizer: {str(e)}')
                self.logger.error(traceback.format_exc())
                raise
            
            # 4-bit quantization
            self.logger.info('Configuring 4-bit quantization')
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            # Update model loading
            self.logger.info(f'Loading LLM pipeline with model: {self.model_name}')
            try:
                self.llm = pipeline(
                    'text-generation',
                    model=self.model_name,
                tokenizer=self.tokenizer,
                model_kwargs={'quantization_config': self.quantization_config},
                    torch_dtype=torch.bfloat16,
                    device_map='auto'
                )
                self.logger.info('LLM pipeline loaded successfully')
            except Exception as e:
                self.logger.error(f'Error loading LLM pipeline: {str(e)}')
                self.logger.error(traceback.format_exc())
                raise
    
        except Exception as e:
            self.logger.critical(f'Critical error occurred while initializing AIAgent: {str(e)}')
            self.logger.critical(traceback.format_exc())
            raise
        
    def generate_response(self, question: str) -> str:
        ''' Generate a response to a question '''
        # Generate a response to a question

        try:
            # Two-step prompt approach
            # 1. Filter the question
            filter_prompt = f'''
            <s>[INST] <<SYS>>{self.filtering_instructions}<</SYS>>
            User Question: {question}

            Is this question about AI? Remember, you must ONLY respond with "NOT_AI_TOPIC" if the question is not about AI.
            [/INST]
            '''
            self.logger.info('Running filtering prompt')
            try:
                # Check if the question is about AI related topics
                filter_response = self.llm(filter_prompt, max_newtokens=64, temperature=0.1)[0]['generated_text']
                filter_response = filter_response.split("[/INST]")[1].strip()
                self.logger.info(f'Filtering response: {filter_response[:50]}...')
            except Exception as e:
                self.logger.error(f'Error filtering question: {str(e)}')
                self.logger.error(traceback.format_exc())
                raise
            
            if 'NOT_AI_TOPIC' in filter_response:
                self.logger.info('Question is not about AI. Returning None.')
                return None
            
            # 2. Generate a response
            prompt = f'''
            <s>[INST] <<SYS>>{self.persona}<</SYS>>
            Question about AI: {question}
            [/INST]
            '''
            self.logger.info('Running response generation')
            try:
                # Generate a response to the question using the model
                response = self.llm(prompt, max_new_tokens=512)[0]['generated_text']
                # Extract the answer from the response
                answer = response.split("[/INST]")[1].strip()
                self.logger.info(f'Generated response length: {len(answer)} characters')
                return answer
            except Exception as e:
                self.logger.error(f'Error generating response: {str(e)}')
                self.logger.error(traceback.format_exc())
                raise
        except Exception as e:
            self.logger.error(f'Unexpected error occurred while generating response: {str(e)}')
            self.logger.error(traceback.format_exc())
            raise
