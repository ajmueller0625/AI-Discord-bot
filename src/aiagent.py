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

        # Define AI acronyms (shortened list for faster processing)
        self.ai_acronyms = {
            'CNN', 'GAN', 'RNN', 'LSTM', 'NLP', 'CV', 'RL', 'ML', 'DL', 'AI',
            'SVM', 'KNN', 'PCA', 'BERT', 'GPT', 'VAE', 'MLP', 'NN', 'SGD',
            'YOLO', 'OCR', 'ANN'
        }

        # Simplified, shorter persona
        self.persona = '''
            You are an AI tutor. Answer AI questions clearly and concisely. 
            Always interpret CNN as Convolutional Neural Network, GAN as Generative Adversarial Network, 
            RNN as Recurrent Neural Network, LSTM as Long Short-Term Memory, etc.
            '''

        # Filtering instructions
        self.filtering_instructions = '''
            Answer ONLY AI/ML questions. If not about AI, respond: "NOT_AI_TOPIC"
            AI acronyms like CNN, GAN, RNN, LSTM, NLP, CV, RL, ML, DL, AI, SVM, KNN, PCA, BERT, GPT, VAE, MLP are ALWAYS AI topics.
            Always answer questions about these acronyms.
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
            self.logger.info(
                f'Loading LLM pipeline with model: {self.model_name}')
            try:
                self.llm = pipeline(
                    'text-generation',
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    model_kwargs={
                        'quantization_config': self.quantization_config},
                    torch_dtype=torch.bfloat16,
                    device_map='auto'
                )
                self.logger.info('LLM pipeline loaded successfully')
            except Exception as e:
                self.logger.error(f'Error loading LLM pipeline: {str(e)}')
                self.logger.error(traceback.format_exc())
                raise

        except Exception as e:
            self.logger.critical(
                f'Critical error occurred while initializing AIAgent: {str(e)}')
            self.logger.critical(traceback.format_exc())
            raise

    def contains_ai_acronym(self, text):
        """Fast check for AI acronyms"""
        words = text.upper().split()
        return any(word.strip('.,!?') in self.ai_acronyms for word in words)

    def generate_response(self, question: str) -> str:
        ''' Generate a response to a question '''

        try:
            # Quick check for AI acronyms (no preprocessing to save time)
            has_ai_acronym = self.contains_ai_acronym(question)

            # If AI acronym detected, skip filtering and generate response directly
            if has_ai_acronym:
                self.logger.info("AI acronym detected, skipping filtering")
                prompt = f'''<s>[INST] <<SYS>>{self.persona}<</SYS>>
                    {question} [/INST]'''

                try:
                    # Large tokens for better response
                    response = self.llm(prompt, max_new_tokens=1024, temperature=0.7, do_sample=True)[
                        0]['generated_text']
                    answer = response.split("[/INST]")[1].strip()
                    self.logger.info(
                        f"Fast response generated, length: {len(answer)} characters")
                    return answer
                except Exception as e:
                    self.logger.error(
                        f"Error generating fast response: {str(e)}")
                    raise

            # For other questions, use simplified two-step approach
            # 1. Quick filter check
            filter_prompt = f'''<s>[INST] <<SYS>>{self.filtering_instructions}<</SYS>>
                {question}
                Is this AI-related? [/INST]'''

            self.logger.info('Running quick filter')
            try:
                # Very short filtering response
                filter_response = self.llm(filter_prompt, max_new_tokens=32, temperature=0.1)[
                    0]['generated_text']
                filter_response = filter_response.split("[/INST]")[1].strip()
                self.logger.info(f'Filter result: {filter_response[:30]}...')
            except Exception as e:
                self.logger.error(f'Error in filtering: {str(e)}')
                raise

            if 'NOT_AI_TOPIC' in filter_response:
                self.logger.info('Not AI topic')
                return None

            # 2. Generate response with reduced tokens
            prompt = f'''<s>[INST] <<SYS>>{self.persona}<</SYS>>
                {question} [/INST]'''

            self.logger.info('Generating main response')
            try:
                # Larger tokens for better response
                response = self.llm(prompt, max_new_tokens=1024, temperature=0.7, do_sample=True)[
                    0]['generated_text']
                answer = response.split("[/INST]")[1].strip()
                self.logger.info(
                    f'Response generated, length: {len(answer)} characters')
                return answer
            except Exception as e:
                self.logger.error(f'Error generating main response: {str(e)}')
                raise

        except Exception as e:
            self.logger.error(f'Unexpected error: {str(e)}')
            self.logger.error(traceback.format_exc())
            raise
