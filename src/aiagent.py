import torch
import traceback
import re
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
from huggingface_hub import login
from logger import get_logger

class AIAgent:
    def __init__(self, huggingface_token: str, llm_model: str):
        ''' initialize the AIAgent with the necessary models and parameters '''
        
        # Initialize the logger
        self.logger = get_logger()
        
        # Define AI acronyms and their meanings
        self.ai_acronyms = {
            "CNN": "Convolutional Neural Network",
            "GAN": "Generative Adversarial Network",
            "RNN": "Recurrent Neural Network",
            "LSTM": "Long Short-Term Memory",
            "NLP": "Natural Language Processing",
            "CV": "Computer Vision",
            "RL": "Reinforcement Learning",
            "ML": "Machine Learning",
            "DL": "Deep Learning",
            "AI": "Artificial Intelligence",
            "SVM": "Support Vector Machine",
            "KNN": "K-Nearest Neighbors",
            "PCA": "Principal Component Analysis",
            "BERT": "Bidirectional Encoder Representations from Transformers",
            "GPT": "Generative Pre-trained Transformer",
            "VAE": "Variational Autoencoder",
            "MLP": "Multi-Layer Perceptron",
            "NN": "Neural Network",
            "SGD": "Stochastic Gradient Descent",
            "YOLO": "You Only Look Once",
            "OCR": "Optical Character Recognition",
            "ANN": "Artificial Neural Network"
        }

        # Initialize the persona and filtering instructions
        self.persona = f'''
        You are an AI tutor specializing in artificial intelligence courses. 
        
        Follow these strict instructions:
        1. Provide clear, concise, and accurate answers to questions about AI topics.
        2. Base your answers on established AI knowledge.
        3. Be educational and informative.
        4. ALWAYS interpret acronyms in the context of AI, including but not limited to:
           {'; '.join([f"{acronym} = {meaning}" for acronym, meaning in self.ai_acronyms.items()])}
        5. When a student asks about any of these acronyms, ALWAYS interpret them as the AI concept listed above, NEVER as any other potential meaning.
        '''

        self.filtering_instructions = f'''
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
        - AI programming frameworks and libraries (TensorFlow, PyTorch, Keras, Hugging Face, etc.)
        - AI ethics and responsible AI development
        - AI research and recent advances
        - Technical implementation of AI systems
        
        CRITICAL INSTRUCTIONS FOR ACRONYMS - READ CAREFULLY:
        Questions about the following acronyms ARE ALWAYS ABOUT AI TOPICS and should ALWAYS be answered, never filtered out:
        {'; '.join([f"{acronym} ({meaning})" for acronym, meaning in self.ai_acronyms.items()])}
        
        IMPORTANT: NEVER interpret these acronyms as anything other than their AI meanings.
        For example:
        - If a question asks about "CNN", it ALWAYS means Convolutional Neural Network, NEVER a news network
        - If a question asks about "GAN", it ALWAYS means Generative Adversarial Network, NEVER anything else
        
        Always answer questions about these acronyms. They are valid AI topics.
        
        Examples of questions that are NOT about AI and should receive "NOT_AI_TOPIC":
        - "What's the weather like today?"
        - "Can you help with my math homework?"
        - "Write me a poem about love"
        - "Who won the World Cup?"
        - "Give me a recipe for chocolate cake"
        - "What's your opinion on politics?"
        
        Examples of AI questions that SHOULD be answered:
        - "What is a CNN?" - THIS IS ABOUT CONVOLUTIONAL NEURAL NETWORKS - YOU SHOULD ANSWER
        - "How do GANs work?" - THIS IS ABOUT GENERATIVE ADVERSARIAL NETWORKS - YOU SHOULD ANSWER
        - "Explain the difference between CNN and RNN" - THIS IS COMPARING AI ARCHITECTURES - YOU SHOULD ANSWER
        - "What's the difference between CNN and GAN?" - THIS IS COMPARING AI ARCHITECTURES - YOU SHOULD ANSWER
        - "Tell me about CNN architecture" - THIS IS ABOUT CONVOLUTIONAL NEURAL NETWORKS - YOU SHOULD ANSWER
        - "What does CNN stand for?" - THIS IS ASKING ABOUT CONVOLUTIONAL NEURAL NETWORKS - YOU SHOULD ANSWER
        
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
    
    def contains_ai_acronym(self, text):
        '''Check if the text contains any AI acronyms using regex to match whole words only'''
        words = re.findall(r'\b[A-Z]{2,}\b', text.upper())  # Find uppercase acronyms (2+ letters)
        return any(acronym in words for acronym in self.ai_acronyms)
    
    def preprocess_question(self, question):
        '''
        Preprocess the question to explicitly handle AI acronyms
        This ensures questions about CNN, GAN, etc. are always interpreted correctly
        '''
        # Check if question contains any AI acronyms
        has_acronym = self.contains_ai_acronym(question)
        
        if has_acronym:
            self.logger.info(f'AI acronym detected in question: {question[:50]}...')
            
            # Find all acronyms in the question
            words = re.findall(r'\b[A-Z]{2,}\b', question.upper())
            detected_acronyms = [word for word in words if word in self.ai_acronyms]
            
            # Log detected acronyms
            if detected_acronyms:
                self.logger.info(f'Detected AI acronyms: {", ".join(detected_acronyms)}')
                
                # Expand the first occurrence of each acronym in the question
                preprocessed = question
                for acronym in detected_acronyms:
                    # Only expand the first occurrence of each acronym
                    pattern = re.compile(r'\b' + re.escape(acronym) + r'\b', re.IGNORECASE)
                    replacement = f"{acronym} ({self.ai_acronyms[acronym]})"
                    preprocessed = pattern.sub(replacement, preprocessed, count=1)
                
                self.logger.info(f'Preprocessed question: {preprocessed[:100]}...')
                return preprocessed, True
            
        return question, has_acronym
        
    def generate_response(self, question: str) -> str:
        ''' Generate a response to a question '''
        # Generate a response to a question

        try:
            # Preprocess the question to handle AI acronyms
            processed_question, has_ai_acronym = self.preprocess_question(question)
            
            # If an AI acronym was detected, skip filtering
            if has_ai_acronym:
                self.logger.info("AI acronym detected, skipping filtering")
                prompt = f'''
                <s>[INST] <<SYS>>{self.persona}<</SYS>>
                Question about AI terminology: {processed_question}
                
                Remember to explain the AI-specific meaning of any acronyms in the question.
                [/INST]
                '''
                
                try:
                    self.logger.info("Generating response for AI acronym question")
                    response = self.llm(prompt, max_new_tokens=512, temperature=0.2)[0]['generated_text']
                    answer = response.split("[/INST]")[1].strip()
                    self.logger.info(f"Generated response length: {len(answer)} characters")
                    return answer
                except Exception as e:
                    self.logger.error(f"Error generating response for AI acronym: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    raise
                    
            # For other questions, use the two-step approach
            # 1. Filter the question
            filter_prompt = f'''
            <s>[INST] <<SYS>>{self.filtering_instructions}<</SYS>>
            User Question: {processed_question}

            Is this question about AI? Remember, you must ONLY respond with "NOT_AI_TOPIC" if the question is not about AI.
            [/INST]
            '''
            self.logger.info('Running filtering prompt')
            try:
                # Check if the question is about AI related topics
                filter_response = self.llm(filter_prompt, max_new_tokens=64, temperature=0.1)[0]['generated_text']
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
            Question about AI: {processed_question}
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