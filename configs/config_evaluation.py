### Models
# Test Models
model_name_or_path = {
    # Baselines
    "Llama3.1-Initialization": "../Models/Llama3.1-Initialization",
    "HERALD_Translator": "../Models/HERALD_Translator",
    "Kimina-Autoformalizer": "../Models/Kimina-Autoformalizer",

    # Different Base Models finetuned on ATLAS Dataset (LoRA)
    "ATLAS_Translator_L": "../Models/ATLAS_Translator_L",
    "ATLAS_Translator_D": "../Models/ATLAS_Translator_D",
    "ATLAS_Translator_Q": "../Models/ATLAS_Translator_Q",

    # Different Base Models finetuned on ATLAS Dataset (Full)
    "ATLAS_Translator*_L": "../Models/ATLAS_Translator*_L",
    "ATLAS_Translator*_D": "../Models/ATLAS_Translator*_D",
    "ATLAS_Translator*_Q": "../Models/ATLAS_Translator*_Q",

    # Ablation Study based on Llama (LoRA)
    "w_o_Synthetic_Data": "../Models/w_o_Synthetic_Data",
    "w_o_Proof_Augmented_Data": "../Models/w_o_Proof_Augmented_Data",
    "w_o_Contraposition_Augmented_Data": "../Models/w_o_Contraposition_Augmented_Data",
}
model_gpus = 4
model_sampling_params = {
    "HERALD_Translator": {
        "max_tokens": 1024,
        "temperature": 0.99
    },
    "Kimina-Autoformalizer": {
        "max_tokens": 2048,
        "temperature": 0.6,
        "top_p": 0.95
    },
    "ATLAS_Translator_Series": {
        "max_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.9
    }
}


# Back_Translation Model
Back_Translation_model_name_or_path = "../Models/internlm2-math-plus-7b"
Back_Translation_model_gpus = 1
Back_Translation_model_sampling_params = {
    "max_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.9,
    "stop": ["[UNUSED_TOKEN_146]", "[UNUSED_TOKEN_145]", "<|im_end|>"]
}


# NLI_Check Model
NLI_Check_model_name_or_path = "qwen-turbo-latest"
NLI_Check_model_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
NLI_Check_model_api_key = "sk-xxx"
NLI_Check_model_sampling_params = {
    "max_tokens": 2048,
    "temperature": 0.1,
    "top_p": 0.9,
}




### Prompts
Prompt_NL_Translation = {
    "Kimina-Autoformalizer": "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n{informal_statement}",
    "HERALD_Translator": "Please translate the natural language statement to Lean4 code with the header\n**Name**\my_favorite_theorem\n**Informal statement**\n{informal_statement}\n",
    "ATLAS_Translator_Series": "You are an expert in the Lean4 theorem prover. Your task is to translate theorems from natural language into formal Lean4 statements. Please follow these guidelines:\n1. Carefully analyze the given theorem in natural language.\n2. Translate it into a correct and precise Lean4 formal statement.\n3. Use the following format for your response:\ntheorem tm_name :\n{{The theorem's Lean4 formal statement}}\n:= by sorry\n4. Focus solely on the translation. Do not attempt to prove the theorem or provide additional explanations.\n5. Ensure that your translation accurately captures all the mathematical concepts and relationships expressed in the natural language version.\n6. Use appropriate Lean4 syntax, including correct use of quantifiers, implications, and mathematical symbols.\n7. If the theorem involves specific mathematical structures (e.g., groups, rings, topological spaces), use the corresponding Lean4 definitions and notations.\nRemember, the goal is to create a syntactically correct and semantically accurate formalization in Lean4. Your translation should be faithful to the meaning of the original theorem while adhering to Lean4 conventions and best practices.\nNow please begin by carefully reading the natural language statement provided, and then proceed with your translation into Lean4.\n{informal_statement}"
}


Prompt_Back_Translation = "[UNUSED_TOKEN_146]user\nConvert the formal statement into natural language:\n```lean\n{formal_statement}\n```[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"


Prompt_Semantic_Contrast = """You are an experienced mathematics expert and educator with extensive experience in mathematical problem analysis. I need you to analyze the fundamental nature of the following two mathematical problems.

# Focus on:
1. Core mathematical concepts and principles
2. Problem-solving approaches and methodologies
3. Ultimate objectives of the problems

# Ignore:
1. Variations in wording
2. Changes in contextual scenarios

# Present your answer using exactly this format:
# Analysis\nInsert your analysis here
# Conclusion\nreply ||same|| or ||different|| with "||" format

# Please approach this analysis with professional rigor.
Math Problem 1: {informal_statement}
Math Problem 2: {back_translation}
"""