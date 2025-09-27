### Models
Student_model_name_or_path = "to_be_filled"
Student_model_gpus = 1
Student_model_sampling_params = {
    "max_tokens": 1024,
    "temperature": 0.6,
    "top_p": 0.9
}


Teacher_model_name_or_path = "deepseek-chat"
Teacher_model_base_url = "https://api.deepseek.com"
Teacher_model_api_key = "sk-xxx"
Teacher_model_sampling_params = {
    "temperature": 0.7
}



### Prompts
Prompt_NL_Generation = """You are an expert mathematics professor tasked with creating proof problems for undergraduate mathematics majors. Your assignment is to construct a proof problem that integrates {concept1} from {domain1} and {concept2} from {domain2}.

Requirements:
1. Create a concise theorem appropriate for undergraduate mathematics majors.
2. The theorem should be brief, not exceeding 50 words.
3. Incorporate both specified concepts into the theorem naturally.
4. State the theorem clearly and concisely.
5. Ensure the theorem is simple enough to be easily translated into Lean4.

Format exactly:
# Answer\nInsert your problem with "||" format, i.e. ||Theorem: Insert the theorem in natural language here.||"""


Prompt_NL_Translation = """You are an expert in the Lean4 theorem prover. Your task is to translate theorems from natural language into formal Lean4 statements. Please follow these guidelines:
1. Carefully analyze the given theorem in natural language.
2. Translate it into a correct and precise Lean4 formal statement.
3. Use the following format for your response:
theorem tm_name : {{The theorem's Lean4 formal statement}} := by sorry
4. Focus solely on the translation. Do not attempt to prove the theorem or provide additional explanations.
5. Ensure that your translation accurately captures all the mathematical concepts and relationships expressed in the natural language version.
6. Use appropriate Lean4 syntax, including correct use of quantifiers, implications, and mathematical symbols.
7. If the theorem involves specific mathematical structures (e.g., groups, rings, topological spaces), use the corresponding Lean4 definitions and notations.
Remember, the goal is to create a syntactically correct and semantically accurate formalization in Lean4. Your translation should be faithful to the meaning of the original theorem while adhering to Lean4 conventions and best practices.
Now please begin by carefully reading the natural language statement provided, and then proceed with your translation into Lean4.
{informal_statement}"""


Prompt_FL_Revision = """You are a math expert and an expert in Lean4. Your task is to modify the Lean4 code based on the given natural language description of a theorem, the corresponding Lean4 code, and the error message from the Lean compiler.

Requirements:
1. Correct the Lean4 code to make it compile successfully.
2. Lean4 code may lack or have additional declarations of certain content. You can add or remove them as much as possible to keep it consistent with the natural language description.
3. No need to import any packages, because Mathlib will be imported by default as import Mathlib.
4. Carefully read the content and provide your modified answer: **Lean4 code**\n{formal_statement}\n**Compiler error messages**\n{compiler_error_messages}\n**natural language statement**\n{informal_statement}

Format exactly:
# Analysis\nInsert your analysis here
# Answer\nInsert your revised Lean4 code with "||" format, i.e. ||theorem tm_name your revised Lean4 code here := by sorry||"""


Prompt_FL_Alignment = """You are a math expert and an expert in Lean4. Your task is to check the alignment between the given natural language description of a theorem and the corresponding Lean4 code.

Requirements:
1. Determine whether the Lean4 code is missing declarations of certain entities.
2. Assess whether the Lean4 code accurately represents the theorem described in the natural language.
3. Carefully read the content and provide your answer: **Lean4 code**\n{formal_statement}\n**natural language statement**\n{informal_statement}

Format exactly:
# Analysis:\nInsert your analysis here
# Answer\nreply ||good||, ||average|| or ||poor||"""


Prompt_FL_Translation = """You are a math expert and an expert in Lean4. Your task is to translate theorems from Lean4 code into natural language.

Requirements:
1. Focus solely on the translation. Do not attempt to prove the theorem or provide additional explanations.
2. The theorem's natural language statement should be brief, not exceeding 50 words.
3. Carefully analyze the given theorem in Lean4 code {formal_statement} and provide your translation in natural language.

Format exactly:
# Answer\nInsert your translation with "||" format, i.e. ||Theorem: Insert the theorem in natural language here.||"""