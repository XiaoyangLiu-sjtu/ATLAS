from utils import LocalModel


class BackTranslation(LocalModel):
    def __init__(self, name_or_path, gpus, seed, sampling_params, prompt):
        super().__init__(name_or_path, gpus, seed, sampling_params, prompt)
        self.prompt = prompt

    def get_query(self, task, system_content):
        query = self.prompt.format(
            formal_statement=task["formal_statement"]
        )
        
        message = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ]
        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)