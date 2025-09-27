from utils import APIModel, LocalModel


class NLGeneration(APIModel):
    def __init__(self, name_or_path, base_url, api_key, sampling_params, prompt):
        super().__init__(name_or_path, base_url, api_key, sampling_params)
        self.prompt = prompt

    def get_query(self, task):
        query = self.prompt.format(
            concept1=task[0]["concept"],
            domain1=task[0]["domain"],
            concept2=task[1]["concept"],
            domain2=task[1]["domain"]
        )

        return [
                {"role": "user", "content": query}
            ]


class FLRevision(APIModel):
    def __init__(self, name_or_path, base_url, api_key, sampling_params, prompt):
        super().__init__(name_or_path, base_url, api_key, sampling_params)
        self.prompt = prompt

    def get_query(self, task):
        query = self.prompt.format(
            formal_statement=task["formal_statement"],
            compiler_error_messages=task["compiler_error_messages"],
            informal_statement=task["informal_statement"]
        )

        return [
                {"role": "user", "content": query}
            ]
    

class FLAlignment(APIModel):
    def __init__(self, name_or_path, base_url, api_key, sampling_params, prompt):
        super().__init__(name_or_path, base_url, api_key, sampling_params)
        self.prompt = prompt

    def get_query(self, task):
        query = self.prompt.format(
            formal_statement=task["formal_statement"],
            informal_statement=task["informal_statement"]
        )

        return [
                {"role": "user", "content": query}
            ]
    

class FLTranslation(APIModel):
    def __init__(self, name_or_path, base_url, api_key, sampling_params, prompt):
        super().__init__(name_or_path, base_url, api_key, sampling_params)
        self.prompt = prompt

    def get_query(self, task):
        query = self.prompt.format(
            formal_statement=task["output"]
        )

        return [
                {"role": "user", "content": query}
            ]


class NLTranslation(LocalModel):
    def __init__(self, name_or_path, gpus, seed, sampling_params, prompt):
        super().__init__(name_or_path, gpus, seed, sampling_params, prompt)
        self.prompt = prompt

    def get_query(self, task, system_content):
        query = self.prompt.format(
            informal_statement=task["informal_statement"]
        )

        message = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ]
        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)