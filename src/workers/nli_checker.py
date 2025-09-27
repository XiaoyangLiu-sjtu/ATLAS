from utils import APIModel


class NLICheck(APIModel):
    def __init__(self, name_or_path, base_url, api_key, sampling_params, prompt):
        super().__init__(name_or_path, base_url, api_key, sampling_params)
        self.prompt = prompt

    def get_query(self, task):
        query = self.prompt.format(
            informal_statement = task["informal_statement"],
            back_translation = task["back_translation"]
        )

        return [
                {"role": "user", "content": query}
            ]