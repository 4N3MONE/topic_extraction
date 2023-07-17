import openai

class ChatGPT:
    def __init__(self, key, prompt_path = "./prompt.txt"):
        self.prompt_path = prompt_path
        openai.api_key = key
        self.prompt = open(self.prompt_path).read() + "\n"

    def request_to_chatgpt(self, instruction, return_all=False):
        chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "당신은 도움이 필요한 어시스턴트입니다."},
            {"role": "user", "content": self.prompt + f"'Input': \"{instruction['instruction']}\"\nTopic:"}
        ]
        )
        topics = chat_response['choices'][0]['message']['content']
        return chat_response if return_all else topics