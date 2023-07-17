import openai

class ChatGPT:
    def __init__(self, key, prompt_path = "./prompt.txt"):
        self.prompt_path = prompt_path
        openai.api_key = key
        self.prompt_text = open(self.prompt_path).read() + "\n"

    def encode_prompt(self, prompt_instruction):
        prompt += f"'Input': \"{prompt_instruction['instruction']}\"\nTopic:"
        return prompt

    def request_to_chatgpt(self, instruction, return_all=False):
        prompt = self.encode_prompt(instruction)
        chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "당신은 도움이 필요한 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ]
        )
        topics = chat_response['choices'][0]['message']['content']
        return chat_response if return_all else topics