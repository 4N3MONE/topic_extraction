import openai
import re

class ChatGPT:
    def __init__(self, key, prompt_path = "./prompt.txt"):
        self.prompt_path = prompt_path
        openai.api_key = key

    def encode_prompt(self, prompt_instructions):
        """Encode multiple prompt instructions into a single string."""
        prompt = open(self.prompt_path).read() + "\n"

        for task_dict in prompt_instructions:
            query = '\"'+task_dict['instruction']+'\"' #+ '\"'+ task_dict['input'] + '\"'
            prompt += f"'Input': {query}\n"
        prompt+="Topic:"
        return prompt

    def request_to_chatgpt(self, instructions):
        prompt = self.encode_prompt(instructions)
        chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "당신은 도움이 필요한 어시스턴트입니다."},
            {"role": "user", "content": prompt},
        ]
        )
        topics = chat_response['choices'][0]['message']['content'].split('\n')
<<<<<<< HEAD
        return topics
    
    
=======
        return topics
>>>>>>> a880eb2 (Delete: demo.py)
