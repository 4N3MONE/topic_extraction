import openai

class ChatGPT:
    def __init__(self, key, prompt_path = "./prompt_cluster.txt"):
        self.prompt_path = prompt_path
        openai.api_key = key
        self.prompt = open(self.prompt_path).read() + "\n"
    
    def get_completion(self, instruction, model='gpt-3.5-turbo'):
        chat_response = openai.ChatCompletion.create(
        model= model,
        messages=[
            {"role": "system", "content": "당신은 도움이 필요한 어시스턴트입니다."},
            {"role": "user", "content": f"{self.prompt} 'Input': \"{instruction}\"\nTopic:"}
        ])
        chat_response = chat_response['choices'][0]['message']['content'].split('\n')
        return chat_response

    def request_to_chatgpt(self, instruction, model='gpt-3.5-turbo'):
            chat_response = self.get_completion(instruction['instruction'], model)
            return chat_response

    def request_to_chatgpt_cluster(self, instruction, model='gpt-3.5-turbo'):
        chat_response = self.get_completion(instruction['topics'], model)
        topics = ' '.join(chat_response)
        return topics