import openai

class ChatGPT:
    def __init__(self, key, prompt_path = "./prompt_cluster.txt"):
        self.prompt_path = prompt_path
        openai.api_key = key
        self.prompt = open(self.prompt_path).read() + "\n"

    def request_to_chatgpt(self, instruction, return_all=False):
        chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 도움이 필요한 어시스턴트입니다."},
            {"role": "user", "content": self.prompt + f"'Input': \"{instruction['instruction']}\"\nTopic:"}
        ]
        )
        topics = chat_response['choices'][0]['message']['content'].split('\n')
        return topics

    def request_to_chatgpt_cluster(self, instruction, return_all=False):
        chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "어시스턴트로써 지시에 답변하세요. 답변 내용은 지시에서 요구하는 형식이어야 합니다."},
            {"role": "user", "content": self.prompt + f"{instruction['topics']}\"\n클러스터 이름:"}
        ]
        )
        topics = ' '.join(chat_response['choices'][0]['message']['content'].split('\n'))
        return topics
