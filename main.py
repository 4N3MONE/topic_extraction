from privateManager import get_key
import openai
from src.chatgpt_handler import ChatGPT
import time
import pandas as pd
import pickle
from tqdm import tqdm

chatgpt = ChatGPT(get_key('openai'))
log = open('./log.txt', 'w')

def load_data(path='./data/kullm.json'):
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return data

data = load_data()
ERROR_LIMIT = 5
topics = []
for idx in tqdm(range(0, 50)):
    error_count = 0
    while(True):
        try:
            output = chatgpt.request.to_chatgpt(data[idx])
            topics.append(output)
            log.write(f'instruction[{idx}]: {data[idx]}\n')
            log.write(f'output[{idx}]: {output}\n\n')
            break
        except Exception as error:
            error_count += 1
            time.sleep(0.5)
            log.write(f'ERROR in {idx}: {error}\n')
            if error_count == ERROR_LIMIT:
                log.write(f'ERROR count exceed {ERROR_LIMIT}: {idx}\n')
                log.close()
                exit()
print('successfully finished.')
log.close()


with open('result.p', 'wb') as f:
    pickle.dump(topics, f)
print('pickle saved.')
df = pd.DataFrame(topics)
df.to_csv('result.csv',encoding='utf-8')
print('csv saved.')
