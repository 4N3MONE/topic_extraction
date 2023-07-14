from privateManager import get_key
import openai
from src.chatgpt_handler import ChatGPT
import time
import pandas as pd
import pickle

chatgpt = ChatGPT(get_key('openai'))

def get_sample_instructions(path='./data/sample.txt'):
    sample_instructions = []
    with open(path, 'r') as f:
        sample_instructions = '\n'.join(f.readlines())

    return sample_instructions

def reconstruct_kullm(input_path='./data/kullm-v2.jsonl', output_path=None):
    import json
    data = []
    with open(input_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    if output_path == None:
        output_path = '.'.join(input_path.split('.')[:-1]) + '.reconstruct.json'
    with open(output_path, 'w') as f:
        json.dump(data, f)
    return output_path

def apply_topics(topics, start_index, interval):
    global data
    for i in range(start_index, start_index + interval):
        data[i]['topic'] = topics[i - start_index]

reconstruct_kullm(output_path='./data/kullm.json')

def load_data(path='./data/kullm.json'):
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def extract_topic(start_index, interval):
    result = []
    instructions = data[start_index : start_index + interval]
    texts = chatgpt.request_to_chatgpt(instructions)[-interval:]
    for text in texts :
        result.append(text.split('Topic: ')[1].replace('"',''))
    return result

from tqdm import tqdm
log = open('./log.txt', 'w')
data = load_data()
interval = 20
error_limit = 100
topics = []
for start_index in tqdm(range(0, 10000, interval)):
    is_comped = False
    error_count = 0
    while(not is_comped):
        try:
            topics.extend(extract_topic(start_index, interval))
            is_comped = True
            print('finished.')
        except:
            error_count += 1
            time.sleep(0.5)
            log.write(f'ERROR: {start_index}\n')
            if error_count >= error_limit:
                log.write(f'FATAL ERROR: {start_index}\n')
                log.close()
                exit()
log.close()


with open('result.p', 'wb') as f:
    pickle.dump(topics, f)
print('pickle saved.')

df = pd.DataFrame(topics)
df.to_csv('result.csv',encoding='utf-8')
print('csv saved.')