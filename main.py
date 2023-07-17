from privateManager import get_key
import openai
from src.chatgpt_handler import ChatGPT
import time
import pandas as pd
import pickle
from tqdm import tqdm

chatgpt = ChatGPT(get_key('openai'))

def load_data(path='./data/kullm.json'):
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def extract_topic(start_index, interval, log):
    error_message = None
    result = []
    try:
        instructions = data[start_index : start_index + interval]
        texts = chatgpt.request_to_chatgpt(instructions)#[-interval:]
        log.write(f'\ninstructions[{start_index} : {start_index + interval}]\n' + '\n'.join(map(lambda x : x['instruction'], instructions)) + '\n\n')
        log.write('\n'.join(texts) + '\n')
        for text in texts :
            result.append(text.split(':')[-1].replace('"',''))
        if len(result)!= interval: 
            return None, f"The length of result is not matched to interval, the length of result is {len(result)} but interval is {interval}"
    except Exception as e:
        error_message = e
    
    return result, error_message


log = open('./log.txt', 'w')
data = load_data()
interval = 1
error_limit = 5
topics = []
for start_index in tqdm(range(0,50, interval)):
    error_count = 0
    while(True):
        try:
            topic, error = extract_topic(start_index, interval, log)
            if error == None:
                topics.extend(topic)
                is_comped = True
                break
        except:
            pass
        error_count += 1
        time.sleep(0.5)
        log.write(f'ERROR in {start_index}: {error}\n')
        if error_count >= error_limit:
            log.write(f'ERROR count exceed {error_limit}: {start_index}\n')
            log.close()
            exit()
print('finished.')
log.close()


with open('result.p', 'wb') as f:
    pickle.dump(topics, f)
print('pickle saved.')
df = pd.DataFrame(topics)
df.to_csv('result.csv',encoding='utf-8')
print('csv saved.')
