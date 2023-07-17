from privateManager import get_key
from src.chatgpt_handler import ChatGPT
import time
from tqdm import tqdm
import json

ERROR_LIMIT = 5
SLEEP_TIME = 0.5

def func(a, b):
    return len(a) + len(b)

def load_data(path='./data/kullm.json'):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_data(obj, path='./data/result_01.json'):
    with open(path, 'w') as f:
        json.dump(obj, f)

if __name__=="__main__":
    chatgpt = ChatGPT(get_key('openai'))
    data = load_data()
    new_data = []
    log = open('./log.txt', 'w')
    
    start_index = 0
    end_index = len(data)
    
    for idx in tqdm(range(start_index, end_index)):
        error_count = 0
        new_data.append({
            'instruction' : '',
            'topic' : '',
            'score' : 0.0
        })
        while(True):
            try:
                new_data[-1]['instruction'] = data[idx]['instruction']
                new_data[-1]['topic'] = chatgpt.request.to_chatgpt(data[idx])
                new_data[-1]['score'] = map(func, *new_data[-1].values()[:2])
                log.write(f'instruction[{idx}]: {data[idx]["instruction"]}\n')
                log.write(f'output[{idx}]: {new_data[-1]["topic"]}\n\n')
                break
            except Exception as error:
                error_count += 1
                time.sleep(SLEEP_TIME)
                log.write(f'ERROR in {idx}: {error}\n')
                if error_count == ERROR_LIMIT:
                    log.write(f'ERROR count exceed {ERROR_LIMIT}: {idx}\n')
                    log.close()
                    exit()
    print(f'successfully finished: [{start_index}:{end_index}]')
    log.close()
    
    save_data(new_data)