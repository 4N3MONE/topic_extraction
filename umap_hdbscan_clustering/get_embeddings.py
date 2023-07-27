import json
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='notebook', rc={'figure.figsize':(5,5)})


# 임베딩값 얻기
def get_embeddings(text):
    # 모델 설정
    model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta-multitask')  # or 'BM-K/KoSimCSE-bert-multitask'
    tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask')  # or 'BM-K/KoSimCSE-bert-multitask'

    # 텍스트 처리
    inputs_output = tokenizer(text, max_length=32, padding='max_length', truncation=True, return_tensors='pt')
    embeddings_output, _ = model(**inputs_output, return_dict = False)
    average_embedding = torch.mean(embeddings_output, dim=1)
    result_embedding = average_embedding.tolist()[0]
    return result_embedding

'''
if __name__ == "__main__":
    # json file에서 query와 topic추출
    with open('./preprocessed.json') as f:
        json_data = json.load(f)

    topic_list = []
    for i in range(len(json_data)):
        topic_list.append(json_data[i]['topic'])
    
    print(f'중복제거 전 토픽의 수: {len(topic_list)}')
    topic_list = list(set(topic_list))
    print(f'중복제거 후 토픽의 수: {len(topic_list)}')
    
    model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta-multitask')  # or 'BM-K/KoSimCSE-bert-multitask'
    tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask')  # or 'BM-K/KoSimCSE-bert-multitask'
    
    embed_list = []
    print('딥러닝 모델을 활용한 임베딩 진행 ')
    for t in tqdm(topic_list):
        embed_list.append(get_embeddings(t))

    # topic과 이에 대한 kosimCSE 임베딩값으로 구성된 데이터프레임 출력
    embed_df = pd.DataFrame(embed_list, index=topic_list, columns=range(768))
    embed_df.to_csv('./output/embedding_df.csv', encoding = 'utf-8-sig')
'''