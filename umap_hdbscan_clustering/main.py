import json
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import hdbscan
from sklearn.metrics import silhouette_samples, silhouette_score
from get_embeddings import get_embeddings
from umap_reduce import umap_reducer
from hdbscan_result import hdbscan_cluster

sns.set(style='white', context='notebook', rc={'figure.figsize':(10,7)})

# unicode minus를 사용하지 않기 위한 설정 (minus 깨짐현상 방지)
plt.rcParams['axes.unicode_minus'] = False
#plt.rcParams['font.family'] = 'Malgun Gothic'


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
    
    #umap으로 차원축소한 데이터프레임과 시각화결과 저장
    umap_df = umap_reducer(neighbors = 5, min_distance = 0.1, reduce_dim = 2, data = embed_df)
    
    #hdbscan을 통해 군집화를 수행한 데이터프레임과 시각화결과 저장
    fin_df = hdbscan_cluster(min_size = 2, min_sample = None, cluster_distance = 0., data = umap_df)