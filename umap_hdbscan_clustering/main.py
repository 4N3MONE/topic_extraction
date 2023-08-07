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
    embed_df.to_csv('./output_new/embedding_df.csv', encoding = 'utf-8-sig')
    
    min_dist = [0.1, 0.2, 0.3, 0.4, 0.5]
    below_cluster_list = []
    below_data_list = []
    file_name = []
    
    for i in range(3, 7):
        for j in min_dist:
            for k in range(2, 5):
                umap_df = umap_reducer(neighbors = i, min_distance = j, reduce_dim = k, data = embed_df)
                umap_df.to_csv('./output_new/umap_'+str(i)+'n_'+str(j)+'dist'+str(k)+'dim.csv')
                fin_df = hdbscan_cluster(min_size = 2, min_sample = None, cluster_distance = 0.01, data = umap_df)
                
                #실루엣 스코어 계산 및 미달 군집과 데이터개수 계산
                X=fin_df.iloc[:,:-1]
                labels=fin_df.iloc[:,-1]
                score_samples=silhouette_samples(X, labels)
                fin_df['silhouette_coeff']=score_samples

                cluster_scores = fin_df.groupby('hdbscan_label')['silhouette_coeff'].mean()
                cluster_below = cluster_scores[cluster_scores < 0.25].index
                data_filtered = fin_df[~fin_df['hdbscan_label'].isin(cluster_below)]
                
                file_name.append('cluster_'+str(i)+'n_'+str(j)+'dist'+str(k)+'dim.csv')  
                below_cluster_list.append(len(cluster_below))
                below_data_list.append(len(fin_df) - len(data_filtered))
                fin_df.to_csv('./output_new/cluster_'+str(i)+'n_'+str(j)+'dist'+str(k)+'dim.csv')
                
                print(f"n{i}, dist{j}, {k}dim 조건 실루엣 스코어가 0.25 미만인 군집 개수: {len(cluster_below)}\n")
                print(f"n{i}, dist{j}, {k}dim 조건 미달 군집에 포함되는 총 데이터 개수: {len(fin_df) - len(data_filtered)}\n")
    
    data = {'파일명': file_name,
        '미달 군집 수': below_cluster_list,
        '미달 데이터 수':below_data_list}
    df = pd.DataFrame(data)
    df.to_csv('./summary.csv')
