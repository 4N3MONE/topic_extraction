
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
sns.set(style='white', context='notebook', rc={'figure.figsize':(10, 7)})

# unicode minus를 사용하지 않기 위한 설정 (minus 깨짐현상 방지)
plt.rcParams['axes.unicode_minus'] = False

def hdbscan_cluster(min_size, min_sample, cluster_distance, data):
  print('hdbscan을 활용한 군집화를 시작합니다...')
  hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples = min_sample, cluster_selection_epsilon = cluster_distance, approx_min_span_tree=True)
  hdbscan_model.fit(data)
  hdbscan_labels = list(hdbscan_model.labels_)
  print(f'총 군집의 개수: {hdbscan_model.labels_.max()}')
  data['hdbscan_label'] = hdbscan_model.labels_
  data['hdbscan_label'] = data['hdbscan_label'].astype(str)
  #data = data.rename(columns = {0: 'x', 1:'y'}, inplace = True)
  data.to_csv('./output/hdbscan_result.csv', encoding = 'utf-8-sig')
  sns.set(rc={'figure.figsize':(5,5)})
  ax = sns.scatterplot(x = 0, y = 1, hue = 'hdbscan_label', data = data, legend = False)
  plt.savefig('./output/hdbscan_scatter.png')
  plt.show()
  return data

'''
if __name__ == "__main__":
    umap_df  = pd.read_csv('./output/umap_reduce_result.csv', index_col = 0)
    fin_df = hdbscan_cluster(min_size = 2, min_sample = None, cluster_distance = 0., data = umap_df)
'''