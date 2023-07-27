from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
import umap.umap_ as umap
sns.set(style='white', context='notebook', rc={'figure.figsize':(10,7)})

# unicode minus를 사용하지 않기 위한 설정 (minus 깨짐현상 방지)
plt.rcParams['axes.unicode_minus'] = False

### default setting:
#n_neighbors: float (optional, default 15)
#min_dist: float (optional, default 0.1)
#n_components: int (optional, default 2)
def umap_reducer(neighbors, min_distance, reduce_dim, data):
  print('UMAP을 통한 차원축소를 시작합니다...')
  reducer = umap.UMAP(low_memory = False, n_neighbors=neighbors, min_dist = min_distance, n_components = reduce_dim)
  embedding = reducer.fit_transform(data)
  # 데이터프레임 형태로 변환
  embedding_df = pd.DataFrame(embedding, index = data.index.to_list())
  embedding_df.to_csv('./output/umap_reduce_result.csv', encoding = 'utf-8-sig')
  # 시각화
  plt.scatter(embedding.T[0], embedding.T[1], color='b')
  plt.legend(loc = 2, bbox_to_anchor = (1,1))
  plt.savefig('./output/umap_result.png')
  plt.show()
  return embedding_df

'''
if __name__ == "__main__":
    embed_df = pd.read_csv('./output/embedding_df.csv', index_col = 0)
    umap_df = umap_reducer(neighbors = 5, min_distance = 0.1, reduce_dim = 2, data = embed_df)
'''