#%%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

# 한글 깨짐 문제 해결
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# %%
### 데이터 확인
file = 'data/logistic.csv'
df = pd.read_csv(file, encoding='cp949')
df.head()

# %%
### 유통량 EDA
## 데이터 전처리
# shpae
print(f"# df shape: {df.shape}")

# %%
# data type
df.info()

# %%
# null
print(f"# df null:\n{df.isnull().sum()}")

# %%
# outlier
df.describe()

# %%
## 배송지역 탐색
df.head()
# %%
# 컬럼명 간소화
df.columns = ['idx', 'start', 'end', 'pdt', 'cnt']
df.head(1)

# %%
# 배송 출발지(start) 개수 확인 (4,229)
df['start'].value_counts()

# %%
# 배송 도착지(end) 개수 확인 (26,875)
df['end'].value_counts()

# %%
# 총 지역(출발+도착) 개수 확인 (30,455)
all_list = list(df['start']) + list(df['end'])
unique_list = set(all_list)
print(f"총 지역 수: {len(unique_list)}")

# %%
## 물품 카테고리 탐색
# 물품 종류 확인
print(f"상품 종류 개수: {df['pdt'].nunique()}")
df['pdt'].value_counts().head(10)

#%%
# 운송장 건수 기준, 내림차순 정렬
df_cnt = df.groupby('pdt', as_index=False)['cnt'].sum()
df_cnt = df_cnt.sort_values(by=['cnt'], ascending=False)
df_cnt.head(10)

#%%
# 운송장 건수 기준, 내림차순 정렬 시각화
plt.bar(df_cnt['pdt'][0:30], df_cnt['cnt'][0:30], label='cnt')
plt.xticks(rotation=-90)
plt.gcf().set_size_inches(25,5)

#%%
# 너무 큰 값을 차지하는 농산물 제거 후 운송장 건수 기준, 내림차순 정렬 시각화
plt.bar(df_cnt['pdt'][1:30], df_cnt['cnt'][1:30], label='cnt')
plt.xticks(rotation=-90)
plt.gcf().set_size_inches(25,5)

# %%
### 지역별 유통량 확인
df.head()

# %%
## 출발지
df['start'].value_counts()

# %%
# 출발지 기준 상위 500개 장소가 출발지 건수의 약 80%를 담당
df['start'].value_counts().head(500).sum() / df['start'].value_counts().sum()

# %%
## 도착지
df['end'].value_counts()

# %%
# 도착지 기준 상위 약 20,000 장소가 도착지 건수의 약 80%를 담당
df['end'].value_counts().head(20500).sum() / df['end'].value_counts().sum()

# %%
# 출발지 배송 건수, 도착지 배송 건수, 총 건수로 이루어진 dataframe 생성
df_start_end = pd.DataFrame({'start': df['start'].value_counts(), 'end': df['end'].value_counts()})
df_start_end.fillna(0, inplace=True)
df_start_end['total'] = df_start_end['start'] + df_start_end['end']
df_start_end.head()

# %%
# 배송 건수가 가장 많은 지역
df_start_end.sort_values(by=['total'], ascending=False).head(10)

# %%
## 유통경로 분석
df.head()

# %%
# 배송 출발지, 도착지, 물품 건수로 이루어진 dataframe 생성
df_route = df.groupby(['start', 'end'], as_index=False)['pdt'].count()
df_route.columns = ['start', 'end', 'pdt']
df_route = df_route.sort_values(by=['pdt'], ascending=False)
df_route.head(10)

# %%
# start = 4141000031030100, end = 5013000635005300 확인
# 물품이 다양하기 때문에 카테고리화 곤란
df[(df['start']==4141000031030100) & (df['end']==5013000635005300)]

#%%
## 출발지 기준 배차 분석
# 출발지 기준으로 가장 많이 배송하는 물품을 대표 물품으로 선정 (start=4141000031030100)
pd.DataFrame(df[df['start']==4141000031030100].groupby('pdt')['cnt'].sum().sort_values(ascending=False)).reset_index().head(1)
# %%
# 모든 배송 출발지에 대해 위의 작업을 수행하
start_list = []
starts = list(df['start'].unique())

for start in starts:
    a = pd.DataFrame(df[df['start']==start].groupby('pdt')['cnt'].sum().sort_values(ascending=False)).reset_index().head(1)
    a['id'] = start
    start_list.append(a)

df_accum_start = pd.concat(start_list)
df_accum_start.head()

# %%
# 물품 전체 배송량 데이터 생성
df_pdt_sum = pd.DataFrame(df.groupby('pdt', as_index=False)['cnt'].sum())
df_pdt_sum.columns = ['pdt', 'total']
df_pdt_sum.head()

# %%
# 출발지 기준 물품 배송량에 left join
df_merge = pd.merge(df_accum_start, df_pdt_sum, on='pdt', how='left')
df_merge = df_merge[['id', 'pdt', 'cnt', 'total']]
df_merge.head()

#%%
