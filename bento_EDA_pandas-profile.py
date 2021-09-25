# -*- coding: utf-8 -*-

"""
コメント：
baseline
"""

'''
データの読み込みと確認
'''

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ランダムシードの設定
import random
np.random.seed(1234)
random.seed(1234)

# データの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# データの確認
print(train.head())
print(train.dtypes)

'''
トレンド
'''

# ライブラリのインポート
from scipy.optimize import curve_fit

# ある値に漸近する指数関数の定義
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

xs = train.index.values
ys = train['y'].values

popt, pcov = curve_fit(func, xs, ys)

a1, b1, c1 = popt
y_reg = a1 * np.exp(-b1 * xs) + c1

# トレンドの可視化
plt.plot(train['y'])
plt.plot(y_reg)
plt.show()

# トレンドの除去
new_y = train['y'] - y_reg
new_y = pd.DataFrame({'new_y': new_y})

new_y.plot()
plt.show()

train['new_y'] = new_y

'''
特徴量エンジニアリング
'''

# ライブラリのインポート
from pandas_profiling import ProfileReport

# datetimeから特徴量dayの抽出
train["day"] = train['datetime'].apply(lambda x: x.split('-')[2] ).astype(int)

# 販売数の Mean/Median encoding」で数値に変換（今回はMedian encoding）
week_encoded = train.groupby('week').new_y.median()
print(week_encoded)

train['week'] = train['week'].replace(week_encoded) # weekの置き換え

# weekを引き算し、曜日がもたらす季節性を超簡易的に除去した販売数を用いて可視化
train['new_new_y'] = train['new_y'] - train['week']
train['new_new_y'].plot()
plt.show()

# 特徴量「カレー」の追加
train['curry'] = train['name'].apply(lambda x : 1 if x.find("カレー") >=0 else 0)

# 特徴量「人気メニュー」と「不人気メニュー」の追加
popular_menu = set(train[train['new_new_y']>15].name) # 人気メニューの抽出
train['popular'] = train['name'].apply(lambda x : 1 if x in popular_menu else 0) # 特徴量「人気メニュー」

unpopular_menu = set(train[train['new_new_y']<-15].name) # 不人気メニューの抽出
train['unpopular'] = train['name'].apply(lambda x : 1 if x in unpopular_menu else 0) # 特徴量「不人気メニュー」

# 特徴量「お楽しみメニュー」の追加
train['fun'] = train['remarks'].apply(lambda x: 1 if x=="お楽しみメニュー" else 0)

# 欠損値の補完
train['kcal'] = train['kcal'].fillna(train['kcal'].mean()) # 平均値で補完
train['payday'] = train['payday'].fillna(0)

# weatherのmedian encoding
weather_word = ['快晴','晴','曇','雨','雪','雷']
row_index = []
number = 1

for x in weather_word:

    row_index = train['weather'].str.contains(x, na=False)
    train.loc[row_index, ['weather']] = number

    if number < 4:
        number += 1
    else:
        number = 4

    row_index = []

weather_encoded = train.groupby('weather').new_y.median()
train['weather'] = train['weather'].replace(weather_encoded)

# 月平均との差分の特徴量「temp」を追加
train['month'] = train['datetime'].apply(lambda x : int(x.split("-")[1]))
temp_mean = train.groupby('month').temperature.mean()
train['month'] =  train['month'].replace(temp_mean)
train['temp'] = train['temperature'] - train['month']

# pandas-profiling
profile = ProfileReport(train)
profile.to_file('profile_report_ver2.html')