# -*- coding: utf-8 -*-
"""
コメント：
baseline作成前のEDA
"""

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 学習データの読み込みと確認
train = pd.read_csv('./data/train.csv')
print(train.dtypes) 
print(train.head())

# 欠損値の確認
print(train.isnull().sum())
print(train['precipitation'].value_counts())

# 販売数の推移
train['y'].plot.line(title='sales of Bento')
plt.xlabel('time step')
plt.ylabel('sales')
plt.show()

# 販売数のヒストグラム
train['y'].plot.hist(title='sales of Bento')
plt.xlabel("sales")
plt.ylabel("frequency")
plt.show()

# 販売数と気温の散布図
train.plot.scatter( x='temperature', y='y', c="blue", title="scatter plot of temperature and sales" )
plt.xlabel("temperature")
plt.ylabel("sales")
plt.show()

# 販売数とカロリーの散布図
train.plot.scatter( x='kcal', y='y', c="blue", title="scatter plot of calory and sales" )
plt.xlabel("calory")
plt.ylabel("sales")
plt.show()

# 曜日による販売数
sns.boxplot( x='week', y='y', data=train, order=["月","火","水","木","金"] )
plt.title("sales of each week")
plt.ylabel("sales")
plt.show()

# 曜日による販売数2
plt.figure(figsize=(10,6))
sns.lineplot( x=train.index, y='y', hue='week', data=train )
plt.xlabel("time step")
plt.ylabel("sales")
plt.title("sales of box lunch")
plt.show()


# remarksカラムの欠損値を埋める
train['remarks'].fillna('No description', inplace=True)

# remarksによる販売数の違い
sns.boxplot(x='y', y='remarks', data=train)
plt.title("sales of each remarks content")
plt.xlabel("sales")
plt.show()

# eventカラムの欠損値を埋める
train['event'].fillna('No event', inplace=True)

# eventによる販売数の違い
sns.boxplot(x='y', y='event', data=train)
plt.title("sales of each event")
plt.xlabel("sales")
plt.show()

# paydayカラムの欠損値を埋める
train['payday'].fillna(0, inplace=True)

# 給料日かどうかによる販売数の違い
sns.boxplot(x='payday', y='y', data=train)
plt.title("sales of payday or not")
plt.xlabel("sales")
plt.show()

# 天気による販売数の違い
sns.boxplot(x='weather', y='y', data=train, order=["快晴","晴れ","曇","薄曇","雨","雪","雷電"])
plt.title("sales of each weather category")
plt.ylabel("sales")
plt.show()