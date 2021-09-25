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

# 特徴量nameのアノテーション
train['pork'] = train['name'].apply(lambda x: 1 if "ハンバーグ" in x \
                                    or 'かつ' in x\
                                    or 'カツ' in x\
                                    or '豚' in x\
                                    or 'ポーク' in x\
                                    or '回鍋肉' in x\
                                    or 'ロース' in x\
                                    or 'ロコモコ' in x\
                                    or 'チンジャオロース' in x\
                                    or '青椒肉絲' in x\
                                    or 'ベーコン' in x\
                                    or '八宝菜' in x\
                                    else 0)

train['beef'] = train['name'].apply(lambda x: 1 if "ハンバーグ" in x \
                                    or 'メンチ' in x\
                                    or '肉じゃが' in x\
                                    or '牛' in x\
                                    or 'ビーフ' in x\
                                    or 'ロコモコ' in x\
                                    else 0)

train['chicken'] = train['name'].apply(lambda x: 1 if "鶏" in x \
                                    or 'チキン' in x\
                                    or '親子' in x\
                                    else 0)

train['fish'] = train['name'].apply(lambda x: 1 if "魚" in x \
                                    or 'サバ' in x\
                                    or 'ます' in x\
                                    or 'サーモン' in x\
                                    or 'アジ' in x\
                                    or 'キス' in x\
                                    or 'かじき' in x\
                                    or 'さわら' in x\
                                    or 'カレイ' in x\
                                    else 0)

train['shrimp'] = train['name'].apply(lambda x: 1 if "海老" in x else 0)

train['otherSeaFood'] = train['name'].apply(lambda x: 1 if "イカ" in x\
                                            or "カキ" in x\
                                            or "ホタテ" in x else 0)

train['vegetable'] = train['name'].apply(lambda x: 1 if "野菜" in x\
                                            or "カキ" in x\
                                            or "ホタテ" in x else 0)
    
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

'''
モデルの構築と評価
'''

# ライブラリのインポート
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from statistics import mean

# 説明変数と目的変数の設定
X_train = train[['popular',
                 'unpopular',
                 'fun',
                 'temperature',
                 'week',
                 'day',
                 'temp',
                 'kcal',
                 'month',
                 'curry',
                 'weather',
                 'otherSeaFood',
                 'chicken',
                 'pork',
                 'beef'
                 ]]
Y_train = train['new_y']

# 10分割する
folds = 10
kf = KFold(n_splits=folds)

# ハイパーパラメータの設定
params = {
    'objective':'regression',
    'random_seed':1234    
}

# 各foldごとに作成したモデルごとの予測値を保存
models = []
rmses = []
oof = np.zeros(len(X_train))

for train_index, val_index in kf.split(X_train):
    x_train = X_train.iloc[train_index]
    x_valid = X_train.iloc[val_index]
    new_y_train = Y_train.iloc[train_index]
    new_y_valid = Y_train.iloc[val_index]
    y_reg_split = y_reg[val_index]
    
    lgb_train = lgb.Dataset(x_train, new_y_train)
    lgb_eval = lgb.Dataset(x_valid, new_y_valid, reference=lgb_train)    
    
    model = lgb.train(params,
                      lgb_train, 
                      valid_sets=lgb_eval, 
                      num_boost_round=1000, # 学習回数の実行回数
                      early_stopping_rounds=20, # early_stoppingの判定基準
                      verbose_eval=10)
    
    new_y_pred = model.predict(x_valid, num_iteration=model.best_iteration)
    y_pred = new_y_pred + y_reg_split
    y_valid = new_y_valid + y_reg_split
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(rmse)
    
    models.append(model)
    rmses.append(rmse)
    oof[val_index] = y_pred

# 平均RMSEを計算する
mean(rmses)

# 現状の予測値と実際の値の違いを可視化
actual_pred_df = pd.DataFrame({
    'actual':Y_train + y_reg,
    'pred': oof})

actual_pred_df.plot(figsize=(12,5))

"""
予測精度：
12.944191915664229
"""

'''
テストデータの予測
'''

# datetimeから特徴量dayの抽出
test["day"] = test['datetime'].apply(lambda x: x.split('-')[2] ).astype(int)

# trainデータからのものを使う
test['week'] = test['week'].replace(week_encoded)
test['popular'] = test['name'].apply(lambda x : 1 if x in popular_menu else 0)
test['unpopular'] = test['name'].apply(lambda x : 1 if x in unpopular_menu else 0)

# weatherのmedian encoding
weather_word = ['快晴','晴','曇','雨','雪','雷']
row_index = []
number = 1

for x in weather_word:

    row_index = test['weather'].astype(str).str.contains(x, na=False)
    test.loc[row_index, ['weather']] = number

    if number < 4:
        number += 1
    else:
        number = 4

    row_index = []


test['weather'] = test['weather'].replace(weather_encoded)

# 特徴量「カレー」の追加
test['curry'] = test['name'].apply(lambda x : 1 if x.find("カレー") >=0 else 0)

# 特徴量「お楽しみメニュー」の追加
test['fun'] = test['remarks'].apply(lambda x: 1 if x=="お楽しみメニュー" else 0)

# 特徴量nameのアノテーション
test['pork'] = test['name'].apply(lambda x: 1 if "ハンバーグ" in x \
                                    or 'かつ' in x\
                                    or 'カツ' in x\
                                    or '豚' in x\
                                    or 'ポーク' in x\
                                    or '回鍋肉' in x\
                                    or 'ロース' in x\
                                    or 'ロコモコ' in x\
                                    or 'チンジャオロース' in x\
                                    or '青椒肉絲' in x\
                                    or 'ベーコン' in x\
                                    or '八宝菜' in x\
                                    else 0)

test['beef'] = test['name'].apply(lambda x: 1 if "ハンバーグ" in x \
                                    or 'メンチ' in x\
                                    or '肉じゃが' in x\
                                    or '牛' in x\
                                    or 'ビーフ' in x\
                                    or 'ロコモコ' in x\
                                    else 0)

test['chicken'] = test['name'].apply(lambda x: 1 if "鶏" in x \
                                    or 'チキン' in x\
                                    or '親子' in x\
                                    else 0)

test['fish'] = test['name'].apply(lambda x: 1 if "魚" in x \
                                    or 'サバ' in x\
                                    or 'ます' in x\
                                    or 'サーモン' in x\
                                    or 'アジ' in x\
                                    or 'キス' in x\
                                    or 'かじき' in x\
                                    or 'さわら' in x\
                                    or 'カレイ' in x\
                                    else 0)

test['shrimp'] = test['name'].apply(lambda x: 1 if "海老" in x else 0)

test['otherSeaFood'] = test['name'].apply(lambda x: 1 if "イカ" in x\
                                            or "カキ" in x\
                                            or "ホタテ" in x else 0)

test['vegetable'] = test['name'].apply(lambda x: 1 if "野菜" in x\
                                            or "カキ" in x\
                                            or "ホタテ" in x else 0)
    
# 欠損値の補完
test['kcal'] = test['kcal'].fillna(test['kcal'].mean()) # 平均値で補完
test['payday'] = test['payday'].fillna(0)

# 月平均との差分の特徴量「temp」を追加
test['month'] = test['datetime'].apply(lambda x : int(x.split("-")[1]))
test_temp_mean = test.groupby('month').temperature.mean()
test['month'] =  test['month'].replace(test_temp_mean)
test['temp'] = test['temperature'] - test['month']

# 説明変数と目的変数の設定
X_test = test[['popular',
                 'unpopular',
                 'fun',
                 'temperature',
                 'week',
                 'day',
                 'temp',
                 'kcal',
                 'month',
                 'curry',
                 'weather',
                 'otherSeaFood',
                 'chicken',
                 'pork',
                 'beef'
                 ]]

# テストデータにおける予測
preds = []

for model in models:
    pred = model.predict(X_test)
    preds.append(pred)
    
# predsの平均を計算
preds_array = np.array(preds)
preds_mean = np.mean(preds_array, axis = 0)

test['y'] = preds_mean
test['index_new'] = test.index + train.index.max() + 1
xs = test['index_new'].values
y_reg = a1 * np.exp(-b1 * xs) + c1

test['y'] = test['y'] + y_reg

'''
提出
'''

# 提出用サンプルの読み込み
sub = pd.read_csv('./data/sample.csv', header=None)

# 'SalePrice'の値を置き換え
sub[1] = test['y']

# CSVファイルの出力
sub.to_csv('./submit/bento_LightGBM.csv', header=None, index=False)