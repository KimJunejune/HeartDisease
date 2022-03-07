# -------------------------------
# 使用随机森林进行二分类
# 5days思想是 将当天即后五天相加，作为目标值
# 再根据四分位数0，1划分 进行二分类
# 目前效果最好
# --------------------------------
import os
import pickle
import pandas as pd
import numpy as np
import graphviz

from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

acs = pd.read_csv(os.path.join(
    "../data", 'mergedNewAcs.csv'
))
acs['time'] = pd.to_datetime(acs['time'])
# ------------------------
# 五天counts相加
# ------------------------
for i in range(acs.shape[0] - 4):
    acs.iloc[i, -1] += acs.iloc[i+1, -1]
    acs.iloc[i, -1] += acs.iloc[i+2, -1]
    acs.iloc[i, -1] += acs.iloc[i+3, -1]
    acs.iloc[i, -1] += acs.iloc[i+4, -1]
acs = acs.iloc[:-4, :]

acs = acs[acs['time'] >= pd.to_datetime('2017/01/01')]
acs = acs[(acs['time'] < pd.to_datetime('2020/01/01')) | (acs['time'] > pd.to_datetime('2020/09/01'))]
# -------------------------
# 四分位数：8， 33， 39， 51， 100
# -------------------------
p_25, p_75 = acs['counts'].quantile([0.25, 0.75])

def filt(x, p25, p75):
    if x<=p25:
        return 0
    elif x>=p75:
        return 1
    else:
        return 2
acs['counts'] = acs.counts.apply(lambda x: filt(x, p_25, p_75))
acs = acs[(acs['counts'] == 1) | (acs['counts'] == 0)]
acs = acs.drop(['time'], axis = 1)

# -------------------------
# 定义评价函数
# -------------------------
def eval(clf, X_train, X_test, y_train, y_test):
    predicted = clf.predict(X_train)
    accu = accuracy_score(y_train, predicted)
    print("训练集准确率：", accu)
    f1 = f1_score(y_train, predicted, average="macro")
    print("训练集f1：", f1)

    predicted = clf.predict(X_test)
    accu = accuracy_score(y_test, predicted)
    print("测试集准确率：", accu)
    f1 = f1_score(y_test, predicted, average="macro")
    print("测试集f1：", f1)

    return accu

# -------------------------
# 切分数据集 进行训练
# -------------------------
train_x, test_x, train_y, test_y = train_test_split(acs.iloc[:, :-1], acs.iloc[:, -1], shuffle=True, test_size=0.3)

# 随机森林
params = {'n_estimators':[40,60,70],'max_depth':[4,5,6,7],
          'criterion':['entropy'],"class_weight":[ 'balanced'],"random_state":[1]}

clf = GridSearchCV(estimator=RandomForestClassifier(),param_grid=params,cv = 5,n_jobs = -1,scoring="f1_macro")
clf.fit(train_x, train_y)  # 模型训练完毕
print("Best Params:{}".format(clf.best_params_))

rdf = RandomForestClassifier(n_estimators=clf.best_params_['n_estimators'], criterion="entropy", random_state =1, max_depth=clf.best_params_['max_depth'], class_weight="balanced")
rdf.fit(train_x, train_y)
eval(rdf, train_x, test_x,train_y,  test_y)

# -------------------------
# 保存随机森林的重要特征
# -------------------------
important_fea = list(rdf.feature_importances_)
fea = []
for idx in range(20):
    impor = max(important_fea)
    index = important_fea.index(impor)
    fea.append((index, impor))
    important_fea[index] = -999

selected_features = []
for i in range(len(fea)):
    print(
        '第', i+1, '个重要的元素是', train_x.columns[fea[i][0]], '重要性为', fea[i][1]
    )
    selected_features.append(train_x.columns[fea[i][0]])

# -------------------------
# 绘制随机森林
# -------------------------

Estimators = rdf.estimators_
clf = Estimators[0]
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("clf")


# ---------------------------
# 使用重要特征重新训练
# ---------------------------
train_x, test_x, train_y, test_y = train_test_split(acs.loc[:,selected_features], acs.iloc[:, -1], shuffle=True, test_size=0.3)
params = {'n_estimators':[50, 80, 100,150],'max_depth':[5,6,7],
          'criterion':['entropy'],"class_weight":[ 'balanced'],"random_state":[1]}

clf = GridSearchCV(estimator=RandomForestClassifier(),param_grid=params,cv = 5,n_jobs = -1,scoring="f1_macro")
clf.fit(train_x, train_y)  # 模型训练完毕
print("Best Params:{}".format(clf.best_params_))

rdf = RandomForestClassifier(n_estimators=clf.best_params_['n_estimators'], criterion="entropy", random_state =1, max_depth=clf.best_params_['max_depth'], class_weight="balanced")
rdf.fit(train_x, train_y)
test_accu = eval(rdf, train_x, test_x,train_y,  test_y)
test_accu = int(test_accu * 10000)

# --------------------------
# 保存模型
# --------------------------
model_name = 'myModel/RF_{}.pkl'.format(test_accu)
with open(os.path.join("../", model_name), 'wb') as f:
    pickle.dump(rdf, f)