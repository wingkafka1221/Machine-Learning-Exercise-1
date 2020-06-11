#!/usr/bin/env python
# coding: utf-8

# In[1]:


#乳腺癌诊断分析
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

#导入数据
data = pd.read_csv('data.csv')

#数据探索
print(data.shape)
print(data.describe())
print(data.head())

#因为数据中不同的字段分别展示了数据的均值、标准差和极值，我们在此作区分，并在后决定使用均值；
features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])

#相关性分析,画出图像
plt.figure(figsize=(14,14))
corr = data[features_mean].corr()
sns.heatmap(corr,annot=True)

#删除相关性较大的字段，仅保留一个；
feature_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']

#数据清洗
#删除id字段
#将目标字段进行序数编码
#划分目标值和特征数据集
data.drop(['id'],axis=1,inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
y = data['diagnosis']
X = data[feature_remain]

#模型构建#模型预测
#定义模型
classifier = [
    SVC(random_state=1),
    DecisionTreeClassifier(random_state=1),
    RandomForestClassifier(random_state=1),
    LogisticRegression(random_state=1),
    KNeighborsClassifier()
]

#模型名称
classifier_name = [
    'svc',
    'decisiontreeclassifier',
    'randomforestclassifier',
    'logisticregression',
    'kneighborsclassifier'
]

#模型参数
classifier_param_grid = [
    {'svc__C':[0.5,1,1.5,2],'svc__kernel':['rbf','linear','poly']},
    {'decisiontreeclassifier__criterion':['gini','entropy'],'decisiontreeclassifier__max_depth':[2,4,6,8],'decisiontreeclassifier__min_samples_leaf':[1,2,4,6]},
    {'randomforestclassifier__n_estimators':[4,6,8,10,20,25,30,40,50],'randomforestclassifier__criterion':['gini','entropy'],'randomforestclassifier__max_depth':[2,4,6,8]},
    {'logisticregression__penalty':['l1','l2'],'logisticregression__solver':['sag','saga','lbfgs','liblinear'],'logisticregression__max_iter':[50,100,150,200]},
    {'kneighborsclassifier__n_neighbors':[2,4,6,8,10,12],'kneighborsclassifier__algorithm':['ball_tree','kd_tree','brute']}
]

#构建网格搜索模型，获取各模型的最佳参数和最优得分
def GridSearchCV_work(pipeline,X,y,param_grid,score='accuray'):
    gridsearch = GridSearchCV(cv = 10,estimator = pipeline,param_grid = param_grid,scoring=score)
    search = gridsearch.fit(X,y)
    print('最佳参数',search.best_params_)
    print('最优得分:%.4lf'%search.best_score_)

for model,model_name,model_param_grid in zip(classifier,classifier_name,classifier_param_grid):
    pipeline = Pipeline([
        ('scaler',StandardScaler()),
        (model_name,model)
    ])
    GridSearchCV_work(pipeline,X,y,model_param_grid,score='accuracy')

