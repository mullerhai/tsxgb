from modeldb.xgb_native.XGBModelDBSyncer import *
import  logging

from sklearn.metrics import *

logger=logging.getLogger("xgboost_modeldb")
from pymongo import MongoClient


train_path='./train_data.csv'
test_path='./test_data.csv'
test_y_path='./test_y.csv'
train_data=pd.read_csv(train_path,header=0)
test_x=pd.read_csv(test_path,header=0)
test_data_y=pd.read_csv(test_y_path,header=0)
tmp_train_data=train_data
train_x=tmp_train_data.drop( labels='SalePrice',axis=1)
train_y=train_data['SalePrice']

test_y=test_data_y['SalePrice']

params={
    'booster':'gbtree',#有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。缺省值为gbtree
	'objective': 'reg:linear',#定义学习任务及相应的学习目标，可选的目标函数如下：reg:linear” –线性回归；“reg:logistic” –逻辑回归；“binary:logistic” –二分类的逻辑回归问题，输出为概率；
	#'eval_metric': 'auc',#校验数据所需要的评价指标，不同的目标函数将会有缺省的评价指标；
	'max_depth':4,  # 树的最大深度;取值范围为：[1,∞];默认6;树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合;建议通过交叉验证（xgb.cv ) 进行调参;通常取值：3-10;
	'lambda':10,#L2 正则的惩罚系数，用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合；默认为0；
	'subsample':0.75,#用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。取值范围为：(0,1]，默认为1；
	'colsample_bytree':0.75,
	'min_child_weight':2, #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。取值范围为: [0,∞],默认1；
	'eta': 0.025,# 一个防止过拟合的参数，默认0.3;取值范围为：[0,1];通常最后设置eta为0.01~0.2;
	'seed':0,
	'nthread':8,#XGBoost运行时的线程数。缺省值是当前系统可以获得的最大线程数;如果你希望以最大速度运行，建议不设置这个参数，模型将自动获得最大线程
    'silent':1 # 打印信息的繁简指标，1表示简， 0表示繁; 建议取0，过程中的输出数据有助于理解模型以及调参。另外实际上我设置其为1也通常无法缄默运行。
      }

dtrain=xgb.DMatrix(train_x,train_y)

name = "xgb_pre_price"
author = "muller"
description = "predicting house "
#host='localhost'
host='10.201.8.9'
collect='modeldb_metadata'
mongo_cli = MongoClient(host,27017)
syncer_obj = Syncer(
    NewOrExistingProject(name, author, description),
    DefaultExperiment(),
    NewExperimentRun("hah"),ThriftConfig(host=host))

#model=xgb.train(params,dtrain,num_boost_round=200)

model=xgb.train_sync(self=xgb,params=params,X_train=train_x,y_train=train_y,num_boost_round=200)


dtest=xgb.DMatrix(test_x)
#pred=model.predict(dtest)

pred= model.predict_sync(test_x)
print(pred)
# self,test_y,y_pred,df

score=mean_absolute_error(test_y,pred)
print(score)
print("sync")

ros=model.mse_sync(test_y=test_y,y_pred=pred,df=train_x )
print(ros)
print(syncer_obj.buffer_list)

import  pymysql
host='10.201.35.123'
port=3306
user='rms_plus_'
pwd='hTTkOzQ3tmBlNd8rK'
db='modeldb_test'
dbz=pymysql.connect(host=host,user=user,passwd=pwd,db=db,port=port,charset='utf8')

savekey=model.save_model_sync(mongo_cli)
print(savekey)
syncer_obj.sync(save_key=savekey,sql_cli=dbz)
# syncer_obj.sync()
#print(roc_auc_score(test_y,pred))