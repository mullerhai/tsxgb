import  numpy as np
import  pandas as pd
import  xgboost as  xgb
from modeldb.thrift.modeldb.ModelDBService import Client
from modeldb.basic.ModelDbSyncerBase import Syncer
from xgb_native.XGBModelDBSyncer import *
from xgb_native.XGBModelDBSyncer import *
from xgb_native.XGBSyncableMetrics import *
import  logging

logger=logging.getLogger("xgboost_modeldb")


from sklearn.datasets import load_iris

from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# read in the iris data
iris = load_iris()

X = iris.data
y = iris.target
path='/home/muller/Documents/modeldb-client/src/main/scala/cod-rna.txt'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)

name = "xgboost2"
author = "muller"
description = "predicting iris"
#host='localhost'
host='10.201.8.9'

#syncer_obj=Syncer.init_sync_enable()
#NewExperimentRun("erer",sha="insert id "),
#ExistingExperimentRun(59),
syncer_obj = Syncer(
    NewOrExistingProject(name, author, description),
    DefaultExperiment(),NewExperimentRun("kperfect",sha="igokeck"),
    ThriftConfig(host=host))

print(y_test)
Dtrain=xgb.DMatrix(X_train,y_train)
params = {
                'booster': 'gbtree',
                'objective': 'multi:softmax',
                'num_class': 6,
                    'gamma': 0.1,
                'max_depth': 6,
                'lambda': 2,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 3,
                'silent': 1,
                'eta': 0.1,
                'seed': 1000,
                'nthread': 4,
}
# model=xgb.train(params,Dtrain,num_boost_round=200)

from pymongo import MongoClient
import  gridfs
from bson.objectid import ObjectId
collect='modeldb_metadata'
mongo_cli = MongoClient(host,27017)

model=xgb.train_sync(self=xgb,params=params,X_train=X_train,y_train=y_train,num_boost_round=200)


Dtest=xgb.DMatrix(X_test)

pred= model.predict_sync(X_test)


from sklearn.metrics import accuracy_score
hah=accuracy_score(y_test,pred)
print(hah)
# result = y_test.reshape(1, -1) == pred
# print('the accuracy:\t', float(np.sum(result)) / len(pred))

model.accuracy_sync(y_test,pred,X_train)
fsz=model.get_fscore()
from  sklearn.metrics import *
# score=f1_score(y_test,pred)
# print(score)

# print(pred)
# model.auc_sync(X_test,pred,X_train)
print(fsz)

# for i in syncer_obj.buffer_list:
#     print(i.__dict__.items())

# print(syncer_obj.buffer_list)

# runs_exps=cliz.getRunsAndExperimentsInProject(14)
# for eid in  runs_exps.experimentRuns:
#     print(eid)
# exp_id=runs_exps.e
cli = syncer_obj.client
cliz = Client(cli._iprot)
print(cliz)
# syncer_obj.set_experiment_run()
# cliz.getProjectOverviews()

eventlist = syncer_obj.buffer_list
for event in eventlist:
    #print(vars(event))
    if isinstance(event, MetricEvent):
        # eve = FitEvent(event)
        print(event.model)
        event.model= "hello world"
    elif isinstance(event,FitEvent):
        print(event.model)
        event.model= "hello world"
        # print(event.model.filepath)
        # event.model.filepath ="hello world"
# prescor=compute_roc_auc_sync(model,test_y= y_test,y_pred=pred ,df= X_train)
# print(prescor)

# data_base =mongo_cli.get_database(collect)
# fs = gridfs.GridFS(data_base)
# import pickle
# print(vars(model))
# model_pkl_file= pickle.dumps(model)
# model_meta_primarykey=fs.put(model_pkl_file)
# print(model_meta_primarykey)

savekey=model.save_model_sync(mongo_cli)

# new_mo=xgb.load_model_sync(mongo_cli,'5bee664d87c5f627186b020b')
#
# predz= new_mo.predict_sync(X_test)


from sklearn.metrics import accuracy_score
# hah=accuracy_score(y_test,predz)
# print(hah)
# result = y_test.reshape(1, -1) == pred
# print('the accuracy:\t', float(np.sum(result)) / len(pred))

#new_mo.accuracy_sync(y_test,predz,X_train)
#fsz=new_mo.get_fscore()
# syncer_obj.experiment_run.sha="what data"
# er=NewExperimentRun("docker",'k8s')
# syncer_obj.set_experiment_run(er,1)
# print(syncer_obj.experiment_run.sha)
# syncer_obj = Syncer(
#     NewOrExistingProject(name, author, description),
#     DefaultExperiment(),
#     NewExperimentRun("erer",sha="insert id "),ThriftConfig(host=host))
# print(syncer_obj.experiment_run.sha)

import  pymysql
host='10.201.6.123'
port=3631
user='rms_plus_w'
pwd='hTTkOzQ3tmBlNd8rK'
db='modeldb_test'
dbz=pymysql.connect(host=host,user=user,passwd=pwd,db=db,port=port,charset='utf8')
print(savekey)
syncer_obj.sync(save_key=savekey,sql_cli=dbz)
#,host='10.201.6.123',port=3631,user='rms_plus_w',pwd='hTTkOzQ3tmBlNd8rK',db='modeldb_test'
# eventlist = syncer_obj.buffer_list
# for event in eventlist:
#     #print(vars(event))
#     if isinstance(event, MetricEvent):
#         # eve = FitEvent(event)
#         print(event.model)
#
#     elif isinstance(event,FitEvent):
#         print(event.model)
