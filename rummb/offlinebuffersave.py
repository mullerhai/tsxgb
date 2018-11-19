#from modeldb.thrift.modeldb.ModelDBService import Client
# from ..modeldb.xgb_native.XGBModelDBSyncer import *
from  modeldb.xgb_native.XGBModelDBSyncer import *
import  logging

logger=logging.getLogger("xgboost_modeldb")


from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

# read in the iris data
iris = load_iris()

X = iris.data
y = iris.target
path='/home/muller/Documents/modeldb-client/src/main/scala/cod-rna.txt'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)

name = "xgboost3"
author = "muller"
description = "predicting iris"
#host='localhost'
host='10.201.8.9'
#host='192.168.199.102'
import  pymysql
#hostz='10.201.35.123'

buffer_list=list()

#syncer_obj=Syncer.init_sync_enable()
#NewExperimentRun("erer",sha="insert id "),
#ExistingExperimentRun(59),


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

collect='modeldb_metadata'
mongo_cli = MongoClient(host,27017)

model=xgb.train_sync(self=xgb,params=params,X_train=X_train,y_train=y_train,num_boost_round=200,buffer_list=buffer_list)


Dtest=xgb.DMatrix(X_test)

pred= model.predict_sync(X_test,buffer_list=buffer_list)


from sklearn.metrics import accuracy_score
acc_score=accuracy_score(y_test,pred)
print(acc_score)
# result = y_test.reshape(1, -1) == pred
# print('the accuracy:\t', float(np.sum(result)) / len(pred))

model.accuracy_sync(y_test,pred,X_train,buffer_list=buffer_list)
fsz=model.get_fscore()
print(fsz)
savekey=model.save_model_sync(mongo_cli)

print(savekey)
hostz='192.168.199.102'
# user='rms_plus_'
# pwd='hTTkOzQ3tmBlNd8rK'
port=3306
#user='rms_plus_'
#pwd='hTTkOzQ3tmBlNd8rK'
# user='muller'
# pwd='7104'
#
# db='modeldb_test'
# dbz=pymysql.connect(host=hostz,user=user,passwd=pwd,db=db,port=port,charset='utf8')

syncer_obj = Syncer(
    NewOrExistingProject(name, author, description),
    DefaultExperiment(),NewExperimentRun("kperfect",sha=savekey),
    ThriftConfig(host=host))
# syncer_obj.sync(save_key=savekey,sql_cli=dbz)

syncer_obj.sync(buffer_list=buffer_list)

# cli = syncer_obj.client
# cliz = Client(cli._iprot)
# print(cliz)

# syncer_obj.set_experiment_run()
# cliz.getProjectOverviews()

# eventlist = syncer_obj.buffer_list
# for event in eventlist:
#     #print(vars(event))
#     if isinstance(event, MetricEvent):
#         # eve = FitEvent(event)
#         print(event.model)
#
#     elif isinstance(event,FitEvent):
#         print(event.model)

        # print(event.model.filepath)
        # event.model.filepath ="hello world"
# prescor=compute_roc_auc_sync(model,test_y= y_test,y_pred=pred ,df= X_train)
# print(prescor)


# new_mo=xgb.load_model_sync(mongo_cli,'5bee664d87c5f627186b020b')
#
# predz= new_mo.predict_sync(X_test)


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



