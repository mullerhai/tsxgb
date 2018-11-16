from modeldb.thrift.modeldb.ModelDBService import Client
from modeldb.basic.ModelDbSyncerBase import Syncer
from xgb_native.XGBModelDBSyncer import *
from xgb_native.XGBSyncableMetrics import *
from pymongo import MongoClient
import  gridfs
from bson.objectid import ObjectId
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from xgb_native.modeldbQuery import *
name = "xgboost2"
author = "muller"
description = "predicting iris"
host='10.201.8.9'
syncer_obj = Syncer(
    NewOrExistingProject(name, author, description),
    DefaultExperiment(),
    NewExperimentRun("Abcz"),ThriftConfig(host=host))

print(syncer_obj.project)

cli = syncer_obj.client
# print(cli)
# print( type( cli._iprot))
# print(cli._iprot)


mongo_cli=MongoClient("localhost",27017)

modelQuery=ModeldbQuery(syncer_obj=syncer_obj,mongo_cli=mongo_cli)

mdoe=modelQuery.load_model_by_gridfsid_model(modelfile_id='5bee664d87c5f627186b020b')
print(mdoe)

modelQuery.save_modelfile_gridfs(mdoe)
prolist=modelQuery.query_all_projectlist()
for pro in prolist:
    print(vars(pro))

#exrun=modelQuery.query_modelList_byProjectId(14)

# mo=modelQuery.query_model_byExperimentRunId(155)
# print(mo)
#
# moz=modelQuery.query_model_hyperList(133)
#
# ids=modelQuery.query_model_creatime(27)
# print(ids)
# for mid in exrun:
#     print(vars(mid))

# transport = TSocket.TSocket('localhost', 6543)
# transport = TTransport.TBufferedTransport(transport)
# protocol = TBinaryProtocol.TBinaryProtocol(transport)
# transport.open()
# cliz= Client(protocol)

cliz = Client(cli._iprot)
#
# print(cliz)
# print(vars( cli._iprot))
status=cliz.testConnection()
print(status)
#
mo=cliz.getModel(1)
print(mo.sha)

modelQuery.query_gridfsId_bymodelId(3)
modelQuery.query_gridfsId_byExperimentRunId(2)
# print(mo.metrics)
# print(mo.filepath)
# print(mo.metadata)
# df=mo.trainingDataFrame
# print(df)

# runs_exps=cliz.getRunsAndExperimentsInProject(14)
# for eid in  runs_exps.experimentRuns:
#     print(eid)
# exp_id=runs_exps.experimentRuns[-1].id
# print(exp_id)
# model_res=cliz.getExperimentRunDetails(94).modelResponses
# print(model_res)


# pros=cliz.getProjectOverviews()
# for pro in pros:
#     print(pro.project)
#print(pros)