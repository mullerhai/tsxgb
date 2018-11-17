from modeldb.thrift.modeldb.ModelDBService import Client
from pymongo import MongoClient

from modeldb.xgb_native.modeldbQuery import *
name = "xgboost2"
author = "muller"
description = "predicting iris"
host='10.201.8.9'
mongo_cli=MongoClient(host,27017)

syncer_obj = Syncer(
    NewOrExistingProject(name, author, description),
    DefaultExperiment(),
    NewExperimentRun("Abcz"),ThriftConfig(host=host))

print(syncer_obj.project)

cli = syncer_obj.client
cliz = Client(cli._iprot)

status=cliz.testConnection()


modelQuery=ModeldbQuery(syncer_obj=syncer_obj,mongo_cli=mongo_cli)

def load_model_assert():
    mdoe=modelQuery.load_model_by_gridfsid_model(modelfile_id='5bee664d87c5f627186b020b')
    print(mdoe)

def save_model_assert(model_obj):
    modelQuery.save_modelfile_gridfs(model_obj)

def query_all_projects_assert():
    prolist=modelQuery.query_all_projectlist()
    for pro in prolist:
        print(vars(pro))

def query_model_byProId_assert(proId=14):
    exrun=modelQuery.query_modelList_byProjectId(proId)
    print(exrun)
def query_model_byExperimentRunId_assert(exId=155):
    mo=modelQuery.query_model_byExperimentRunId(exId)
    print(mo)

def get_modelRespone_assert(modedId=1):
    mo=cliz.getModel(modedId)
    print(mo.sha)

def getGridfsId_byModelId_assert(model_id=3):
    gridfsId=modelQuery.query_gridfsId_bymodelId(model_id)
    print(gridfsId)
def getGridfsId_byExRunId_assert(exrunId=2):
    modelQuery.query_gridfsId_byExperimentRunId(exrunId)


# print(cliz)
# print(vars( cli._iprot))
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


# print(status)
#

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