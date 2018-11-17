import  fasttext
import jieba
from gensim.models import word2vec
import logging
novel_seg_path="./wingman_segment.txt"
novel_wzv_path='./w2v.txt'
model_savepath='./news_fast.model'
load_model='./news_fast.model.bin'
test_path='./test_wing_seg.txt'
def novel_segment():
    novel=open("wingman.txt")
    content=novel.read()
    novel_segmented=open("./wingman_segment.txt",'w')

    cutword=jieba.cut(content,cut_all=False)
    seg = ' '.join(cutword).replace(',','').replace('。','').replace('“','').replace('”','').replace('：','').replace('…','')\
                .replace('！','').replace('？','').replace('~','').replace('（','').replace('）','').replace('、','').replace('；','')
    print(seg,file=novel_segmented)

    novel.close()
    novel_segmented.close()

def generate_word2vec():
    s=word2vec.LineSentence(novel_seg_path)
    model = word2vec.Word2Vec(s, size=20, window=5, min_count=5, workers=4)
    model.save(novel_wzv_path)
    return model



def fasttext_etl(dataset,model_savepath):
    print("fast")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
    classifier=fasttext.supervised(dataset,model_savepath,label_prefix='__label__')
    return classifier

def fasttext_predict(model_filepath,test_dataset):
    classifier=fasttext.load_model(model_filepath,label_prefix='__label__')
    result=classifier.test(test_dataset)
    print(result)
    return result

if __name__ == '__main__':
   # model=  generate_word2vec()
#    # print(vars(model))
#    # print(model['杨过'])
#     model=fasttext_etl(novel_seg_path,model_savepath)
#     result=model.test(test_path)
    result=fasttext_predict(load_model,test_path)
    print(result.precision)
    print(result.recall)
