from sklearn.datasets import load_files
from modules_txtprocess import proprocess



mycorpus=load_files("E:\\train")
# 未去停用词时countvec维度是（25000，74849），去停用词后维度为（25000，74539）
# 再做词形还原处理后维度为（25000，68777）
for i in range(0,len(mycorpus.data)):
    mycorpus.data[i]=proprocess(mycorpus.data[i].decode())#data[i]数据类型是bytes，需解码为String
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer_count=CountVectorizer(stop_words="english")
# countvec=vectorizer_count.fit_transform(mycorpus.data)
# print(countvec.shape)
docvec=TrainDocVec(mycorpus.data)
# 
