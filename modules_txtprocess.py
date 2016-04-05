from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csr import csr_matrix
import re


# 分词前的预处理，包括以下几个部分
# 1、去除网页标签
# 2、分句
# 3、分词
# 4、词性标注
# 5、取词根(因为取词根会破坏原词的意义，对结果影响好坏难以确定，在此不做处理)
# 6、词形还原
# 7、拼接还原
# 输入为string，返回还是string
def proprocess(content):
    dr=re.compile(r'<[^>]+>',re.S)
    content2=dr.sub("",content)

    sentences=sent_tokenize(content2)
    sentstokens=[]
    for sent in sentences:
        sentstokens.append(word_tokenize(sent))

    sentstokenstag=[]
    for senttokens in sentstokens:
        sentstokenstag.append(pos_tag(senttokens))

    lemmatizer=WordNetLemmatizer()
    def lemmatize(token,tag):
        if tag[0].lower() in ['n','v']:
            return lemmatizer.lemmatize(token,tag[0].lower())
        return token

    newsents=[]
    for senttag in sentstokenstag:
        sent=[lemmatize(token,tag) for token,tag in senttag]
        newsents.append(sent)

    newcontent=''
    for sent in newsents:
        newcontent+=' '.join(sent)

    return newcontent

# 训练文档向量（基于词频），输入为文档集list对象，每个元素为一篇文档的字符串表示，输出为tf-idf文档向量
# TfidfVectorizer参数如下：
# TfidfVectorizer(self, 
#     input='content', 
#     encoding='utf-8', 编码方式，默认为utf-8
#     decode_error='strict', 解码错误处理方式{'strict', 'ignore', 'replace'}
#     strip_accents=None, 在preprocessing阶段移除（强调？），{'ascii', 'unicode', None}，默认为None
#     lowercase=True, 小写
#     preprocessor=None, 预处理函数
#     tokenizer=None, 分词函数
#     analyzer='word',分析单元 {'word', 'char'} or callable
#     stop_words=None, 停用词string {'english'}, list, or None (default)
#     token_pattern='(?u)\\b\\w\\w+\\b', 默认分词正则表达式
#     ngram_range=(1, 1), n元语法建模，（1，2）为1元和2元语法模型
#     max_df=1.0, 单词文档频率最大值（最大数，当为整数时），当指定词典时，该参数无效
#     min_df=1, 单词文档频率最小值（最小数，当为整数时），当指定词典时，该参数无效
#     max_features=None, 最大特征数（基于单词文档频率排序）
#     vocabulary=None, 指定词典
#     binary=False, 单词有无用0或1表示
#     dtype=<class 'numpy.int64'>, 
#     norm='l2', 归一化，'l1', 'l2' or None, optional，1范数或2范数归一化，默认2范数
#     use_idf=True, 是否用idf
#     smooth_idf=True, 是否平滑（加一法）
#     sublinear_tf=False) Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)
def TrainDocVec(doclist):
    tfidfvectorizer=TfidfVectorizer(stop_words="english",max_df=0.5,min_df=10)
    tfidfvec=tfidfvectorizer.fit_transform(doclist)
    return tfidfvec

