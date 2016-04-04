from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
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
