import tfidf, os
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

#构建语料
def build_corpus(news_path):
    corpus = []
    files = []

    #不同类别新闻各选50支
    for foldernames in os.listdir(news_path):
        for filenames in os.listdir(os.path.join(news_path, foldernames))[:50]:
            print(f'正在获取 {foldernames} 新闻：{filenames}')
            files.append(f'{foldernames} {filenames}')
            with open(os.path.join(news_path, foldernames, filenames), 'r', encoding='utf-8-sig') as f:
                seg_list = []
                for line in f.readlines():
                    #分词，去除\u3000,空格,\n等
                    seg_list += list(jieba.cut(line.replace(u'\u3000',u'').replace('\n', '').replace('\r', '').replace(" ","")))
                corpus.append(" ".join(seg_list))
    
    return corpus, files

def tf_idf(corpus, files):
    print("开始生成tf-idf...")
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names()
    weight = pd.DataFrame(tfidf.toarray(), columns=words)
    weight.insert(0, "filenames", files)
    weight.to_csv("weight.csv", index=False, encoding='utf-8-sig')
    return words, weight

if __name__ == "__main__":
    df = pd.read_csv("senti_2_tfidf.csv", encoding='utf-8-sig')
    data_features = df.iloc[:,1:]
    data_targets = df.iloc[:,0]
    x_train,x_test,y_train,y_test = train_test_split(data_features,data_targets,test_size=0.1)

    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    accuracy = classifier.score(x_test, y_test)
    print(accuracy)