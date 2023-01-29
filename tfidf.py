import jieba, os, csv, math, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

news_path = "D:\Shaw\Documents\Miscellany\网课\R计划数据挖掘方向\课程\Homework 1\THUCNews"

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

def cos_sim(l1, l2):
    return np.sum(l1*l2/(np.linalg.norm(l1) * np.linalg.norm(l2)))

def pca(df):
    print("开始进行PCA...")
    X_scaler = StandardScaler()
    x = X_scaler.fit_transform(df)

    pca_2d = PCA(0.9)
    new_x = pca_2d.fit_transform(x)
    new_df = pd.DataFrame(new_x)
    new_df.to_csv("PCA.csv", index=False, encoding='utf-8-sig')
    return new_df

if __name__ == "__main__":
    """ corpus, files = build_corpus(news_path)
    words, weight = tf_idf(corpus, files) """

    df = pd.read_csv("weight.csv", encoding='utf-8-sig')

    print(cos_sim(df.iloc[0,1:], df.iloc[100,1:]))
    print(cos_sim(df.iloc[0,1:], df.iloc[200,1:]))
    print(cos_sim(df.iloc[0,1:], df.iloc[500,1:]))

    print()

    #pca(df.iloc[:,1:])
    df_pca = pd.read_csv("PCA.csv", encoding="utf-8-sig")

    """ plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.xticks(np.arange(0,1,0.005))
    plt.xticks(np.arange(0,1,0.005))
    plt.scatter(df_pca.iloc[:10,0], df_pca.iloc[:10,1], marker = 'x',color = 'red', s = 40 ,label = '体育')
    plt.scatter(df_pca.iloc[10:20,0], df_pca.iloc[10:20,1], marker = '+', color = 'blue', s = 40, label = '娱乐')
    plt.scatter(df_pca.iloc[20:,0], df_pca.iloc[20:,1], marker = 'o', color = 'green', s = 40, label = '居家')
    #plt.legend(loc = 'best')    # 设置 图例所在的位置 使用推荐位置

    plt.show()   """
    print(cos_sim(df_pca.iloc[0,:], df_pca.iloc[100,:]))
    print(cos_sim(df_pca.iloc[0,:], df_pca.iloc[200,:]))
    print(cos_sim(df_pca.iloc[0,:], df_pca.iloc[500,:]))

    """ kmeans = KMeans(n_clusters=3).fit(df_pca)
    print(kmeans.labels_.reshape([3,100])) """

    