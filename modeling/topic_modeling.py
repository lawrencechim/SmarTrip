import pandas as pd
import numpy as np
from collections import Counter

from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import linear_kernel
import lda

from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

def load_data(filepath):
    '''
    load data into pandas
    input: filepath
    output: pandas dataframe
    '''
    df = pd.read_json(filepath)
    return df

def tokenize(doc):
    '''
    Tokenize and stem/lemmatize the document.
    INPUT: string
    OUTPUT: list of strings
    '''
    tokenizer = RegexpTokenizer(r'\w+')
    doc1 = tokenizer.tokenize(doc.lower())
    stop = stopwords.words('english')
    stop.append('category')
    doc2 = [word for word in doc1 if word not in stop]
    snowball = SnowballStemmer('english')
    doc3 = [snowball.stem(word) for word in doc2]
    return ' '.join(doc2)

def df_tokenzie(df):
    '''
    Tokenize, remove stopwords and stem/lemmatize the document.
    INPUT: df
    OUTPUT: df with descriptions tokenized
    '''
    df['description'].apply(lambda x: tokenize(x))
    return df

def vectorize(df, n_features):
    '''
    INPUT: df
    OUTPUT: transformed matrix tfidf and tf, feature names
    '''
    X = df.description.values
    tfidf_vectoriizer = TfidfVectorizer('content', max_df=0.95, min_df=2,
                        max_features=n_features, stop_words='english')
    tfidf = tfidf_vectoriizer.fit_transform(X)

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                        max_features=n_features,stop_words='english')
    tf = tf_vectorizer.fit_transform(X)
    feature_names = tfidf_vectoriizer.get_feature_names()
    return tfidf, tf, feature_names

def kmeans(matrix, n, features): # every run is slightly different, result is good
                                #but not very consistent
    '''
    INPUT: tfidf, number of clusters
    OUTPUT: kmeans model, labels, print top topics for each cluster
    '''
    kmeans = KMeans(n_clusters=n).fit(matrix)
    label = kmeans.labels_
    top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
    print "top features for each cluster:"
    for num, centroid in enumerate(top_centroids):
        print "%d: %s" % (num, ", ".join(features[i] for i in centroid))
    return kmeans, label

def plot_similarity(documents):
    '''
    INPUT: documents, a list of text
    OUTPUT: cosine similarities plot
    '''
    vect = TfidfVectorizer(stop_words='english')
    X = vect.fit_transform(documents)
    cosine_similarities = linear_kernel(X, X)
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(cosine_similarities, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.savefig('cos_similarity')
    #plt.show()

def hierarch_clust(df):
    """
    INPUT: df
    OUTPUT:
    - dendrogram
    """
    # first vectorize
    tfidf=TfidfVectorizer('content', max_features=1000, stop_words='english', ngram_range=(1,1))
    matrix = tfidf.fit_transform(df.description.values)

    # now get distances
    distxy = squareform(pdist(matrix.todense(), metric='cosine'))
    distxy = np.nan_to_num(distxy) # convert nan values to zero

    # Pass this matrix into scipy's linkage function to compute our
    link = linkage(distxy, method='complete')

    # Using scipy's dendrogram function plot the linkages as a hierachical tree.
    plt.figure(figsize=(30,20))
    dendro = dendrogram(link, color_threshold=1.5, leaf_font_size=9,
               labels=df.name.values)
    # fix spacing to better view dendrogram and the labels
    plt.subplots_adjust(top=.99, bottom=0.5, left=0.05, right=0.99)
    plt.savefig('dendrogram.png')
    plt.show()


def lda_model(tf, n_topics): # scikit learn lda
    '''
    INPUT: tf matrix, number of topics
    OUTPUT: lda model and labels
    '''
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0).fit(tf)
    transform_lda = lda.transform(tf)
    label_lda = np.argmax(transform_lda, axis=1)
    print Counter(label_lda)
    return lda, label_lda

def lda_model2(tf, n_topics, n_top_words, feature_names): # python lda, better
    '''
    INPUT: tf matrix, number of topics, number of top words, feature names
    OUTPUT: lda model and labels, print the top words for each topic
    '''
    model_lda = lda.LDA(n_topics=n_topics, n_iter=1500, random_state=1)
    model_lda.fit(tf)  # model.fit_transform(X) is also available
    topic_word = model_lda.topic_word_  # model.components_ also works
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(feature_names)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    doc_topic = model_lda.doc_topic_ # (523, 12) transformed vectors
    label_lda = np.argmax(doc_topic, axis=1)
    print Counter(label_lda)
    return model_lda, label_lda

def nmf_model(tfidf, n_topics): # best model
    '''
    INPUT: tfidf matrix, number of topics
    OUTPUT: nmf model and labels
    '''
    nmf = NMF(n_components=n_topics, random_state=1,
      alpha=.1, l1_ratio=.5).fit(tfidf)
    transform_nmf = nmf.transform(tfidf)
    label_nmf = np.argmax(transform_nmf, axis=1)
    #print Counter(label_nmf)
    return nmf, label_nmf

def print_top_words(model, feature_names, n_top_words):
    '''
    INPUT: model, feature names, number of top words
    OUTPUT: print the top words for each topic
    '''
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

def choose_topics(df, label_nmf):
    '''
    INPUT: df, labels, let user choose from a list of topic numbers
    OUTPUT: print the top 10 recommended places to visit
    '''
    df['nmf_label'] = label_nmf
    topics = ['0 Landmarks and Views', '3 Nature and Parks', '5 Museums', '1 Shopping and Arts',
    '4 Tours', '2 Theaters and Concerts', '6 Religion', '8 Historical walking areas', '11 Food',
    '10 Convention centers', '7 Bars', '9 Churches']
    print 'Please choose from this list of topics: ', topics
    topic = raw_input("Please enter your topic number: ")
    label = int(topic)
    #label = int(topic.split(' ')[0])
    print 'Top 10 recommended places to visit are:'
    print df.name[df.nmf_label==label] [:10]

def main():
    filepath = '/Users/weiansheng/Desktop/coding/TripNow/data/sf_attractions_info_final.json'
    df = load_data(filepath)
    df = df_tokenzie(df)
    n_features = 1000
    n_topics = 12
    n_top_words = 10
    tfidf, tf, feature_names = vectorize(df, n_features)
    nmf, label_nmf = nmf_model(tfidf, n_topics)
    print_top_words(nmf, feature_names, n_top_words)
    choose_topics(df, label_nmf)

    # lda, label_nmf = lda_model(tfidf, n_topics)
    # print_top_words(lda, feature_names, n_top_words)

    # lda_model2(tf, n_topics, n_top_words, feature_names)

if __name__ == "__main__":
    main()
