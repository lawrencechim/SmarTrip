import pandas as pd
import numpy as np
import graphlab
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from topic_modeling import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def read_clean(filepath):
    '''
    load data into pandas and drop duplicates
    input: filepath
    output: pandas dataframe
    '''
    df = pd.read_json(filepath)
    df.sort_index(inplace = True)
    df = df.drop_duplicates()
    return df

def recommender_factorization(df):
    '''
    build recommender with graphlab using matrix factorization recommender
    input: pandas dataframe
    output: recommender model
    '''
    sf = graphlab.SFrame(df)
    rec = graphlab.recommender.factorization_recommender.create(
            sf,
            user_id='user_name',
            item_id='item_name',
            target='rating',
            solver='als',
            side_data_factorization=False)
    return rec

def recommender_similarity(df):
    '''
    build recommender with graphlab using matrix factorization recommender
    input: pandas dataframe
    output: recommender model
    '''
    sf = graphlab.SFrame(df)
    rec = graphlab.item_similarity_recommender.create(
            sf,
            user_id="user_name",
            item_id="item_name",
            target="rating")
    return rec

def item_matrix_creator(filepath, filepath_item):
    '''
    input: filepath of review data, filepath of item data
    output: sframe of item data correspoinding to the same item_name in review data
    '''
    df = read_clean(filepath)
    df_item = pd.read_json(filepath_item)
    df_item2= df_item[np.isfinite(df_item['review_count'])][['name','description']]
    namelist_review = df.item_name.unique()
    namelist = df_item2.name.values
    names = non_exist_name_helper(namelist,namelist_review)
    df_item3 = df_item2.drop([308,395,398])
    X = df_item3.description.apply(lambda x: tokenize(x))
    tfidf=TfidfVectorizer('content', max_features=100, stop_words='english')
    matrix = tfidf.fit_transform(X).toarray()
    df_item4 = pd.DataFrame(matrix)
    df_item4['item_name'] = df_item3['name'].values
    sf_item = graphlab.SFrame(df_item4)
    return sf_item

def non_exist_name_helper(namelist,namelist_review):
    '''
    helper function find the non-exist item name in the review data
    '''
    names = []
    for name in namelist:
        if name not in namelist_review:
            names.append(name)
    return names

def content_boosted_recommender(df, sf_item):
    '''
    input: dataframe of review data, sframe of item data
    output: recommender model
    '''
    sf = graphlab.SFrame(df)
    rec_combine = graphlab.recommender.factorization_recommender.create(
            sf,
            user_id='user_name',
            item_id='item_name',
            target='rating',
            item_data = sf_item,
            solver='als',
            side_data_factorization=False)
    return rec_combine

def evaluate(rec):
    '''
    input: recommender model
    output: rmse of the model
    '''
    rmse = rec.evaluate_rmse(sf, target="rating")
    return rmse

def recommend(username, rec):
    '''
    input: username and recommender model
    output: top 10 recommendations
    '''
    return rec.recommend(users=[username], k=10)

def main():
    '''
    run the flow of recommender function
    user input their username at tripadvisor,
    print out top 10 things to do at San Francisco
    '''
    filepath = '/Users/weiansheng/Desktop/coding/TripNow/data/sf_reviews_info.json'
    filepath_item = '/Users/weiansheng/Desktop/coding/TripNow/data/sf_attractions_info_final.json'
    df = read_clean(filepath)
    sf_item = item_matrix_creator(filepath, filepath_item)
    rec = content_boosted_recommender(df, sf_item)
    username = raw_input("Please enter your username: ")
    print recommend(username, rec)

if __name__ == '__main__':
    main()
