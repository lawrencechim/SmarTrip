from collections import Counter
from pymongo import MongoClient
from bs4 import BeautifulSoup
import requests
import os

# dic1 = {'e':1, 'f':2}
# dic2 = {'g':1, 'h':2}
# predict =[dic1,dic2]
def test_to_mongodb():
    url = 'http://localhost:5353/check'
    r = requests.get(url)
    test_data_raw = r#.json()
    db = MongoClient()['fraud_test']
    collection = db['test_data']
    #for case in predict:
    db.test_data.insert_one(test_data_raw)
    cursor = db.test_data.find()
    return cursor.count()

def predict_to_mongodb(predict):
    db = MongoClient()['fraud_test']
    collection = db['test_collection']
    for case in predict:
        db.test_collection.insert_one(case)
    cursor = db.test_collection.find()
    return cursor.count()

if __name__ == '__main__':
    print test_to_mongodb()
    #print predict_to_mongodb(predict)
