import pandas as pd
import random
import urllib
import threading
from pymongo import MongoClient
from bs4 import BeautifulSoup
import requests
from urllib2 import urlopen
import os
import re
import json
import pickle

def get_attraction_links(base_link, num_pages):
    '''
    INPUT:
        base_link: string
            base URL link for attractions/things to do in a city
        num_pages: integer
            number of pages of attractions in a city
    OUTPUT:
        attraction_names: list of things to do in the city
        attraction_urls: URL links to details and reviews for that attraction
    '''
    attraction_names = []
    attraction_urls = []
    attractions = {}
    attractions_list = []
    for page_number in xrange (0,num_pages):
        html = requests.get(base_link+'-oa'+str(30*page_number))
        soup = BeautifulSoup(html.text, "html.parser")
        for tag in soup.select('div.property_title > a'):
            name = tag.text.split(' (')[0]
            link = tag.get('href')
            attraction_names.append(name)
            attraction_urls.append(link)
            attractions['name'] = name
            attractions['url'] = link
            attractions_list.append(attractions.copy())
    return attraction_names, attraction_urls, attractions_list

def save_json(filepath,attractions):
    with open(filepath+'sf_attractions.json', 'w') as f:
        json.dump(attractions, f)

def save_pickle (filepath,attraction_names,attraction_urls):
    with open(filepath+'sf_attractions.pickle', 'w') as f:
        pickle.dump([attraction_names,attraction_urls], f)

if __name__ == '__main__':
    url_sf = 'http://www.tripadvisor.com/Attractions-g60713-Activities' # San Francisco
    url_sea = 'http://www.tripadvisor.com/Attractions-g60878-Activities' # Seattle
    url_pdx = 'http://www.tripadvisor.com/Attractions-g52024-Activities' # Portland
    url_sd = 'http://www.tripadvisor.com/Attractions-g60750-Activities' # San Diego
    url_la = 'http://www.tripadvisor.com/Attractions-g32655-Activities' # Los Angels
    attraction_names, attraction_urls, attractions_list = get_attraction_links(url_sf,18)
    filepath = '/Users/weiansheng/Desktop/coding/TripNow/'
    save_json(filepath,attractions_list)
    save_pickle (filepath,attraction_names,attraction_urls)
