import requests
from bs4 import BeautifulSoup
import string
import pickle
import json, urllib, urllib2

def get_attraction_info(attraction_names,attraction_urls,filepath,filename):
    '''
    Save all the attractions information for a city into jason file.
    Creat a list of dictionaries, each have attributes of attraction names,
    urls, rating, review counts, type, description and address, fill None
    if not available. Then save to jason file, which can be easily read in pandas
    INPUT:
        attraction names and urls from previous scrapping
        filepath and filename save in local
    OUTPUT:
        None
    '''
    attraction = {}
    attraction_list =[]
    baseurl = 'http://www.tripadvisor.com'
    for i, url in enumerate(attraction_urls):
        attraction['name'] = attraction_names[i]
        attraction['url'] = url
        html = requests.get(baseurl + url)
        soup = BeautifulSoup(html.text, "html.parser")
        try:
            rating = soup.select('div.rs.rating > span > img')[0]['content']
            attraction['rating'] = float(rating)
        except:
            attraction['rating'] = None
            print 'rating for ' + attraction_names[i] + ' not recorded'

        try:
            review_count = soup.select('div.rs.rating > a')[0].text.split(' ')[0]
            attraction['review_count'] = int(review_count.replace(',',''))
        except:
            attraction['review_count'] = None
            print 'review count for ' + attraction_names[i] + ' not recorded'

        try:
            category = soup.select('div.separator > div.detail')[0].text.strip('\n')
            attraction['category'] = category
        except:
            attraction['category'] = None
            print 'category for ' + attraction_names[i] + ' not recorded'

        ## try to get all the descriptions, seperate descriptions from recommendated hours
        # if soup.select('div.listing_details') != []:
        #     description = soup.select('div.listing_details')[0].find('p').text
        #     attraction['description'] = description
        # elif soup.select('div.details_wrapper') != []:
        #     try:
        #         description = soup.select('div.details_wrapper')[1].find('p').text
        #         attraction['description'] = description
        #     except:
        #         print 'description for ' + attraction_names[i] + ' not recorded'
        #         attraction['description'] = 'Category for ' + attraction_names[i] +' is '+ attraction['category']
        # else:
        #     print 'description for ' + attraction_names[i] + ' not recorded'
        #     attraction['description'] = 'Category for ' + attraction_names[i] +' is '+ attraction['category']

        try:
            description = soup.select('div.listing_details')[0].find('p').text
            if attraction['category'] != None:
                attraction['description'] = attraction_names[i] +' is '+ attraction['category'] + ' ' + description
            else:
                attraction['description'] = attraction_names[i] + ' ' + description
        except:
            if attraction['category'] != None:
                attraction['description'] = attraction_names[i] +' is '+ attraction['category']
            else:
                attraction['description'] = attraction_names[i]
            print 'description for ' + attraction_names[i] + ' not recorded'

        try:
            address = soup.find('span',{'property':'address'}).text.split(': ')[-1].strip('\n ')
            attraction['address'] = address
        except:
            attraction['address'] = None
            print 'address for ' + attraction_names[i] + ' not recorded'

        # get no of english reviews
        try:
            review_english = soup.select('div.col.language.extraWidth')[0].findAll('label')[1].text.split('\n')[1]
            attraction['review_english'] = int(review_english.replace('(', '').replace(')', '').replace(' ','').replace(',',''))
        except:
            attraction['review_english'] = None
            #print 'rating for ' + attraction_names[i] + ' not recorded'

        attraction_list.append(attraction.copy())

    with open(filepath+filename, 'w') as f:
        json.dump(attraction_list, f)

if __name__ == '__main__':
    filepath = '/Users/weiansheng/Desktop/coding/TripNow/data/'
    with open(filepath+'sf_attractions.pickle') as f:
        attraction_names,attraction_urls = pickle.load(f)
    print 'attractions collected'
    # attraction_names = attraction_names[:30]
    # attraction_urls = attraction_urls[:30]
    filename = 'sf_attractions_info_final.json'
    get_attraction_info(attraction_names,attraction_urls,filepath,filename)
