import requests
from bs4 import BeautifulSoup
import string
import pickle
import json, urllib, urllib2

def get_review_info(attraction_names,attraction_urls,filepath,filename):
    '''
    Scrap the user item matrix in the format of a list of dictionaries, each have attributes
    of user names, item names and ratings. Then save to jason file, which can be
    easily read in pandas
    INPUT:
        attraction names and urls from previous scrapping
        filepath and filename save in local
    OUTPUT:
        None
        save the user-item matrix in local
    '''
    review = {}
    review_list =[]
    baseurl = 'http://www.tripadvisor.com'
    for i in xrange(0,len(attraction_urls)):
    # for i, url in enumerate(attraction_urls):
        html = requests.get(baseurl + attraction_urls[i])
        soup = BeautifulSoup(html.text, "html.parser")
        try:
            total_reviewpages = soup.select('div.pageNumbers > a')[-1].text
        except:
            total_reviewpages = 1
        total_reviewpages = int(total_reviewpages)
        # if total_reviewpages > 40:
        #     total_reviewpages = 40
        # else:
        #     total_reviewpages = total_reviewpages
        # print 'total_reviewpages is ', total_reviewpages
        try:
            for num in xrange(total_reviewpages):
                urlr = baseurl + attraction_urls[i].split('Reviews')[0]+'reviews-or'+str(num*10)
                htmlr = requests.get(urlr)
                soupr = BeautifulSoup(htmlr.text, "html.parser")

                ## scrap ratings
                ratings = soupr.select('div.rating.reviewItemInline > span.rate.sprite-rating_s.rating_s')
                rating_list = [a.find('img')['alt'][0] for a in ratings[1:]]

                ## scrap username
                usernames = soupr.select("#REVIEWS")
                username_list = [user.text.strip('\n') for user in usernames[0].select("div.username.mo")]
                for j, username in enumerate(username_list):
                    review['item_name'] = attraction_names[i]
                    review['user_name'] = username
                    review['rating'] = rating_list[j]
                    review_list.append(review.copy())
                # print review_list
            print 'review scrapped for ' + attraction_names[i]
            # print review_list

                ## scrap content of reviews
                # reviews = soup0.select('p.partial_entry')
                # review_list = [review.text.strip('\n') for review in reviews[1:]]
                # for review0 in review_list:
                #     review['reviews'] = review0

        except:
            print 'information for ' + attraction_names[i] + ' incomplete'

    with open(filepath+filename, 'w') as f:
        json.dump(review_list, f)

def multithreading_review_scraper(attraction_names,attraction_urls):
    '''
    INPUT:
        attraction_names
        attraction_urls
    OUTPUT:
        None
    '''
    threads_lst = []
    baseurl = 'http://www.tripadvisor.com'
    for i in xrange(0,len(attraction_urls)):
        t = threading.Thread(target=get_review_info, args=(i,))
        threads_lst.append(t)

    for t in threads_lst:
        t.start()
    for t in threads_lst:
        t.join()

if __name__ == '__main__':
    filepath = '/Users/weiansheng/Desktop/coding/TripNow/data/'
    with open(filepath+'sf_attractions.pickle') as f:
        attraction_names,attraction_urls = pickle.load(f)
    print 'attractions collected'
    # attraction_names = attraction_names[:10]
    # attraction_urls = attraction_urls[:10]
    filename = 'sf_reviews_info.json'
    get_review_info(attraction_names,attraction_urls,filepath,filename)
