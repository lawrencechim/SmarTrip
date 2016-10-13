# SmarTrip
2-week capstone project at Galvanize

## Overview
No matter when we visit a new city or we spend our weekends at our current city, we want to find the best attractions or 'Things to Do' which are fun and fit our interests. SmarTrip will guide you and recommend you suggestions based on your interest and its contents. TripNow categorizes all the attractions in San Francisco into categories using topic modeling, it also combines the distance data of different locations.

## Data Source
All attractions are scrapped from TripAdvisor, description and address, also users, reviews and ratings for each attractions are scrapped from TripAdvisor. Locations and Distances are obtained from Google maps with the address data from TripAdvisor.

## Data Pipeline and Modeling
description

## Tools and Packages used

#### Stack:

* python
* git
* markdown

#### Web Scraping:

* requests
* beautiful soup
* json
* pickle

#### Databasing:

* mongodb
* pymongo
* AWS EC2 & S3

#### Feature Engineering:

* NLTK
* regex

#### Modeling:

* numpy
* pandas
* scikit learn
* graphlab

#### Data visualization:

* matplotlib
* seaborn
* plotly
* D3

#### Web App:

* flask
* bootstrap
* html
* css
* JavaScript


## Future work
* Graph attractions with nodes as attractions and edges as their similarity and scores
* Expand the app to other cities in the US with same data from TripAdvisor
