'''
A data collection script for both Twitter, and Facebook posts posted
by news organizations.

FB scraper github: https://github.com/kevinzg/facebook-scraper
Twitter scraper github: https://github.com/bisguzar/twitter-scraper

Author: Nick Rabb
nicholas.rabb@tufts.edu
'''

from facebook_scraper import get_posts
import json
import pandas as pd

DATA_DIR = './news-data'

fb_page_names = {
  'fox': 'FoxNews',
  'dailykos': 'dailykos',
  'huffington_post': 'HuffPost',
  'new_york_times': 'nytimes',
  'cnn': 'cnn',
  'washington_post': 'washingtonpost',
  'the_blaze': 'theblaze',
  'breibart': 'breitbart'
}

def get_fb_posts_from_march_onward(page):
  p = 250
  f = open(f'./data/fb-posts/{page}.json', 'w')
  f.write('[')
  for post in get_posts(page, pages=p):
    post['time'] = str(post['time'])
    f.write(json.dumps(post) + ',\n')
  f.write(']')
  f.close()

def collect_fb_data():
  for key in fb_page_names:
    print(f'Fetching data for {key}...')
    get_fb_posts_from_march_onward(fb_page_names[key])

def json_posts_to_df(filepath):
  return pd.read_json(filepath)

'''
PANDAS FUNCTIONS
'''

def rows_containing_covid(df):
  return df[df['post_text'].str.contains('COVID|coronavirus|pandemic|virus')]
