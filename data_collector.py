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
from bs4 import BeautifulSoup
import requests
from multiprocessing import Pool
import io
import numpy as np
import re
import math
from datetime import datetime

DATA_DIR = './news-data'

'''
FACEBOOK DATA
'''

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
MEDIA CLOUD DATA
'''

# Mappings from media name to media_id
NYT = 1
FOX = 1092
HUFF_POST = 623375
BREITBART = 19334

def scrape_body(url):
  headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 'Accept-Encoding': 'gzip, deflate, br', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'}
  body = requests.get(url, headers=headers)
  return body

def parseNYTText(body):
  soup = BeautifulSoup(body)
  text = ''
  article_class = 'css-1r7ky0e'
  paragraph_class = 'css-axufdj'
  # text += str(soup.find(attrs={'class': article_class}))
  for p in soup.find_all(attrs={'class': paragraph_class}):
    text += p.text + "\n"
  return text

def parseFoxText(body):
  soup = BeautifulSoup(body)
  text = ''
  body_class = 'article-body'
  # text += str(soup.find(attrs={'class': body_class}))
  for p in soup.find(attrs={'class': body_class}).find_all('p'):
    text += p.text + "\n"
  return text

def parseHuffPostText(body):
  soup = BeautifulSoup(body)
  text = ''
  par_class = 'content-list-component'
  # article_class = 'entry__text'
  # text += str(soup.find(attrs={'class': article_class}))
  for p in soup.find_all(attrs={'class': par_class}):
    text += p.text + "\n"
  return text

def parseBreitbartText(body):
  soup = BeautifulSoup(body)
  body_class = 'entry-content'
  text = ''
  for p in soup.find(attrs={'class': body_class}).find_all('p'):
    text += p.text + '\n'
  return text

text_parsing_functions = {
  NYT: parseNYTText,
  FOX: parseFoxText,
  HUFF_POST: parseHuffPostText,
  BREITBART: parseBreitbartText,
}

media_id_to_name = {
  NYT: 'New York Times',
  FOX: 'Fox News',
  HUFF_POST: 'Huffington Post - United States',
  BREITBART: 'Breitbart'
}

organizations_to_parse = [
  NYT, FOX, HUFF_POST, BREITBART
]

NUM_THREADS = 10

# NOTE: To read mediacloud df, the separator character is below:
MC_SEP = '\x1c'
# the command is pd.read_csv(path, sep=MC_SEP)

def load_article_parallel(row):
  print(f'Parsing url at row {row[0]}')
  try:
    row_data = row[1]
    parse_fn = text_parsing_functions[row_data.media_id]
    body = scrape_body(row_data.url)
    return parse_fn(body.text)
  except Exception as e:
    print(f'Error parsing article data for row {row[0]}: {e}')
    return ''

def load_mediacloud_df(path):
  df = pd.read_csv(path)
  valid_orgs_df = df[df['media_id'].isin(organizations_to_parse)]
  article_texts = []
  print(f'Fetching article data for {len(valid_orgs_df)} rows on {NUM_THREADS} threads...')
  with Pool(NUM_THREADS) as p:
    article_texts = p.map(load_article_parallel, valid_orgs_df.iterrows())
  # for row in valid_orgs_df.iterrows():
  #   print(f'Parsing url at row {row[0]}')
  #   try:
  #     row_data = row[1]
  #     parse_fn = text_parsing_functions[row_data.media_id]
  #     article_texts.append(parse_fn(row_data.url))
  #   except Exception as e:
  #     print(f'Error parsing article data for row {row[0]}: {e}')
  valid_orgs_df['article_data'] = article_texts
  return valid_orgs_df

def article_raw_text(row):
  print(f'Parsing url at row {row[0]}')
  try:
    row_data = row[1]
    parse_fn = text_parsing_functions[row_data.media_id]
    body = row_data.article_data
    return parse_fn(body)
  except Exception as e:
    print(f'Error parsing article data for row {row[0]}: {e}')
    return ''

def parse_mc_article_html(df):
  article_texts = []
  print(f'Fetching article data for {len(df)} rows on {NUM_THREADS} threads...')
  with Pool(NUM_THREADS) as p:
    article_texts = p.map(article_raw_text, df.iterrows())
  df['article_data_raw'] = article_texts
  return df

# This is from the post_accounts_seed.sql file (should be correct??)
POST_ACCOUNTS_IDS = {
  NYT: 6,
  FOX: 1,
  HUFF_POST: 2,
  BREITBART: 3
}

def write_mc_df_to_sql(df):
  files = { name: io.open(f'mysql/articles/{name}.sql', 'w', encoding='utf-8') for name in media_id_to_name.values() }
  for row in df.iterrows():
    print(f'Writing row {row[0]}')
    row_data = row[1]
    f = files[row_data.media_name]
    pars = get_keyword_paragraph(row_data.article_data_raw, 'mask', 2)
    for par in pars:
      f.write(f'INSERT INTO `articles_mask` (`post`,`native_id`,`post_account_id`,`post_type`) VALUES ("{par}","{row_data.stories_id}","{POST_ACCOUNTS_IDS[row_data.media_id]}","{2}");\n')
  for f in files.values():
    f.close()

def write_mc_df_to_sql_date_sample(df):
  files = { name: io.open(f'mysql/articles/{name}_sample.sql', 'w', encoding='utf-8') for name in media_id_to_name.values() }
  NUM_PER_MONTH = 10
  ids_written = []
  written_per_month = { month: 0 for month in range(1,13) }
  for row in df.iterrows():
    print(f'Checking row {row[0]}')
    dt_format = '%Y-%m-%d %H:%M:%S'
    row_data = row[1]
    if pd.isna(row_data.publish_date):
      continue
    dt_str = row_data.publish_date
    if ('.' in dt_str):
      dt_str = dt_str[:dt_str.index('.')]
    dt = datetime.strptime(dt_str, dt_format)
    if row[0] not in ids_written and written_per_month[dt.month] < NUM_PER_MONTH:
      print(f'Writing row {row[0]}')
      f = files[row_data.media_name]
      pars = get_keyword_paragraph(row_data.article_data_raw, 'mask', 2)
      for par in pars:
        f.write(f'INSERT INTO `articles_mask` (`post`,`native_id`,`post_account_id`,`post_type`) VALUES ("{par}","{row_data.stories_id}","{POST_ACCOUNTS_IDS[row_data.media_id]}","{2}");\n')
      ids_written.append(row[0])
      written_per_month[dt.month] += 1
  for f in files.values():
    f.close()


def get_keyword_paragraph(text, keyword, num_surround_pars):
  keyword_idx = np.array([m.start() for m in re.finditer(keyword, text)])
  break_idx = np.array([m.start() for m in re.finditer('\n', text)])
  paragraphs = []
  for k_idx in keyword_idx:
    break_diff = break_idx-k_idx
    next_break_idx = -1
    last_break_idx = -1
    if len(break_diff[break_diff > 0]) > 0:
      next_break_idx = np.where(break_diff==min(break_diff[break_diff > 0]))[0]
    if len(break_diff[break_diff < 0]) > 0:
      last_break_idx = np.where(break_diff==-1*min(abs(break_diff[break_diff < 0])))[0]
    
    end_idx = len(text)
    if next_break_idx > -1:
      end_idx = break_idx[min(next_break_idx+num_surround_pars, len(break_idx)-1)]

    start_idx = 0
    if last_break_idx > -1:
      # Add the +1 to skip the newline character
      start_idx = break_idx[max(last_break_idx-num_surround_pars, 0)]+1

    par = text[int(start_idx):int(end_idx)]
    if par not in paragraphs:
      paragraphs.append(par.replace('\n','\\n ').replace('"','\\"'))
  return paragraphs

'''
PANDAS FUNCTIONS
'''

def rows_containing_covid(df):
  return df[df['post_text'].str.contains('COVID|coronavirus|pandemic|virus')]
