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

# For TOR interactions
from stem import Signal
from stem.control import Controller
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

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

# signal TOR for a new connection
def switchIP():
  with Controller.from_port(port = 9051) as controller:
    controller.authenticate()
    controller.signal(Signal.NEWNYM)

# get a new selenium webdriver with tor as the proxy
def my_proxy(PROXY_HOST,PROXY_PORT):
  fp = webdriver.FirefoxProfile()
  # Direct = 0, Manual = 1, PAC = 2, AUTODETECT = 4, SYSTEM = 5
  fp.set_preference("network.proxy.type", 1)
  fp.set_preference("network.proxy.socks",PROXY_HOST)
  fp.set_preference("network.proxy.socks_port",int(PROXY_PORT))
  fp.update_preferences()
  options = Options()
  options.headless = True
  return webdriver.Firefox(options=options, firefox_profile=fp)

def scrape_body(url):
  # proxy = my_proxy("127.0.0.1", 9050)
  headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 'Accept-Encoding': 'gzip, deflate, br', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'}
  body = requests.get(url, headers=headers)
  # body = proxy.get(url, headers=headers)
  # proxy.get(url)
  # body = proxy.page_source
  # switchIP()
  return body.text

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

# Mappings from media name to media_id
NYT = 1
FOX = 1092
TUCKER_CARLSON = 1093
SEAN_HANNITY = 1094
LAURA_INGRAHAM = 1095
HUFF_POST = 623375
BREITBART = 19334

text_parsing_functions = {
  NYT: parseNYTText,
  FOX: parseFoxText,
  TUCKER_CARLSON: parseFoxText,
  SEAN_HANNITY: parseFoxText,
  LAURA_INGRAHAM: parseFoxText,
  HUFF_POST: parseHuffPostText,
  BREITBART: parseBreitbartText,
}

media_id_to_name = {
  NYT: 'New York Times',
  FOX: 'Fox News',
  TUCKER_CARLSON: 'TuckerCarlson',
  SEAN_HANNITY: 'SeanHannity',
  LAURA_INGRAHAM: 'LauraIngraham',
  HUFF_POST: 'Huffington Post - United States',
  BREITBART: 'Breitbart'
}

organizations_to_parse = [
  NYT, FOX, HUFF_POST, BREITBART, TUCKER_CARLSON, SEAN_HANNITY, LAURA_INGRAHAM
]

NUM_THREADS = 10

# NOTE: To read mediacloud df, the separator character is below:
MC_SEP = '\x1c'
# the command is pd.read_csv(path, sep=MC_SEP)
# to get raw text: parse_mc_article_html(df)

def load_article_parallel(row):
  print(f'Parsing url at row {row[0]}')
  try:
    row_data = row[1]
    parse_fn = text_parsing_functions[row_data.media_id]
    body = scrape_body(row_data.url)
    return parse_fn(body)
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

'''
javascript to get JSON for the article data:
$('.article > .info > header > .title > a').toArray().reduce((accum, cur) => { accum.push({ url: cur.href, title: cur.innerHTML }); return accum}, [])
'''
def write_article_json_to_mc_csv(in_filename, out_filename, media_id):
  in_file = open(in_filename, 'r')
  # out_file = open(out_file, 'w')
  articles = json.load(in_file)
  mc_data = []
  for article in articles:
    mc_data.append({
      'stories_id': '', 'publish_date': '', 'media_url': '',
      'media_id': media_id,
      'media_name': media_id_to_name[media_id],
      'url': article['url'],
      'language': 'en',
      'title': article['title'], 
    })
  df = pd.DataFrame(mc_data)
  df.to_csv(out_filename)
  return df

# This is from the post_accounts_seed.sql file (should be correct??)
POST_ACCOUNTS_IDS = {
  NYT: 6,
  FOX: 1,
  HUFF_POST: 2,
  BREITBART: 3,
  TUCKER_CARLSON: 8,
  SEAN_HANNITY: 9,
  LAURA_INGRAHAM: 10
}

def write_mc_df_to_sql(df):
  media_ids = df.media_id.unique()
  files = { media_id_to_name[media_id]: io.open(f'mysql/articles/{media_id_to_name[media_id]}.sql', 'w', encoding='utf-8') for media_id in media_ids }
  for row in df.iterrows():
    print(f'Writing row {row[0]}')
    row_data = row[1]
    f = files[row_data.media_name]
    pars = get_keyword_paragraph(row_data.article_data_raw, 'mask', 2)
    print(len(pars))
    for par in pars:
      f.write(f'INSERT INTO `articles_mask` (`post`,`native_id`,`post_account_id`,`post_type`) VALUES ("{par}","{row_data.stories_id}","{POST_ACCOUNTS_IDS[row_data.media_id]}","{2}");\n')
  for f in files.values():
    f.close()

def write_mc_df_to_sql_date_sample(df):
  files = { name: io.open(f'mysql/articles/{name}_sample.sql', 'w', encoding='utf-8') for name in media_id_to_name.values() }
  NUM_PER_MONTH = 10
  ids_written = []
  written_per_month = { month: 0 for month in range(1,13) }
  while len(ids_written) < NUM_PER_MONTH * 12:
    sample = df.sample()
    for row in sample.iterrows():
    # print(row)
      # print(f'Checking row {row[0]}')
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
        pars = get_keyword_paragraph(row_data.article_data_raw, 'mask', 1)
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
      paragraphs.append(par.replace('\n','<br/><br/>').replace('"','\\"'))
  return paragraphs

'''
PANDAS FUNCTIONS
'''

def rows_containing_covid(df):
  return df[df['post_text'].str.contains('COVID|coronavirus|pandemic|virus')]
