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
from datetime import datetime, date

from nlp_helper import split_into_sentences

# For TOR interactions
# from stem import Signal
# from stem.control import Controller
# from selenium import webdriver
# from selenium.webdriver.firefox.options import Options

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
  '''
  Use the facebook scraper library to get posts from a certain fb page up to
  250 pages back. When this was written, it got posts back to March 2020, but
  that is no longer the case.

  :param page: The name of the page to fetch (as it appears in the fb url)
  '''
  p = 250
  f = open(f'./data/fb-posts/{page}.json', 'w')
  f.write('[')
  for post in get_posts(page, pages=p):
    post['time'] = str(post['time'])
    f.write(json.dumps(post) + ',\n')
  f.write(']')
  f.close()

def collect_fb_data():
  '''
  Collect facebook data from the pages contained in the dictionary object above.
  '''
  for key in fb_page_names:
    print(f'Fetching data for {key}...')
    get_fb_posts_from_march_onward(fb_page_names[key])

def json_posts_to_df(filepath):
  return pd.read_json(filepath)

'''
MEDIA CLOUD DATA

Use these functions for working with MediaCloud data -- MediaCloud is a web platform
developed by researchers at the Berkman-Klein center at Harvard for analyzing
media discourse. It aggregates media stories from a variety of sources, and can
search them with boolean-style query language.
'''

# signal TOR for a new connection
def switchIP():
  '''
  When the TOR process is running, get a new IP address.
  '''
  with Controller.from_port(port = 9051) as controller:
    controller.authenticate()
    controller.signal(Signal.NEWNYM)

# get a new selenium webdriver with tor as the proxy
def my_proxy(PROXY_HOST,PROXY_PORT):
  '''
  Get a new TOR proxy to send web requests through.

  :param PROXY_HOST: The host to set up the proxy with (usually localhost)
  :param PROXY_PORT: The port to point it to.
  '''
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
  '''
  Given a web url, issue an http GET request to get the entire HTML body of the
  page for later parsing.

  :param url: The url to fetch data from.
  '''
  # proxy = my_proxy("127.0.0.1", 9050)
  headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 'Accept-Encoding': 'gzip, deflate, br', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'}
  body = requests.get(url, headers=headers)
  # body = proxy.get(url, headers=headers)
  # proxy.get(url)
  # body = proxy.page_source
  # switchIP()
  return body.text

def parseNYTText(body):
  '''
  Use the BeautifulSoup library (a markup parser) to pull out the main story text
  from a New York Times article.

  :param body: Raw HTML text to feed into BeautifulSoup for further parsing.
  '''
  soup = BeautifulSoup(body)
  text = ''
  article_class = 'css-1r7ky0e'
  paragraph_class = 'css-axufdj'
  # text += str(soup.find(attrs={'class': article_class}))
  for p in soup.find_all(attrs={'class': paragraph_class}):
    text += p.text + "\n"
  return text

def parseFoxText(body):
  '''
  Use the BeautifulSoup library (a markup parser) to pull out the main story text
  from a Fox News article.

  :param body: Raw HTML text to feed into BeautifulSoup for further parsing.
  '''
  soup = BeautifulSoup(body)
  text = ''
  body_class = 'article-body'
  # text += str(soup.find(attrs={'class': body_class}))
  for p in soup.find(attrs={'class': body_class}).find_all('p'):
    text += p.text + "\n"
  return text

def parseHuffPostText(body):
  '''
  Use the BeautifulSoup library (a markup parser) to pull out the main story text
  from a Huffington Post article.

  :param body: Raw HTML text to feed into BeautifulSoup for further parsing.
  '''
  soup = BeautifulSoup(body)
  text = ''
  par_class = 'content-list-component'
  # article_class = 'entry__text'
  # text += str(soup.find(attrs={'class': article_class}))
  for p in soup.find_all(attrs={'class': par_class}):
    text += p.text + "\n"
  return text

def parseBreitbartText(body):
  '''
  Use the BeautifulSoup library (a markup parser) to pull out the main story text
  from a Breitbart article.

  :param body: Raw HTML text to feed into BeautifulSoup for further parsing.
  '''
  soup = BeautifulSoup(body)
  body_class = 'entry-content'
  text = ''
  for p in soup.find(attrs={'class': body_class}).find_all('p'):
    text += p.text + '\n'
  return text

def parseVoxText(body):
  '''
  Use the BeautifulSoup library (a markup parser) to pull out the main story text
  from a Vox article.

  :param body: Raw HTML text to feed into BeautifulSoup for further parsing.
  '''
  soup = BeautifulSoup(body)
  body_class = 'c-entry-content'
  text = ''
  for p in soup.find(attrs={'class': body_class}).find_all('p'):
    text += p.text + '\n'
  return text

def parseDailyKosText(body):
  '''
  Use the BeautifulSoup library (a markup parser) to pull out the main story text
  from a Daily Kos article.

  :param body: Raw HTML text to feed into BeautifulSoup for further parsing.
  '''
  soup = BeautifulSoup(body)
  body_class = 'story__content'
  text = ''
  for p in soup.find_all(attrs={'class': body_class})[1].find_all('p'):
    text += p.text + '\n'
  return text

'''
Mappings from media name to media_id: These are the media-ids used by MediaCloud
for different news sources. Some of them are also made up by me (Carlson, Hannity,
and Ingraham) because MediaCloud does not aggregate their transcripts.
'''
NYT = 1
FOX = 1092
TUCKER_CARLSON = 1093
SEAN_HANNITY = 1094
LAURA_INGRAHAM = 1095
HUFF_POST = 623375
BREITBART = 19334
DAILY_KOS = 115
VOX = 104828

'''
A mapping of media_id -> parsing function for ease of use
'''
text_parsing_functions = {
  NYT: parseNYTText,
  FOX: parseFoxText,
  TUCKER_CARLSON: parseFoxText,
  SEAN_HANNITY: parseFoxText,
  LAURA_INGRAHAM: parseFoxText,
  HUFF_POST: parseHuffPostText,
  BREITBART: parseBreitbartText,
  VOX: parseVoxText,
  DAILY_KOS: parseDailyKosText
}

'''
A mapping of media_id -> readable name
'''
media_id_to_name = {
  NYT: 'New York Times',
  FOX: 'Fox News',
  TUCKER_CARLSON: 'TuckerCarlson',
  SEAN_HANNITY: 'SeanHannity',
  LAURA_INGRAHAM: 'LauraIngraham',
  HUFF_POST: 'Huffington Post - United States',
  BREITBART: 'Breitbart',
  VOX: 'Vox',
  DAILY_KOS: 'Daily Kos'
}

'''
This is from the post_accounts_seed.sql file (should be correct??)
A mapping from media_id -> unique ID used in the mysql database (used for sql file
writing).
'''
POST_ACCOUNTS_IDS = {
  NYT: 6,
  FOX: 1,
  HUFF_POST: 2,
  BREITBART: 3,
  TUCKER_CARLSON: 8,
  SEAN_HANNITY: 9,
  LAURA_INGRAHAM: 10,
  VOX: 11,
  DAILY_KOS: 5
}

'''
An array to turn on/off certain organizations to look at from a large MediaCloud
dataframe that may include many organizations that we don't want to parse data
from.
'''
organizations_to_parse = [
  NYT, FOX, HUFF_POST, BREITBART, TUCKER_CARLSON, SEAN_HANNITY, LAURA_INGRAHAM, DAILY_KOS, VOX
]

NUM_THREADS = 10

# NOTE: To read mediacloud df, the separator character is below:
MC_SEP = '\x1c'
# the command is pd.read_csv(path, sep=MC_SEP)
# to get raw text: parse_mc_article_html(df)
SENT_SEP = chr(181)

'''
The number of surrounding paragraphs to get for entries containing
the keyword.
'''
SURROUNDING_PARAGRAPHS = 0

def load_article_parallel(row):
  '''
  A function that can parse one row of data from a dataframe, pull the data from
  the url, and return the properly parsed function based on the media_id
  of the row. This function is written purposely this way so it can be
  parallelized with a thread pool.

  :param row: One dataframe row to process.
  '''
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
  '''
  Read MediaCloud data from a pandas dataframe CSV, which includes:
  (1) Filtering the media organizations only to those we want to parse data from.
  (2) In parallel, scrape the website URL for data and parse properly to return
  a raw HTML version of the article body.
  (3) Add a column to the dataframe called 'article_data' that contains HTML from
  the article body.

  :param path: A path to the pandas df CSV file.
  '''
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
  '''
  Take article HTML and pull out raw text separated by '\n' newline characters.

  :param row: A dataframe row with HTML in the 'article_data' column to parse.
  '''
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
  '''
  Parse an entire dataframe's worth of mediacloud article HTML into raw text. This
  loops through the entire dataframe in parallel and converts all HTML into raw text
  housed in the column 'article_data_raw'

  :param df: The dataframe to parse.
  '''
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
  '''
  Convert JSON representing a list of media articles into a MediaCloud-style
  dataframe CSV file (so it can be used like other MediaCloud data frames). Thus
  far, this has only been used to convert lists of Tucker Carlson, Sean Hannity,
  and Laura Ingraham transcripts to readable dataframes.

  :param in_filename: A path and filename to the JSON file.
  :param out_filename: A path and filename to the CSV file to output.
  :param media_id: Some media_id to assign to all of the stories in the new
  dataframe.
  '''
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

def write_mc_df_to_sql(df):
  '''
  Write a MediaCloud dataframe's content into SQL files that can be used to seed
  a mySQL database.

  :param df: The dataframe to read article data from, and write to SQL files.
  '''
  media_ids = df.media_id.unique()
  files = { media_id_to_name[media_id]: io.open(f'./mysql/articles/{media_id_to_name[media_id]}.sql', 'w', encoding='utf-8') for media_id in media_ids }
  for row in df.iterrows():
    print(f'Writing row {row[0]}')
    row_data = row[1]
    f = files[row_data.media_name]
    pars = get_keyword_paragraph(row_data['article_data_raw'], 'mask', SURROUNDING_PARAGRAPHS)
    for par in pars:
      # sentences = split_into_sentences(par)
      # text = SENT_SEP.join(sentences)
      text = par
      f.write(f'INSERT INTO `articles_mask` (`post`,`native_id`,`post_account_id`,`post_type`) VALUES ("{text}","{row_data.stories_id}","{POST_ACCOUNTS_IDS[row_data.media_id]}","{2}");\n')
  for f in files.values():
    f.close()

def write_mc_df_to_sql_date_sample(df):
  '''
  Write a sampling of a MediaCloud dataframe's content into SQL files that can
  be used to seed a mySQL database. The sampling is based on time, attempting to
  only fetch NUM_PER_MONTH rows per month (e.g. only 10 articles per month, randomly fetched).

  :param df: The dataframe to read article data from, and write to SQL files.
  '''
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
        pars = get_keyword_paragraph(row_data.article_data_raw, 'mask', SURROUNDING_PARAGRAPHS)
        for par in pars:
          sentences = split_into_sentences(par)
          text = SENT_SEP.join(sentences)
          f.write(f'INSERT INTO `articles_mask` (`post`,`native_id`,`post_account_id`,`post_type`) VALUES ("{text}","{row_data.stories_id}","{POST_ACCOUNTS_IDS[row_data.media_id]}","{2}");\n')
        ids_written.append(row[0])
        written_per_month[dt.month] += 1
  for f in files.values():
    f.close()

def get_keyword_paragraph(text, keyword, num_surround_pars):
  '''
  Given a body of text, fetch series of paragraphs surrounding one where a specified
  keyword appears (e.g. 1 paragraph on either side of one where the word 'mask'
  appears). This function attempts to not return duplicate entires in the array
  of paragraphs returned.

  :param text: Raw text with newline characters separating paragraphs.
  :param keyword: A string keyword to search for.
  :param num_surround_pars: How many paragraphs before and after to fetch alongside
  the paragraph where a keyword appears (e.g. num_surround_pairs=2 will return
  two paragraphs before and after one containing the keyword).
  '''
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
    formatted_par = par.replace('\n',' <br/><br/> ').replace('"','\\"')
    if formatted_par not in paragraphs:
      paragraphs.append(formatted_par)
  return paragraphs

'''
PANDAS FUNCTIONS
'''

def rows_containing_covid(df):
  return df[df['post_text'].str.contains('COVID|coronavirus|pandemic|virus')]

def rows_within_time_range(df, start_date_string, end_date_string):
  '''
  Filter a dataframe by time start and end (inclusive).

  :param df: The dataframe to filter -- should have 'publish_date' as
  the date time stamp column.
  :param start_date_string: A string to start the date filtering at,
  should be in format 'YYYY-MM-DD'.
  :param end_date_string: A string to end the date filtering at,
  should be in format 'YYYY-MM-DD'.
  '''
  return df[(df['publish_date'] >= start_date_string) & (df['publish_date'] <= end_date_string)]

def rows_from_sources(df, source_list):
  query_str = ''
  for source in source_list:
    query_str += f'(media_id == {source}) or '
  return df.query(query_str[:-4])

def df_with_experiment_filters(df):
  if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
  if 'Unnamed: 0.1' in df.columns:
    df = df.drop(columns=['Unnamed: 0.1'])
  # TODO: Need to add CNN and Daily Kos
  source_list = [NYT, FOX, BREITBART]
  df_for_dates = rows_within_time_range(df, '2020-04-01','2020-06-14')
  df_for_sources = rows_from_sources(df_for_dates, source_list)
  df_without_nan = df_for_sources.dropna(subset=['article_data_raw'])
  return df_without_nan