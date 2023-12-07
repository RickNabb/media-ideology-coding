import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from datetime import date, timedelta
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import xml.etree.ElementTree as ET
from data_collector import NYT, FOX, TUCKER_CARLSON, SEAN_HANNITY, LAURA_INGRAHAM, HUFF_POST, BREITBART, DAILY_KOS, VOX, media_id_to_name, MC_SEP, POST_ACCOUNTS_IDS, add_dates_to_opinion_transcripts

# Pulled from https://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python

def daterange(start_date, end_date):
  for n in range(int((end_date - start_date).days)):
    yield start_date + timedelta(n)

def label_training_agreement_analysis(mask_training_codes_df):
  agreement_codes = {
    1: {
      'mw_comfort_breathe': 0,
      'mw_comfort_hot': 2,
      'mw_efficacy_health': 1,
      'mw_efficacy_eff': 1,
      'mw_access_diff': 2,
      'mw_access_cost': 2,
      'mw_compensation': 2,
      'mw_inconvenience_remember': 2,
      'mw_inconvenience_hassle': 2,
      'mw_appearance': 2,
      'mw_attention_trust': 2,
      'mw_attention_uncomfortable': 2,
      'mw_independence_forced': 2,
      'mw_independence_authority': 2
    },
    4: {
      'mw_comfort_breathe': 2,
      'mw_comfort_hot': 2,
      'mw_efficacy_health': 2,
      'mw_efficacy_eff': 2,
      'mw_access_diff': 0,
      'mw_access_cost': 2,
      'mw_compensation': 2,
      'mw_inconvenience_remember': 2,
      'mw_inconvenience_hassle': 2,
      'mw_appearance': 0,
      'mw_attention_trust': 0,
      'mw_attention_uncomfortable': 0,
      'mw_independence_forced': 0,
      'mw_independence_authority': 2
    },
  }

  # Get agreement score per participant -- WITHOUT CONFIDENCE
  agreement_scores = { session_id: {} for session_id in mask_training_codes_df['session_id'].unique() }
  for session_id in agreement_scores.keys():
    for article_id in agreement_codes.keys():
      session_article_codes = mask_training_codes_df[(mask_training_codes_df['session_id'] == session_id) & (mask_training_codes_df['article_id'] == article_id)][['attribute','code','confidence']]
      code_vector = np.array(session_article_codes['code'], dtype=int)
      gold_vector = np.array([ agreement_codes[article_id][attr.replace('_training','')] for attr in session_article_codes['attribute'] ], dtype=int)

      # print(f'{session_id},{article_id}')
      # print(code_vector)
      # print(gold_vector)
      
      # This should not be the case, but somehow it is
      if len(session_article_codes) == 0:
        continue

      agreement_vector = np.array([ agreement_codes[article_id][row[1]['attribute'].replace('_training','')] == row[1]['code'] for row in session_article_codes.iterrows() ], dtype=int)
      cohen_kappa = cohen_kappa_score(gold_vector, code_vector)
      agreement_scores[session_id][article_id] = { 'raw_agree': -1, 'cohen': -1, 'fleiss': -1 }
      agreement_scores[session_id][article_id]['cohen'] = cohen_kappa
      agreement_scores[session_id][article_id]['raw_agree'] = agreement_vector.sum() / len(agreement_vector)
  return agreement_scores

def percent_paragraphs_labeled_for_labeled_stories(mask_codes_df, articles_db_df):
  articles = mask_codes_df['native_id'].unique()
  article_percentages = { article_id: (len(mask_codes_df[mask_codes_df['native_id'] == article_id]['article_id'].unique()) / len(articles_db_df[articles_db_df['native_id'] == article_id])) for article_id in articles }
  return sorted(article_percentages.items(), key=lambda item: item[1])

def label_date_range_analysis(mask_codes_df, articles_df):
  outlets = [FOX, NYT, BREITBART, VOX, DAILY_KOS]
  cols = ['date'] + list(mask_codes_df['attribute'].unique()) + [ media_id_to_name[outlet] for outlet in outlets ] + ['total_labels']

  articles_no_nan = articles_df.dropna(subset=['publish_date'])
  # Article id = stories_id

  date_df = pd.DataFrame(columns=cols)
  start_date = date(2020, 4, 6)
  end_date = date(2020, 6, 8)
  for day in daterange(start_date, end_date):
    articles_for_date = articles_no_nan[articles_no_nan['publish_date'].str.contains(str(day))][['stories_id','media_id']]

    codes_for_date = mask_codes_df[mask_codes_df['native_id'].isin(articles_for_date['stories_id'])]
    non_nomention_codes = codes_for_date[codes_for_date['code'] != 2]

    # Media present in articles per date
    media_for_date = articles_for_date[articles_for_date['stories_id'].isin(non_nomention_codes['native_id'])]
    media_present = { outlet: len(media_for_date[media_for_date['media_id'] == outlet]) for outlet in outlets }
    
    # Codes per attribute
    num_codes_per_attr = { attr: len(non_nomention_codes[non_nomention_codes['attribute'] == attr]) for attr in list(mask_codes_df['attribute'].unique()) }
    num_codes_per_attr_list = list(num_codes_per_attr.values())
    total_codes = sum(num_codes_per_attr.values())

    new_row = [ day ] + num_codes_per_attr_list + list(media_present.values()) + [ total_codes ]
    date_df.loc[len(date_df)] = new_row

  return date_df

def num_times_paragraphs_labeled(labels_df):
  article_ids = labels_df['article_id'].unique()
  num_times_article_labeled = { article_id: len(labels_df[labels_df['article_id'] == article_id]['session_id'].unique()) for article_id in article_ids }
  num_times_labeled = { times: len([ num for num in num_times_article_labeled.values() if num == times ]) for times in set(num_times_article_labeled.values()) }
  return num_times_labeled

def num_paragraphs_labeled_per_category(labels_df):
  '''
  Return a dictionary of the number of labels (0,1,2) per category
  (the attribute) -- showing how many labels of each type we have.
  '''
  categories = labels_df['attribute'].unique()
  labels = labels_df['code'].unique()
  return { category: { label: len(labels_df[(labels_df['attribute'] == category) & (labels_df['code'] == label)]['article_id'].unique()) for label in labels } for category in categories }

def num_confidences_per_category(labels_df):
  '''
  Return a dictionary of the number of confidence scores of each value (1-7)
  for each category (attribute).
  '''
  categories = labels_df['attribute'].unique()
  confidences = labels_df['confidence'].unique()
  return { category: { confidence: len(labels_df[(labels_df['attribute'] == category) & (labels_df['confidence'] == confidence)]['article_id'].unique()) for confidence in confidences } for category in categories }

def graph_labels_across_dates(label_date_range_results):
  fig,ax = plt.subplots(figsize=(8,4))
  start_date = date(2020, 4, 6)
  end_date = date(2020, 6, 8)
  dates = list(daterange(start_date, end_date))

  ax.bar(dates, label_date_range_results['total_labels'])
  ax.set_xticks(dates)
  ax.set_xticklabels([f'{date.month}-{date.day}' for date in dates], rotation=-45, ha='left', fontsize=6)
  ax.set_xlabel('Date')
  ax.set_ylabel('Number of labels (without "Does not mention")')

  plt.show()

def graph_media_outlets_across_dates(label_date_range_results):
  outlets = [FOX, NYT, BREITBART, VOX, DAILY_KOS]

  fig,ax = plt.subplots(figsize=(8,4))
  start_date = date(2020, 4, 6)
  end_date = date(2020, 6, 8)
  dates = list(daterange(start_date, end_date))

  bottom = np.zeros(len(dates))
  for outlet in outlets:
    labels_for_outlet = label_date_range_results[media_id_to_name[outlet]]
    ax.bar(dates, labels_for_outlet, bottom=bottom, label=media_id_to_name[outlet])
    bottom += labels_for_outlet

  ax.set_xticks(dates)
  ax.set_xticklabels([f'{date.month}-{date.day}' for date in dates], rotation=-45, ha='left', fontsize=6)
  ax.set_xlabel('Date')
  ax.set_ylabel('Number of labeled articles per outlet\n(without "Does not mention")')
  ax.legend()

  plt.show()

def graph_media_outlets_total_across_dates(articles_df):
  outlets = [FOX, NYT, BREITBART, VOX, DAILY_KOS]
  articles_no_nan = articles_df.dropna(subset=['publish_date'])
  # Article id = stories_id

  date_df = pd.DataFrame(columns=outlets)
  start_date = date(2020, 4, 6)
  end_date = date(2020, 6, 8)
  for day in daterange(start_date, end_date):
    articles_for_date = articles_no_nan[articles_no_nan['publish_date'].str.contains(str(day))][['stories_id','media_id']]
    media_articles_for_date = [ len(articles_for_date[articles_for_date['media_id'] == outlet]['stories_id'].unique()) for outlet in outlets ]
    date_df.loc[len(date_df)] = media_articles_for_date

  fig,ax = plt.subplots(figsize=(8,4))
  dates = list(daterange(start_date, end_date))

  bottom = np.zeros(len(dates))
  for outlet in outlets:
    articles_for_outlet = date_df[outlet]
    ax.bar(dates, articles_for_outlet, bottom=bottom, label=media_id_to_name[outlet])
    bottom += articles_for_outlet

  ax.set_xticks(dates)
  ax.set_xticklabels([f'{date.month}-{date.day}' for date in dates], rotation=-45, ha='left', fontsize=6)
  ax.set_xlabel('Date')
  ax.set_ylabel('Number of articles per outlet')
  ax.legend()

  plt.show()

def graph_percent_articles_at_least_one_code_across_dates(mask_wearing_df, articles_df):
  date_df = pd.DataFrame(columns=['date','percent_articles_w_label'])
  articles_no_nan = articles_df.dropna(subset=['publish_date'])
  start_date = date(2020, 4, 6)
  end_date = date(2020, 6, 8)
  for day in daterange(start_date, end_date):
    articles_for_date = articles_no_nan[articles_no_nan['publish_date'].str.contains(str(day))]['stories_id']
    articles_w_label = mask_wearing_df[mask_wearing_df['native_id'].isin(articles_for_date)]['native_id'].unique()
    date_df.loc[len(date_df)] = [day, (len(articles_w_label)/len(articles_for_date))]

  dates = list(daterange(start_date, end_date))
  fig,ax = plt.subplots(figsize=(8,4))

  ax.bar(dates, date_df['percent_articles_w_label']*100)
  ax.set_xticks(dates)
  ax.set_xticklabels([f'{date.month}-{date.day}' for date in dates], rotation=-45, ha='left', fontsize=6)
  ax.set_xlabel('Date')
  ax.set_ylabel('Percent of articles w/ at least 1 label')

  plt.show()

def graph_percent_paragraphs_at_least_one_code_across_dates(mask_wearing_df, articles_df, articles_db_df):
  date_df = pd.DataFrame(columns=['date','percent_paragraphs_w_label'])
  articles_no_nan = articles_df.dropna(subset=['publish_date'])
  start_date = date(2020, 4, 6)
  end_date = date(2020, 6, 8)
  for day in daterange(start_date, end_date):
    articles_for_date = articles_no_nan[articles_no_nan['publish_date'].str.contains(str(day))]['stories_id']
    paragraphs_for_date = articles_db_df[articles_db_df['native_id'].isin(articles_for_date)]['id']
    paragraphs_w_label = mask_wearing_df[mask_wearing_df['article_id'].isin(paragraphs_for_date)]['native_id'].unique()
    date_df.loc[len(date_df)] = [day, (len(paragraphs_w_label)/len(paragraphs_for_date))]

  dates = list(daterange(start_date, end_date))
  fig,ax = plt.subplots(figsize=(8,4))

  ax.bar(dates, date_df['percent_paragraphs_w_label']*100)
  ax.set_xticks(dates)
  ax.set_xticklabels([f'{date.month}-{date.day}' for date in dates], rotation=-45, ha='left', fontsize=6)
  ax.set_xlabel('Date')
  ax.set_ylabel('Percent of paragraphs w/ at least 1 label')

  plt.show()

def graph_media_outlet_total_distribution(label_date_range_results):
  outlets = [FOX, NYT, BREITBART, VOX, DAILY_KOS]

  fig,ax = plt.subplots(figsize=(8,4))

  labels_per_outlet = [ sum(label_date_range_results[media_id_to_name[outlet]]) for outlet in outlets ]
  ax.bar([ media_id_to_name[outlet] for outlet in outlets ], labels_per_outlet)

  ax.set_xticklabels([media_id_to_name[outlet] for outlet in outlets], fontsize=8)
  ax.set_xlabel('Media Outlet')
  ax.set_ylabel('Number of labeled articles per outlet (without "Does not mention")')

  plt.show()

def graph_articles_per_outlet_distribution(articles_db_df):
  outlets = [FOX, NYT, BREITBART, VOX, DAILY_KOS]

  fig,ax = plt.subplots(figsize=(8,4))

  articles_per_outlet = [ len(articles_db_df[articles_db_df['article_account_id'] == POST_ACCOUNTS_IDS[outlet]]['native_id'].unique()) for outlet in outlets ]
  ax.bar([ media_id_to_name[outlet] for outlet in outlets ], articles_per_outlet)

  ax.set_xticklabels([media_id_to_name[outlet] for outlet in outlets], fontsize=8)
  ax.set_xlabel('Media Outlet')
  ax.set_ylabel('Number of articles per outlet')

  plt.show()

def graph_labels_by_partisan_diet_across_dates(label_date_range_results):
  media_diets = { 
    'rep': [ media_id_to_name[outlet] for outlet in REP_MEDIA_OUTLETS ],
    'mod': [ media_id_to_name[outlet] for outlet in MOD_MEDIA_OUTLETS ],
    'dem': [ media_id_to_name[outlet] for outlet in DEM_MEDIA_OUTLETS ]
  }
  media_diets_and_colors = { 'rep': 'red', 'mod': 'gray', 'dem': 'blue' }

  fig,ax = plt.subplots(figsize=(8,4))
  start_date = date(2020, 4, 6)
  end_date = date(2020, 6, 8)
  dates = list(daterange(start_date, end_date))

  bottom = np.zeros(len(dates))
  for name,diet in media_diets.items():
    labels_from_outlets = [ list(label_date_range_results[outlet]) for outlet in diet ]
    labels_for_diet = np.array(labels_from_outlets).sum(axis=0)
    ax.bar(dates, labels_for_diet, bottom=bottom, label=name, color=media_diets_and_colors[name])
    bottom += labels_for_diet

  ax.set_xticks(dates)
  ax.set_xticklabels([f'{date.month}-{date.day}' for date in dates], rotation=-45, ha='left', fontsize=6)
  ax.set_xlabel('Date')
  ax.set_ylabel('Number of labeled articles per partisanship\n(without "Does not mention")')
  ax.legend()

  plt.show()

def graph_total_labels_by_partisan_diet(mask_wearing_df, articles_db_df):
  media_diets = [ 
    [ POST_ACCOUNTS_IDS[outlet] for outlet in REP_MEDIA_OUTLETS ],
    [ POST_ACCOUNTS_IDS[outlet] for outlet in MOD_MEDIA_OUTLETS ],
    [ POST_ACCOUNTS_IDS[outlet] for outlet in DEM_MEDIA_OUTLETS ]
  ]

  fig,ax = plt.subplots(figsize=(8,4))

  articles_per_diet = [ articles_db_df[articles_db_df['article_account_id'].isin(media_diet)]['id'].unique() for media_diet in media_diets ]
  paragraphs_labels_per_diet = [ len(mask_wearing_df[mask_wearing_df['article_id'].isin(articles)]['article_id'].unique()) for articles in articles_per_diet ]
  bar_labels = [ 'Rep', 'Mod', 'Dem' ]
  ax.bar(bar_labels, paragraphs_labels_per_diet, label=bar_labels, color=['red','gray','blue'])

  ax.set_xticklabels(bar_labels, fontsize=8)
  ax.set_xlabel('Partisanship')
  ax.set_ylabel('Number of labeled paragraphs per\npartisan media diet')

  plt.show()

def graph_labels_by_partisan_diet(mask_wearing_df, articles_db_df):
  media_diets = { 
    'rep': [ POST_ACCOUNTS_IDS[outlet] for outlet in REP_MEDIA_OUTLETS ],
    'mod': [ POST_ACCOUNTS_IDS[outlet] for outlet in MOD_MEDIA_OUTLETS ],
    'dem': [ POST_ACCOUNTS_IDS[outlet] for outlet in DEM_MEDIA_OUTLETS ]
  }

  codes = mask_wearing_df['code'].unique()
  articles_per_diet = { partisan: articles_db_df[articles_db_df['article_account_id'].isin(media_diet)]['id'].unique() for partisan,media_diet in media_diets.items() }
  paragraphs_labels_per_diet = { partisan: [ len(mask_wearing_df[(mask_wearing_df['article_id'].isin(articles)) & (mask_wearing_df['code'] == code)]['article_id'].unique()) for code in codes ] for partisan,articles in articles_per_diet.items() }

  x = np.arange(len(codes))
  width = 0.25
  multiplier = 0

  fig,ax = plt.subplots(figsize=(8,4))
  media_diets_and_colors = { 'rep': 'red', 'mod': 'gray', 'dem': 'blue' }

  for partisan,num_codes in paragraphs_labels_per_diet.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, num_codes, width, label=partisan, color=media_diets_and_colors[partisan])
    ax.bar_label(rects, padding=3)
    multiplier+=1

  # bar_labels = [ 'Rep', 'Mod', 'Dem' ]
  ax.set_xticks(x+offset)
  ax.set_xticklabels(codes, fontsize=8)
  ax.set_xlabel('Code')
  ax.set_ylabel('Number of labeled paragraphs per\npartisan media diet')
  ax.legend()

  plt.show()

def graph_articles_per_partisanship_distribution(articles_db_df):
  media_diets = [ 
    [ POST_ACCOUNTS_IDS[outlet] for outlet in REP_MEDIA_OUTLETS ],
    [ POST_ACCOUNTS_IDS[outlet] for outlet in MOD_MEDIA_OUTLETS ],
    [ POST_ACCOUNTS_IDS[outlet] for outlet in DEM_MEDIA_OUTLETS ]
  ]

  fig,ax = plt.subplots(figsize=(8,4))

  articles_per_outlet = [ len(articles_db_df[articles_db_df['article_account_id'].isin(media_diet)]['native_id'].unique()) for media_diet in media_diets ]
  bar_labels = [ 'Rep', 'Mod', 'Dem' ]
  ax.bar(bar_labels, articles_per_outlet)

  ax.set_xticklabels(bar_labels, fontsize=8)
  ax.set_xlabel('Partisanship')
  ax.set_ylabel('Number of articles per partisan media diet')

  plt.show()

def graph_paragraphs_per_partisanship_distribution(articles_db_df):
  media_diets = [ 
    [ POST_ACCOUNTS_IDS[outlet] for outlet in REP_MEDIA_OUTLETS ],
    [ POST_ACCOUNTS_IDS[outlet] for outlet in MOD_MEDIA_OUTLETS ],
    [ POST_ACCOUNTS_IDS[outlet] for outlet in DEM_MEDIA_OUTLETS ]
  ]

  fig,ax = plt.subplots(figsize=(8,4))

  paragraphs_per_outlet = [ len(articles_db_df[articles_db_df['article_account_id'].isin(media_diet)]) for media_diet in media_diets ]
  bar_labels = [ 'Rep', 'Mod', 'Dem' ]
  ax.bar(bar_labels, paragraphs_per_outlet)

  ax.set_xticklabels(bar_labels, fontsize=8)
  ax.set_xlabel('Partisanship')
  ax.set_ylabel('Number of paragraphs per partisan media diet')

  plt.show()

def graph_articles_partisanship_over_time(articles_df):
  media_diets = [ 
    REP_MEDIA_OUTLETS,
    MOD_MEDIA_OUTLETS,
    DEM_MEDIA_OUTLETS 
  ]

  date_df = pd.DataFrame(columns=['rep','mod','dem'])
  articles_no_nan = articles_df.dropna(subset=['publish_date'])
  start_date = date(2020, 4, 6)
  end_date = date(2020, 6, 8)
  for day in daterange(start_date, end_date):
    articles_for_date = articles_no_nan[articles_no_nan['publish_date'].str.contains(str(day))][['stories_id', 'media_id']]
    # paragraphs_for_date = articles_db_df[articles_db_df['native_id'].isin(articles_for_date)]['id']
    date_df.loc[len(date_df)] = [
      len(articles_for_date[articles_for_date['media_id'].isin(media_diets[0])]),
      len(articles_for_date[articles_for_date['media_id'].isin(media_diets[1])]),
      len(articles_for_date[articles_for_date['media_id'].isin(media_diets[2])]),
    ]

  fig,ax = plt.subplots(figsize=(8,4))
  dates = list(daterange(start_date, end_date))

  bottom = np.zeros(len(dates))
  media_diets_and_colors = { 'rep': 'red', 'mod': 'gray', 'dem': 'blue' }
  for diet in media_diets_and_colors:
    articles_for_outlet = date_df[diet]
    ax.bar(dates, articles_for_outlet, bottom=bottom, label=diet, color=media_diets_and_colors[diet])
    bottom += articles_for_outlet

  ax.set_xticks(dates)
  ax.set_xticklabels([f'{date.month}-{date.day}' for date in dates], rotation=-45, ha='left', fontsize=6)
  ax.set_xlabel('Date')
  ax.set_ylabel('Number of articles per partisan media diet')
  ax.legend()

  plt.show()

def graph_paragraphs_partisanship_over_time(articles_df, articles_db_df):
  media_diets = [ 
    [ POST_ACCOUNTS_IDS[outlet] for outlet in REP_MEDIA_OUTLETS ],
    [ POST_ACCOUNTS_IDS[outlet] for outlet in MOD_MEDIA_OUTLETS ],
    [ POST_ACCOUNTS_IDS[outlet] for outlet in DEM_MEDIA_OUTLETS ]
  ]

  date_df = pd.DataFrame(columns=['rep','mod','dem'])
  articles_no_nan = articles_df.dropna(subset=['publish_date'])
  start_date = date(2020, 4, 6)
  end_date = date(2020, 6, 8)
  for day in daterange(start_date, end_date):
    articles_for_date = articles_no_nan[articles_no_nan['publish_date'].str.contains(str(day))]['stories_id']
    paragraphs_for_date = articles_db_df[articles_db_df['native_id'].isin(articles_for_date)][['id','article_account_id']]
    date_df.loc[len(date_df)] = [
      len(paragraphs_for_date[paragraphs_for_date['article_account_id'].isin(media_diets[0])]),
      len(paragraphs_for_date[paragraphs_for_date['article_account_id'].isin(media_diets[1])]),
      len(paragraphs_for_date[paragraphs_for_date['article_account_id'].isin(media_diets[2])]),
    ]

  fig,ax = plt.subplots(figsize=(8,4))
  dates = list(daterange(start_date, end_date))

  bottom = np.zeros(len(dates))
  media_diets_and_colors = { 'rep': 'red', 'mod': 'gray', 'dem': 'blue' }
  for diet in media_diets_and_colors:
    articles_for_outlet = date_df[diet]
    ax.bar(dates, articles_for_outlet, bottom=bottom, label=diet, color=media_diets_and_colors[diet])
    bottom += articles_for_outlet

  ax.set_xticks(dates)
  ax.set_xticklabels([f'{date.month}-{date.day}' for date in dates], rotation=-45, ha='left', fontsize=6)
  ax.set_xlabel('Date')
  ax.set_ylabel('Number of articles per partisan media diet')
  ax.legend()

  plt.show()

def graph_label_confidence_distribution(mask_codes_df):
  fig,ax = plt.subplots(figsize=(8,4))

  ax.bar(range(1,8), [ len(mask_codes_df[mask_codes_df['confidence'] == i]) for i in range(1,8) ])
  ax.set_xticks(range(1,8))
  ax.set_xticklabels(range(1,8))
  ax.set_xlabel('Confidence Score')
  ax.set_ylabel('Number of labels for confidence score')

  plt.show()

def graph_label_confidence_distribution_per_category(mask_codes_df):
  for category in mask_codes_df['attribute'].unique():
    fig,ax = plt.subplots(figsize=(8,4))

    ax.bar(range(1,8), [ len(mask_codes_df[(mask_codes_df['attribute'] == category) & (mask_codes_df['confidence'] == i)]) for i in range(1,8) ])
    ax.set_xticks(range(1,8))
    ax.set_xticklabels(range(1,8))
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Number of labels for confidence score')
    ax.set_title(f'Confidences for "{category}"')

    plt.show()

def graph_article_belief_over_time(mask_codes_df, articles_db_df):
  '''
  
  Note: mask_codes_df should have resolved all codes already, so that
  there are not multiple sets of codes per article
  '''
  date_df = pd.DataFrame(columns=['0','1','2','3','4','5','6'])
  articles_no_nan = articles_db_df.dropna(subset=['publish_date'])
  start_date = date(2020, 4, 6)
  end_date = date(2020, 6, 8)
  for day in daterange(start_date, end_date):
    articles_for_date = articles_no_nan[articles_no_nan['publish_date'].str.contains(str(day))]['stories_id']
    codes_for_date = mask_codes_df[mask_codes_df['native_id'].isin(articles_for_date)]
    num_beliefs = np.zeros(7)
    for article_id in codes_for_date['article_id'].unique():
      article_rows = codes_for_date[codes_for_date['article_id']==article_id]
      attribute_code_pairs = { row[1]['attribute']: row[1]['code'] for row in article_rows.iterrows() }
      article_belief = article_belief_value(attribute_code_pairs)
      num_beliefs[article_belief] += 1

    date_df.loc[len(date_df)] = num_beliefs

  dates = list(daterange(start_date, end_date))
  x = np.arange(len(dates))
  width = 1
  bottom = np.zeros(len(x))

  fig,ax = plt.subplots(figsize=(8,4))
  rgb_to_hex = lambda rgb: '%02x%02x%02x' % tuple(rgb)
  resolution = 7
  line_color = lambda key: f"#{rgb_to_hex([ 255 - round((255/(resolution-1))*int(key)), 0, round((255/(resolution-1)) * int(key)) ])}"
  belief_and_colors = { str(bel): line_color(bel) for bel in range(7) }

  for belief in date_df.columns:
    beliefs_over_dates = date_df[belief]
    rects = ax.bar(x, beliefs_over_dates, width, bottom=bottom, label=belief, color=belief_and_colors[belief])
    ax.bar_label(rects, padding=3)
    bottom += beliefs_over_dates

  # bar_labels = [ 'Rep', 'Mod', 'Dem' ]
  ax.set_xticks(x)
  ax.set_xticklabels(dates, fontsize=8, rotation=45)
  ax.set_xlabel('Date')
  ax.set_ylabel('Number of encoded beliefs per date')
  ax.legend()

  plt.show()

def ratings_for_category_for_paragraph(mask_codes_df, article_id, attribute):
  '''
  Returns a dataframe containing all ratings for a given article_id (paragraph).
  This reports the rating across one category (attribute).
  '''
  agreement_cols = [ attribute ]
  codes_for_article = mask_codes_df[mask_codes_df['article_id'] == article_id]
  coders = codes_for_article['session_id'].unique()
  table = pd.DataFrame(columns=agreement_cols)
  for coder in coders:
    table.loc[len(table)] = [ codes_for_article[(codes_for_article['session_id'] == coder) & (codes_for_article['attribute'] == col)].iloc[0]['code'] for col in agreement_cols ]
  return table

def ratings_across_categories_for_paragraph(mask_codes_df, article_id):
  '''
  Returns a dataframe containing all ratings for a given article_id (paragraph).
  This reports the rating across all categories (attributes).
  '''
  print(f'rating for article id {article_id}')
  agreement_cols = mask_codes_df['attribute'].unique()
  codes_for_article = mask_codes_df[mask_codes_df['article_id'] == article_id]
  coders = codes_for_article['session_id'].unique()
  table = pd.DataFrame(columns=agreement_cols)
  for coder in coders:
    session_attribute_code = lambda codes_for_article, col: codes_for_article[(codes_for_article['session_id'] == coder) & (codes_for_article['attribute'] == col)]
    table.loc[len(table)] = [ session_attribute_code(codes_for_article, col).iloc[0]['code'] if len(session_attribute_code(codes_for_article, col)) > 0 else -1 for col in agreement_cols ]
  return table

def inter_rater_agreement_for_category(mask_codes_df, article_id, attribute):
  table = ratings_for_category_for_paragraph(mask_codes_df, article_id, attribute)
  # return aggregate_raters(np.transpose(table.to_numpy()))
  return fleiss_kappa(aggregate_raters(np.transpose(table.to_numpy()))[0])

def inter_rater_agreement_across_categories(mask_codes_df, article_id):
  table = ratings_across_categories_for_paragraph(mask_codes_df, article_id)
  if len(table) > 2:
  # return aggregate_raters(np.transpose(table.to_numpy()))
    return fleiss_kappa(aggregate_raters(np.transpose(table.to_numpy()))[0])
  else:
    table_np = table.to_numpy()
    return cohen_kappa_score(table_np[0], table_np[1])

def inter_rater_agreement_across_attributes_histogram(mask_codes_df):
  fig,ax = plt.subplots(figsize=(8,4))
  paragraphs = mask_codes_df['article_id'].unique()
  multi_coded_paragraphs = [ paragraph for paragraph in paragraphs if len(mask_codes_df[mask_codes_df['article_id'] == paragraph]['session_id'].unique()) > 1 ]
  agreements = { paragraph: inter_rater_agreement_across_categories(mask_codes_df, paragraph) for paragraph in multi_coded_paragraphs }

  ax.hist([ round(val, 2) for val in agreements.values() ])
  ax.set_xlabel('Agreement scores')
  ax.set_ylabel('Frequency of agreement')
  ax.set_title(f'Inter-rater agreement histogram')

  plt.show()

def combine_coding_rounds():
  data_path = './labeled-data/'
  sub_paths = [ 'round1', 'round2', 'round3', 'round4' ]
  training_df = pd.DataFrame(columns=['id','article_id','attribute','code','confidence','session_id','datetime'])
  code_df = pd.DataFrame(columns=['id','article_id','attribute','code','confidence','session_id','datetime'])

  training_dfs = [training_df]
  code_dfs = [code_df]

  for sub_path in sub_paths:
    file_path = f'{data_path}/{sub_path}'
    training_round_df = pd.read_csv(f'{file_path}/mask_wearing_training_codes.csv')
    codes_round_df = pd.read_csv(f'{file_path}/mask_wearing_codes.csv')

    if 'Unnamed: 0' in training_round_df.columns:
      training_round_df.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Unnamed: 0' in codes_round_df.columns:
      codes_round_df.drop(columns=['Unnamed: 0'], inplace=True)
    
    training_dfs.append(training_round_df)
    code_dfs.append(codes_round_df)

  training_combined_df = pd.concat(training_dfs)
  code_combined_df = pd.concat(code_dfs)
  return { 'training': training_combined_df, 'codes': code_combined_df } 

def mask_wearing_codes_df():
  combined_codes_df = combine_coding_rounds()
  mask_codes_df = combined_codes_df['codes']
  articles_db_df = pd.read_csv('./labeled-data/round4/articles_mask.csv')
  mask_codes_df = mask_codes_df.merge(articles_db_df[['id','native_id']], left_on='article_id',right_on='id', how='left')
  mask_codes_df.drop(columns=['id_y'], inplace=True)
  mask_codes_df.rename(columns={'id_x': 'id'}, inplace=True)
  return mask_codes_df

def mask_wearing_training_codes_df():
  combined_codes_df = combine_coding_rounds()
  training_codes_df = combined_codes_df['training']
  return training_codes_df

def articles_df():
  # Read in article data
  kos_vox_df = pd.read_csv('./news-data/df_csvs/kos-vox-w-article.csv')
  fox_nyt_breit_df = pd.read_csv('./news-data/df_csvs/covid-or-mask_w-article.csv', sep=MC_SEP)
  carlson_df = pd.read_csv('./news-data/df_csvs/carlson-mask.csv')
  carlson_df_with_dates = add_dates_to_opinion_transcripts(carlson_df)
  hannity_df = pd.read_csv('./news-data/df_csvs/hannity-mask.csv')
  hannity_df_with_dates = add_dates_to_opinion_transcripts(hannity_df)
  ingraham_df = pd.read_csv('./news-data/df_csvs/ingraham-mask.csv')
  ingraham_df_with_dates = add_dates_to_opinion_transcripts(ingraham_df)
  articles_df = pd.concat([kos_vox_df, fox_nyt_breit_df, carlson_df_with_dates, hannity_df_with_dates, ingraham_df_with_dates])
  articles_df.drop(columns=['Unnamed: 0','Unnamed: 0.1'], inplace=True)
  return articles_df

def high_quality_codes_across_categories(mask_wearing_df):
  article_ids = mask_wearing_df['article_id'].unique()
  high_quality_paragraphs = []
  for article_id in article_ids:
    rows_for_article_id = mask_wearing_df[mask_wearing_df['article_id'] == article_id]
    num_annotaters = len(rows_for_article_id['session_id'].unique())
    # If there are multiple annotaters, resolve by agreement
    if num_annotaters > 1:
      agreement = inter_rater_agreement_across_categories(mask_wearing_df, article_id)
      # Rated as moderate agreement from Landis & Koch 1977 (https://datatab.net/tutorial/fleiss-kappa)
      if agreement > 0.4:
        high_quality_paragraphs.append(article_id)
    # Otherwise, look at high confidences
    else:
      if rows_for_article_id['confidence'].mean() >= 5:
        high_quality_paragraphs.append(article_id)
  return high_quality_paragraphs

def high_quality_resolved_codes(mask_wearing_df):
  '''
  Select only high quality codes from the entire coded set, and then
  resolve articles with multiple coders.

  :param mask_wearing_df: A data frame of mask wearing codes combined
  across coding rounds.
  '''
  high_quality_ids = high_quality_codes_across_categories(mask_wearing_df)
  mask_wearing_hq_df = mask_wearing_df[mask_wearing_df['article_id'].isin(high_quality_ids)]
  mask_wearing_hq_df.drop(columns=['id'],inplace=True)
  resolved_codes = resolve_multiple_codes_per_paragraph(mask_wearing_hq_df)
  codes_without_multiples = mask_wearing_hq_df[~mask_wearing_hq_df['article_id'].isin(resolved_codes)]
  mask_wearing_hq_resolved_df = codes_without_multiples.copy()
  mask_wearing_hq_resolved_df.reset_index(inplace=True)
  mask_wearing_hq_resolved_df.drop(columns=['index'],inplace=True)
  for article_id,codes in resolved_codes.items():
    for attr,code in codes.items():
      # All the attributes we care about (datetime, native_id) are the same
      # for all, so we can fetch iloc[0]
      row_for_article_id = mask_wearing_hq_df[mask_wearing_hq_df['article_id']==article_id].iloc[0]
      # print(row_for_article_id['datetime'])
      # print(f'added row {article_id}: {attr}: {code}')
      # print(len(mask_wearing_hq_resolved_df))
      mask_wearing_hq_resolved_df.loc[len(mask_wearing_hq_resolved_df)] = [article_id, attr, code, -1, -1, row_for_article_id['datetime'], row_for_article_id['native_id']]
      # print(len(mask_wearing_hq_resolved_df))
  return mask_wearing_hq_resolved_df

def add_text_to_codes_df(mask_wearing_df,article_db_df):
  mask_wearing_with_text = mask_wearing_df.copy()
  texts = []
  for row in mask_wearing_df.iterrows():
    data = row[1]
    text = article_db_df[article_db_df['id']==data['article_id']].iloc[0]['post']
    texts.append(text)
  mask_wearing_with_text['text'] = texts
  return mask_wearing_with_text

def label_analysis():
  # Read in label data
  mask_training_codes_df = mask_wearing_training_codes_df()
  mask_codes_df = mask_wearing_codes_df()
  articles_db_df = pd.read_csv('./labeled-data/round4/articles_mask.csv')
  articles_all_df = articles_df()

  training_agreement_scores = label_training_agreement_analysis(mask_training_codes_df)
  date_range_results = label_date_range_analysis(mask_codes_df, articles_all_df)
  percent_articles_labeled = percent_paragraphs_labeled_for_labeled_stories(mask_codes_df, articles_db_df)

  return (training_agreement_scores, date_range_results, percent_articles_labeled)

##################
# PSEUDO-LABEL ANALYSIS
##################

def read_all_pseudo_label_data(path):
  attributes = ['mw_independence_authority', 'mw_comfort_breathe',
       'mw_independence_forced', 'mw_attention_trust',
       'mw_attention_uncomfortable', 'mw_appearance', 'mw_compensation',
       'mw_access_cost', 'mw_inconvenience_remember',
       'mw_inconvenience_hassle', 'mw_efficacy_health', 'mw_efficacy_eff',
       'mw_access_diff', 'mw_comfort_hot']
  data = { attr: read_pseudo_label_data(path, attr) for attr in attributes }
  return data

def read_pseudo_label_data(path, attribute):
  dfs = [ pd.read_json(f'{path}/{attribute}-{i}.json') for i in range(5) ]
  for df in dfs:
    df['article_id'] = df.index
  return dfs

def majority_vote_all_label_data(pseudo_label_data):
  return { attr: majority_vote_pseudo_labels(pseudo_label_data[attr]) for attr in pseudo_label_data }

def majority_vote_pseudo_labels(ps_dfs):
  label_weights = {}
  for article_id in ps_dfs[0]['article_id']:
    label_weight = {}
    for i in range(len(ps_dfs)):
      df = ps_dfs[i]
      row = df[df['article_id'] == article_id]
      pseudo_label = row['pseudo_label'].iloc[0]
      pseudo_weight = row['pseudo_weight'].iloc[0]
      if pseudo_label not in label_weight:
        label_weight[pseudo_label] = pseudo_weight
      else:
        # TODO: Not sure if this should be add or multiply
        label_weight[pseudo_label] += pseudo_weight
    label_weights[article_id] = label_weight
  # print(label_weights)
  final_labels = {}
  for article_id,label_weight in label_weights.items():
    max_label = max(label_weight, key=label_weight.get)
    final_labels[article_id] = (max_label, max(label_weight.values()))
  return final_labels

def code_format_pseudo_label_df(pseudo_labels):
  '''
  :param pseudo_labels: The result of running majority_vote_pseudo_labels --
  a dict of { attr: { article_id: code } }
  '''
  attributes = ['mw_independence_authority', 'mw_comfort_breathe',
       'mw_independence_forced', 'mw_attention_trust',
       'mw_attention_uncomfortable', 'mw_appearance', 'mw_compensation',
       'mw_access_cost', 'mw_inconvenience_remember',
       'mw_inconvenience_hassle', 'mw_efficacy_health', 'mw_efficacy_eff',
       'mw_access_diff', 'mw_comfort_hot']
  df = pd.DataFrame(columns=['article_id','attribute','code','confidence','session_id','datetime'])
  for article_id in pseudo_labels['mw_comfort_hot']:
    for attr in attributes:
      ps_entry = pseudo_labels[attr][article_id]
      df.loc[len(df)] = [article_id, attr, ps_entry[0], ps_entry[1], -1, -1]
  return df


##################
# OPINION ANALYSIS
##################

REP_MEDIA_OUTLETS = [BREITBART, FOX, TUCKER_CARLSON, LAURA_INGRAHAM, SEAN_HANNITY]
MOD_MEDIA_OUTLETS = [NYT, FOX]
DEM_MEDIA_OUTLETS = [NYT, VOX, DAILY_KOS]

REP_STARTING_OPINION = 43.12
MOD_STARTING_OPINION = 48.58
DEM_STARTING_OPINION = 65.34

OPINION_CHANGE_POLARITY_BY_ATTR = {
  'mw_comfort_breathe': { 0: -1, 1: 1, 2: 0 },
  'mw_comfort_hot': { 0: -1, 1: 1, 2: 0 },
  'mw_efficacy_health': { 0: -1, 1: 1, 2: 0 },
  'mw_efficacy_eff': { 0: -1, 1: 1, 2: 0 },
  'mw_access_diff': { 0: -1, 1: 1, 2: 0 },
  'mw_access_cost': { 0: -1, 1: 1, 2: 0 },
  'mw_compensation': { 0: -1, 1: 1, 2: 0 },
  'mw_inconvenience_remember': { 0: -1, 1: 1, 2: 0 },
  'mw_inconvenience_hassle': { 0: -1, 1: 1, 2: 0 },
  'mw_appearance': { 0: -1, 1: 1, 2: 0 },
  'mw_attention_trust': { 0: -1, 1: 1, 2: 0 },
  'mw_attention_uncomfortable': { 0: -1, 1: 1, 2: 0 },
  'mw_independence_forced': { 0: -1, 1: 1, 2: 0 },
  'mw_independence_authority': { 0: -1, 1: 1, 2: 0 },
}

def graph_opinion_timeseries(rep_timeseries, mod_timeseries, dem_timeseries):
  fig,ax = plt.subplots(figsize=(8,4))
  start_date = date(2020, 4, 6)
  end_date = date(2020, 6, 9)
  dates = list(daterange(start_date, end_date))

  ax.plot(dates, rep_timeseries, color='red')
  ax.plot(dates, mod_timeseries, color='gray')
  ax.plot(dates, dem_timeseries, color='blue')
  ax.set_xticks(dates)
  ax.set_xticklabels([f'{date.month}-{date.day}' for date in dates], rotation=-45, ha='left', fontsize=6)
  ax.set_xlabel('Date')
  ax.set_ylabel('% Support for Wearing Masks')

  plt.show()

# def opinion_change_naive_linear_for_media(article_codes)

def opinion_change_naive_linear(article_codes):
  '''
  :param article_codes: Attribute code pairs for a given article
  '''
  total_change = 0
  for attr, code in article_codes.items():
    total_change += OPINION_CHANGE_POLARITY_BY_ATTR[attr][code]
  return total_change

def article_belief_value(article_codes):
  pos_codes = 0
  neg_codes = 0
  for attr, code in article_codes.items():
    code_polarity = OPINION_CHANGE_POLARITY_BY_ATTR[attr][code]
    if code_polarity == 1:
      pos_codes += 1
    elif code_polarity == -1:
      neg_codes += 1
  return (3 + round(3 * ((pos_codes - neg_codes) / (1 + pos_codes + neg_codes))))

def curr_sigmoid_p(exponent, translation):
  '''
  A curried sigmoid function used to calculate probabilty of belief
  given a certain distance. This way, it is initialized to use exponent
  and translation, and can return a function that can be vectorized to
  apply with one param -- message_distance.

  :param exponent: An exponent factor in the sigmoid function.
  :param translation: A translation factor in the sigmoid function.
  '''
  return lambda message_distance: sigmoid_contagion_p(message_distance, exponent, translation)

def sigmoid_contagion_p(message_distance, exponent, translation):
  '''
  A sigmoid function to calcluate probability of belief in a given distance
  between beliefs, governed by a few parameters.
  '''
  return (1 / (1 + math.exp(exponent * (message_distance - translation))))

def opinion_change_dissonant(population_opinion, article_belief, beta_fn):
  '''
  :param population_opinion: A vector of belief values for a population
  :param article_belief: The belief value for the article being received
  :param beta_fn: A function that takes in two belief values and returns a
  probability of belief update.
  '''
  dists = abs(population_opinion - article_belief)
  beta_vectorized = np.vectorize(beta_fn)
  probabilities = beta_vectorized(dists)
  rands = np.random.rand(len(probabilities))
  adopters = probabilities >= rands
  same = ~adopters
  new_opinion = (adopters * article_belief) + (same * population_opinion)
  return new_opinion

def opinion_change_dissonant_by_media(dem_opinion, rep_opinion, mod_opinion, article_belief, beta_fn, media_id):
  next_rep = rep_opinion
  next_dem = dem_opinion
  next_mod = mod_opinion
  if media_id in REP_MEDIA_OUTLETS:
    next_rep = opinion_change_dissonant(rep_opinion, article_belief, beta_fn)
  if media_id in DEM_MEDIA_OUTLETS:
    next_dem = opinion_change_dissonant(dem_opinion, article_belief, beta_fn)
  if media_id in MOD_MEDIA_OUTLETS:
    next_mod = opinion_change_dissonant(mod_opinion, article_belief, beta_fn)
  return next_rep, next_mod, next_dem

OPINION_CHANGE_METHODS = {
  'NAIVE_LINEAR': 0,
  'DISSONANT': 1
}

def initialize_population_opinion(poll_percent):
  '''
  :param poll_percent: The percent of individuals who are represented in
  the poll as having the poll belief.

  Assumptions: Since this is using a 'belief' 0-6, we assume that anything
  >= 4 is believing; anything <= 3 is not counted in the poll percent
  '''
  poll_population = np.round(np.random.uniform(4, 6, poll_percent))
  nonpoll_population = np.round(np.random.uniform(0, 3, 100-poll_percent))
  return np.concatenate((poll_population, nonpoll_population))

def opinion_change_simulation_population(mask_wearing_codes, articles_df, opinion_change_method):
  # Seed initial data for 3 groups
  rep_opinion_timeseries = [initialize_population_opinion(round(REP_STARTING_OPINION))]
  mod_opinion_timeseries = [initialize_population_opinion(round(MOD_STARTING_OPINION))]
  dem_opinion_timeseries = [initialize_population_opinion(round(DEM_STARTING_OPINION))]

  # Progress through codes over time and increment based on codes
  articles_no_nan = articles_df.dropna(subset=['publish_date'])
  start_date = date(2020, 4, 6)
  end_date = date(2020, 6, 8)
  for day in daterange(start_date, end_date):
    print(f'Evaluating {day}')
    next_rep = rep_opinion_timeseries[-1]
    next_mod = mod_opinion_timeseries[-1]
    next_dem = dem_opinion_timeseries[-1]

    articles_for_date = articles_no_nan[articles_no_nan['publish_date'].str.contains(str(day))]['stories_id']
    codes_for_date = mask_wearing_codes[mask_wearing_codes['native_id'].isin(articles_for_date)]
    resolved_codes = resolve_multiple_codes_per_paragraph(codes_for_date)
    # print(f'multi-coded pars: {resolved_codes}')

    # Paragraphs w/ more than one code
    for article_id in resolved_codes:
      native_id = mask_wearing_codes[mask_wearing_codes['article_id'] == article_id]['native_id'].iloc[0]
      media_id = articles_df[articles_df['stories_id'] == native_id]['media_id'].iloc[0]
      if opinion_change_method == OPINION_CHANGE_METHODS['NAIVE_LINEAR']:
        print('not implemented')
      elif opinion_change_method == OPINION_CHANGE_METHODS['DISSONANT']:
        article_belief = article_belief_value(codes_for_article_pairs)
        dissonance_fn = curr_sigmoid_p(4, 1)
        next_rep, next_mod, next_dem = opinion_change_dissonant_by_media(next_dem, next_rep, next_mod, article_belief, dissonance_fn, media_id)
    
    # Non-multi-coded paragraphs
    single_coded_pars = codes_for_date[~codes_for_date['article_id'].isin(resolved_codes)]
    non_multicoded_ids = single_coded_pars['article_id'].unique()
    # print(f'single-coded pars: {single_coded_pars}')
    for article_id in non_multicoded_ids:
      native_id = mask_wearing_codes[mask_wearing_codes['article_id'] == article_id]['native_id'].iloc[0]
      media_id = articles_df[articles_df['stories_id'] == native_id]['media_id'].iloc[0]
      codes_for_article = single_coded_pars[single_coded_pars['article_id']==article_id]
      # Format: { attr: code }
      codes_for_article_pairs = { row[1]['attribute']: row[1]['code'] for row in codes_for_article.iterrows() }
      if opinion_change_method == OPINION_CHANGE_METHODS['NAIVE_LINEAR']:
        print('not implemented')
      elif opinion_change_method == OPINION_CHANGE_METHODS['DISSONANT']:
        article_belief = article_belief_value(codes_for_article_pairs)
        dissonance_fn = curr_sigmoid_p(4, 1)
        next_rep, next_mod, next_dem = opinion_change_dissonant_by_media(next_dem, next_rep, next_mod, article_belief, dissonance_fn, media_id)
    
    rep_opinion_timeseries.append(next_rep)
    mod_opinion_timeseries.append(next_mod)
    dem_opinion_timeseries.append(next_dem)

  # Return timeseries for each category
  poll_value = lambda population_belief: (population_belief >= 4).sum()
  rep_value_timeseries = np.array([ poll_value(population_belief) for population_belief in rep_opinion_timeseries ])
  mod_value_timeseries = np.array([ poll_value(population_belief) for population_belief in mod_opinion_timeseries ])
  dem_value_timeseries = np.array([ poll_value(population_belief) for population_belief in dem_opinion_timeseries ])
  return { 'rep': rep_value_timeseries, 'mod': mod_value_timeseries, 'dem': dem_value_timeseries }

def opinion_change_simulation_single_value(mask_wearing_codes, articles_df, opinion_change_method):
  # Seed initial data for 3 groups
  rep_opinion_timeseries = [REP_STARTING_OPINION]
  mod_opinion_timeseries = [MOD_STARTING_OPINION]
  dem_opinion_timeseries = [DEM_STARTING_OPINION]

  # Progress through codes over time and increment based on codes
  articles_no_nan = articles_df.dropna(subset=['publish_date'])
  start_date = date(2020, 4, 6)
  end_date = date(2020, 6, 8)
  for day in daterange(start_date, end_date):
    print(f'Evaluating {day}')
    next_rep = rep_opinion_timeseries[-1]
    next_mod = mod_opinion_timeseries[-1]
    next_dem = dem_opinion_timeseries[-1]

    rep_diff = 0
    mod_diff = 0
    dem_diff = 0

    articles_for_date = articles_no_nan[articles_no_nan['publish_date'].str.contains(str(day))]['stories_id']
    codes_for_date = mask_wearing_codes[mask_wearing_codes['native_id'].isin(articles_for_date)]
    resolved_codes = resolve_multiple_codes_per_paragraph(codes_for_date)
    # print(f'multi-coded pars: {resolved_codes}')

    # Paragraphs w/ more than one code
    for article_id in resolved_codes:
      native_id = mask_wearing_codes[mask_wearing_codes['article_id'] == article_id]['native_id'].iloc[0]
      media_id = articles_df[articles_df['stories_id'] == native_id]['media_id'].iloc[0]

      if opinion_change_method == OPINION_CHANGE_METHODS['NAIVE_LINEAR']:
        opinion_change = opinion_change_naive_linear(resolved_codes[article_id])
        if media_id in REP_MEDIA_OUTLETS:
          rep_diff += opinion_change
        if media_id in MOD_MEDIA_OUTLETS:
          mod_diff += opinion_change
        if media_id in DEM_MEDIA_OUTLETS:
          dem_diff += opinion_change
        # rep_diff, mod_diff, dem_diff = opinion_fn(resolved_codes[article_id], media_id)
      elif opinion_change_method == OPINION_CHANGE_METHODS['DISSONANT']:
        print('not implemented')
    
    # Non-multi-coded paragraphs
    single_coded_pars = codes_for_date[~codes_for_date['article_id'].isin(resolved_codes)]
    # print(f'single-coded pars: {single_coded_pars}')
    for row in single_coded_pars.iterrows():
      attr = row[1]['attribute']
      code = row[1]['code']
      native_id = row[1]['native_id']
      media_id = articles_df[articles_df['stories_id'] == native_id]['media_id'].iloc[0]
      opinion_change = OPINION_CHANGE_POLARITY_BY_ATTR[attr][code]
      if media_id in REP_MEDIA_OUTLETS:
        rep_diff += opinion_change
      if media_id in MOD_MEDIA_OUTLETS:
        mod_diff += opinion_change
      if media_id in DEM_MEDIA_OUTLETS:
        dem_diff += opinion_change
    
    # print(f'dem diff: {dem_diff}')

    next_rep += rep_diff
    next_mod += mod_diff
    next_dem += dem_diff
    
    rep_opinion_timeseries.append(next_rep)
    mod_opinion_timeseries.append(next_mod)
    dem_opinion_timeseries.append(next_dem)

  # Return timeseries for each category
  return { 'rep': rep_opinion_timeseries, 'mod': mod_opinion_timeseries, 'dem': dem_opinion_timeseries }

def resolve_multiple_codes_per_paragraph(mask_wearing_codes):
  '''
  This function resolves multiple annotations for paragraphs into one
  annotation. The method by which it does this is majority vote, with ties
  resolved by the total confidence score for the tied votes. In the case
  where confidence scores are equal, a random code is chosen.

  Returned is a dictionary of { article_id: { attribute: winning_code }}
  '''
  article_ids = mask_wearing_codes['article_id'].unique()
  session_ids_per_article = { article_id: len(mask_wearing_codes[mask_wearing_codes['article_id'] == article_id]['session_id'].unique()) for article_id in article_ids }
  pars_with_multiple_codes = [ article_id for article_id,num_sessions in session_ids_per_article.items() if num_sessions > 1 ]

  # Take majority vote
  attributes = mask_wearing_codes['attribute'].unique()
  labels = mask_wearing_codes['code'].unique()
  code_votes = { article_id: {
    attribute: {
      label: len(mask_wearing_codes[(mask_wearing_codes['article_id'] == article_id) & (mask_wearing_codes['attribute'] == attribute) & (mask_wearing_codes['code'] == label)]['session_id'].unique()) for label in labels
    } for attribute in attributes
  } for article_id in pars_with_multiple_codes }
  majority_vote_winners = { article_id: {
    attribute: [ key for key in code_votes[article_id][attribute] if code_votes[article_id][attribute][key] == max(code_votes[article_id][attribute].values()) ] for attribute in attributes
  } for article_id in code_votes }
  
  # Resolve ties
  for article_id in majority_vote_winners:
    votes_for_article = majority_vote_winners[article_id]
    for attribute in votes_for_article:
      votes = votes_for_article[attribute]
      if len(votes) > 1:
        # print(f'resolving with confidences {votes}')
        total_confidences_per_vote = { code: sum(mask_wearing_codes[(mask_wearing_codes['article_id'] == article_id) & (mask_wearing_codes['attribute'] == attribute) & (mask_wearing_codes['code'] == code)]['confidence']) for code in votes }
        # print(f'confidence scores: {total_confidences_per_vote}')
        max_votes_by_confidence = [ code for code in total_confidences_per_vote if total_confidences_per_vote[code] == max(total_confidences_per_vote.values()) ]

        # Check if there's still a tie
        if len(max_votes_by_confidence) > 1:
          # print('remaining tie resolved by random')
          majority_vote_winners[article_id][attribute] = random.choice(max_votes_by_confidence)
        else:
          # print(f'resolved as: {max_votes_by_confidence[0]}')
          majority_vote_winners[article_id][attribute] = max_votes_by_confidence[0]
      else:
        majority_vote_winners[article_id][attribute] = votes[0]

  return majority_vote_winners

def opinion_from_gallup_data():
  rep_opinion_timeseries = opinion_data_from_svg('./gallup-data/SVG/rep-opinion.svg', REP_STARTING_OPINION)
  mod_opinion_timeseries = opinion_data_from_svg('./gallup-data/SVG/mod-opinion.svg', MOD_STARTING_OPINION)
  dem_opinion_timeseries = opinion_data_from_svg('./gallup-data/SVG/dem-opinion.svg', DEM_STARTING_OPINION)
  return { 'rep': rep_opinion_timeseries, 'mod': mod_opinion_timeseries, 'dem': dem_opinion_timeseries }

def opinion_data_from_svg(path, starting_opinion):
  svg_y_points = process_graph_svg(path)
  starting_diff = starting_opinion - svg_y_points[0]
  opinion_data = np.array(svg_y_points) + starting_diff
  return opinion_data

def process_graph_svg(path):
  tree = ET.parse(path)
  points_str = tree.getroot().find('{http://www.w3.org/2000/svg}g').find('{http://www.w3.org/2000/svg}polyline').get('points')
  points = points_str.split(' ')
  points_matrix = np.matrix([ [float(points[i]), float(points[i+1])] for i in range(0, len(points), 2)])
  line_flipped = points_matrix * np.matrix([[1, 0],[0, -1]])
  y_points = [ el[1] for el in line_flipped ]
  return y_points
