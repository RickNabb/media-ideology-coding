import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from data_collector import NYT, FOX, TUCKER_CARLSON, SEAN_HANNITY, LAURA_INGRAHAM, HUFF_POST, BREITBART, DAILY_KOS, VOX, media_id_to_name, MC_SEP

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
      agreement_vector = np.array([ agreement_codes[article_id][row[1]['attribute'].replace('_training','')] == row[1]['code'] for row in session_article_codes.iterrows() ], dtype=int)
      agreement_scores[session_id][article_id] = agreement_vector.sum() / np.ones(len(agreement_vector)).sum()

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
  ax.set_ylabel('Number of labels per outlet (without "Does not mention")')
  ax.legend()

  plt.show()

def graph_media_outlet_total_distribution(label_date_range_results):
  outlets = [FOX, NYT, BREITBART, VOX, DAILY_KOS]

  fig,ax = plt.subplots(figsize=(8,4))

  labels_per_outlet = [ sum(label_date_range_results[media_id_to_name[outlet]]) for outlet in outlets ]
  ax.bar([ media_id_to_name[outlet] for outlet in outlets ], labels_per_outlet)

  ax.set_xticklabels([media_id_to_name[outlet] for outlet in outlets], fontsize=8)
  ax.set_xlabel('Media Outlet')
  ax.set_ylabel('Number of labels per outlet (without "Does not mention")')

  plt.show()


def label_analysis():
  # Read in label data
  mask_codes_df = pd.read_csv('./labeled-data/round1/mask_wearing_codes.csv')
  mask_training_codes_df = pd.read_csv('./labeled-data/round1/mask_wearing_training_codes.csv')
  articles_db_df = pd.read_csv('./labeled-data/round1/articles_mask.csv')
  mask_codes_df = mask_codes_df.merge(articles_db_df[['id','native_id']], left_on='article_id',right_on='id')
  mask_codes_df.drop(columns=['id_y'], inplace=True)
  mask_codes_df.rename(columns={'id_x': 'id'}, inplace=True)
  
  # Read in article data
  kos_vox_df = pd.read_csv('./news-data/df_csvs/kos-vox-w-article.csv')
  fox_nyt_breit_df = pd.read_csv('./news-data/df_csvs/covid-or-mask_w-article.csv', sep=MC_SEP)
  articles_df = pd.concat([kos_vox_df, fox_nyt_breit_df])

  training_agreement_scores = label_training_agreement_analysis(mask_training_codes_df)
  date_range_results = label_date_range_analysis(mask_codes_df, articles_df)
  percent_articles_labeled = percent_paragraphs_labeled_for_labeled_stories(mask_codes_df, articles_df)

  return (training_agreement_scores, date_range_results, percent_articles_labeled)