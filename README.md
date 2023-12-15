# Media Coding Project

### Author: Nick Rabb (nicholas.rabb@tufts.edu)

## Table of Contents

* [Getting Started](#getting-started)

-----------

## Getting Started

Here are some steps to get the project running in a Python environment.

### Prerequisites

Before you run the project, you should have installed:

1. Python 3
2. Anaconda (for environment management)

### First Run

1. Open an anaconda terminal and navigate to this directory
2. Run the command: ```$ activate covid_misinfo_env```
3. Open some sort of Python terminal (I use iPython)

## Analyzing Data

Public data supporting this project is available through the Harvard Dataverse at [URL TBD].

Running the following commands will load the data used for this project and analyses based on that data:

```py
from data_analysis import *
mask_wearing_df = pd.read_csv('./<your_directory>/annotations.csv')
article_df = pd.read_csv('./<your_directory>/articles.csv')
paragraph_df = pd.read_csv('./<your_directory>/paragraphs.csv')
```

To filter annotations to high-quality annotations only, run the following command:

```py
hq_mask_wearing_df = high_quality_resolved_codes(mask_wearing_df)
```

## Data Descriptions

### `annotations.csv`
Contains the crowd-sourced annotations given by human labelers, including the paragraph ID of the paragraph labeled, the time it was labeled, the labeler's unique session ID, and the ID of the source article

Specific meaning of columns in the data file:
* `paragraph_id`: The ID of the paragraph being labeled. This references the ID column in `paragraphs.csv`.
* `attribute`: The name of the attribute from Howard 2020's Face Mask Perceptions Scale that is being coded for. Each shorthand name here corresponds to 1 of 14 attributes we asked participants to code for. Shorthand names are as follows:
  * `mw_comfort_breathe`: 'It is difficult to breathe when wearing a face mask'
  * `mw_comfort_hot`: 'Face masks get too hot'
  * `mw_independence_authority`: 'People want to prove a point against authority'
  * `mw_independence_forced`: 'People do not like feeling forced to do something'
  * `mw_attention_trust`: 'Face masks make people seem untrustworthy'
  * `mw_attention_uncomfortable`: 'Face masks make other people uncomfortable'
  * `mw_appearance`: 'Face masks look ugly or weird'
  * `mw_compensation`: 'You can simply stay away from people when you go out'
  * `mw_access_cost`: 'Face masks are too expensive'
  * `mw_access_diff`: 'It is difficult to get a face mask'
  * `mw_inconvenience_remember`: 'People do not like remembering to wear a face mask'
  * `mw_inconvenience_hassle`: 'Wearing a face mask is too much of a hassle'
  * `mw_efficacy_health`: 'Face masks provide health benefits'
  * `mw_efficacy_eff`: 'Face masks are effective'
* `code`: The annotation assigned to this attribute, where 0 is the anti-mask perspective (e.g., for `mw_comfort_hot` that face masks do get too hot), 1 is the pro-mask perspective (e.g., for `mw_comfort_hot` that face masks do NOT get too hot), and 2 is 'does not mention'.
* `confidence`: A confidence score in the annotation from 1-7 where 1 is very unconfident and 7 is very confident.
* `session_id`: A unique browser session ID assigned to the annotator so they could maintain their session if their browser refreshed or they lost internet connection. Some annotators who performed the task multiple times received a new session ID because they browser cleared local storage, so this is NOT a unique participant identifier.
* `datetime`: The date and time that the annotation was assigned. Note that some rows do not have a datetime because they were not tracked in early annotation rounds.
* `native_id`: The ID of the article that this paragraph came from, which matches with `stories_id` from `articles.csv`.

### `articles.csv`
Contains data from MediaCloud and other scraped sources for all news articles relevant to this data set. Some data that it includes is the article ID, media ID of the publisher, the full article text, URL of the article, and publication date. Much of this data comes directly from the MediaCloud platform.

Specific meaning of columns in the data file:
* `ap_syndicated`: Whether or not the article was gathered through the Associated Press.
* `collect_date`: The date when the article was collected into MediaCloud.
* `feeds`: Any MediaCloud feeds this article is a part of.
* `guid`: Usually the URL of the story as a unique identifier.
* `language`: The language the story is in.
* `media_id`: The ID of the media who published this; matches with `media_id` in the `media_accounts.csv` file.
* `media_name`: The name of the media who published this.
* `media_url`: The general domain URL of the media who published this.
* `metadata`: A JSON dictionary of metadata MediaCloud collected on the story.
* `processed_stories_id`: An ID in long format.
* `publish_date`: The date this story was published.
* `stories_id`: A unique ID for this story; matches with `native_id` in `annotations.csv` and `paragraphs.csv`.
* `story_tags`: A list of any tags assigned to the story.
* `title`: The title of the story.
* `url`: The URL of the story.
* `word_count`: A word count for the story.
* `article_data_raw`: The text content of the story, without HTML data, separated by the newline character `\n` after being parsed with a Natural Language Processing algorithm to detect paragraphs.
* `themes`: A list of themes in the story.
* `article_data`: If populated, the full HTML of the story page.

### `paragraphs.csv`
Contains each paragraph from the data set with a unique ID, its text, the ID of the article it came from, and the media account id (referenced by `account_id' in the media_accounts.csv file) of the publisher

Specific meaning of columns in the data file:
* `id`: The unique ID of this paragraph.
* `paragraph_text`: The text of the paragraph.
* `native_id`: The ID of the article this came from; matches `stories_id` from `articles.csv`.
* `article_account_id`: The ID of the media account that published this; matches the `account_id` column of `media_accounts.csv`

### `media_accounts.csv`
Contains a short list of media accounts in this data set, including two types of ID used for each account, and a name.

Specific meaning of columns in the data file:
* `name`: The name of the media account.
* `account_id`: One ID of this account; matches `article_account_id` from `paragraphs.csv`.
* `media_id`: One ID of the account; matches `media_id` from `articles.csv`.

## Analysis functions supporting article analyses

TBD