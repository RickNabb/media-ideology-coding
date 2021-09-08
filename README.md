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

### Analyzing Data

Once you have set up your environment, you can run the following commands to load a data frame of media data to test:

```python
from data_collector import *
df = pd.read_csv('news-data/[choose your file].csv', sep=MC_SEP)
parse_mc_article_html(df)
```

After these commands, you should have a `pandas` dataframe containing raw text of whichever media articles you read in from your `.csv` file. Perhaps your dataframe looks like this:

```python
     Unnamed: 0  stories_id  publish_date  ...  language                                              title                                   article_data_raw
0             2      771438           NaN  ...        en  Tucker: News organizations took money from the...  'Tucker Carlson Tonight' host calls out media ...
1             5      821940           NaN  ...        en  Tucker: The mainstream media's job is to defen...  'Tucker Carlson Tonight' host asks what do civ...
2             6      594061           NaN  ...        en  Tucker Carlson: Science is a seeking of the tr...  'Tucker Carlson Tonight' host asks what do civ...
3             7      308386           NaN  ...        en  Tucker Carlson: Democrats and the CDC have bee...   Fox News host weighs in on mask recommendatio...
4             8      369627           NaN  ...        en  Tucker: Democrats rode virus panic all the way...  'Tucker Carlson Tonight' host says they are be...
..          ...         ...           ...  ...       ...                                                ...                                                ...
170         454      887534           NaN  ...        en  Adam Carolla: California doesn't care about th...  Democrats in disarray ahead of New Hampshire p...
171         460      898262           NaN  ...        en  Victor Davis Hanson says Bernie Sanders' surge...  Democratic presidential candidate Bernie Sande...
172         465      583121           NaN  ...        en  Sen. Hawley: We're watching the Democrats' cas...  Three Democrats could vote to acquit President...
173         507      356250           NaN  ...        en  Tucker Carlson: After 3 years of screeching ab...  House Speaker Nancy Pelosi insists Democrats t...
174         508      569613           NaN  ...        en  Jeff Sessions questions why Democrats voted fo...  There's zero chance the Senate will convict Pr...

[175 rows x 10 columns]
```

Now you can interact with the article data contained in the dataframe. For example, you could split raw article text into paragraphs which contain a keyword ("mask") for later analysis:

```python
text = df.iloc[0].article_data_raw
pars = get_keyword_paragraph(text, 'mask', 1)
print(pars)
```