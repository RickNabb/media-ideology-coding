#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:58:18 2021

@author: amyowens
"""

import pandas as pd
import os

# 10166101979250354

#foxNews = pd.read_json("news-data/fb-posts/FoxNews.json")
#foxNews.rename(columns = {"post_id":"native_id"},inplace=True)

result = pd.DataFrame()

print('Reading in fb post data')
for filename in os.listdir("news-data/fb-posts/"):
    if '.json' in filename:
        file = filename.split(".")
        file = file[0]
        filename = "./news-data/fb-posts/"+filename
        print(filename)
        data = pd.read_json(filename)
        data.rename(columns = {"post_id":"native_id"},inplace=True)
        if "native_id" in data.columns:
            result = result.append(data)

result = result.set_index('native_id')

print('Reading in label data')
for filename in os.listdir("labeled-data"):
    file = filename.split(".")
    file = file[0]
    filename = "./labeled-data/"+filename
    print(filename)
    data = pd.read_json(filename)
    data.rename(columns={"code":file}, inplace=True)
    if "native_id" in data.columns:
        result = result.join(data.set_index('native_id'))

#result = pd.concat([df.set_index("native_id") for df in dataframes], ignore_index=False)

"""
    try:
        newPD = pd.merge(foxNews[["native_id","text"]], data, on="native_id", how="left")
        foxNews=newPD
        print(foxNews)
    except KeyError:
        pass
    """

