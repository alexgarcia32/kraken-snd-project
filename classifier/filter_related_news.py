import numpy as np
from collections import OrderedDict
import os
import pandas as pd
#from lectura import df, date
import operator
import copy


#############################################################################################################
#    We create a filter to select those news (from the knowledge graph) that have at least x EN in common   #
#    with a given fake news                                                                                 #
#############################################################################################################


def knowledge_filtered_fake(df,min_common_en,fake_news_dict):
    # df: Original knowledge graph
    # min_common_en : min number of EN in common we want
    # fake_news_dict: fake news graph

    # Copy of the original dictionary
    knowledge_filtered = copy.deepcopy(df)
    # ENs of the fake news
    fake_news_en = fake_news_dict["ENs"].keys()

    for news in list(df.keys()):  # for news in knowledge graph
        news_en = df[news]["ENs"].keys()
        # Intersection
        intersection = set(fake_news_en) & set(news_en)
        common_en = len(intersection)
        # We delete news satisfying the following condition
        if common_en < min_common_en:
            del knowledge_filtered[news]
    # We decide if knowledge filtered is big enough
    size_df = len(df)
    size_kf = len(knowledge_filtered)
    error=0
    if ((min_common_en == 1) and (size_kf < 1)): ## 0.6*size_df
        error=1
        print("There is not enough information to determine credibility")
    elif ((min_common_en == 2) and (size_kf <0*size_df )): ##0.4*size_df
        error=1
        print("There is not enough information to determine credibility")
    elif ((min_common_en == 3) and (size_kf < 0*size_df)): ## 0.33*size_df
        error=1
        print("There is not enough information to determine credibility")
    elif (min_common_en > 3):
        error=1
        raise ValueError("Recommended min_common_en is between 1 and 3")

    return knowledge_filtered,  error