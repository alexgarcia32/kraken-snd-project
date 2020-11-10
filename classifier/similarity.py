import numpy as np
from collections import OrderedDict
import pandas as pd
import operator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


############################################################
#######             PREVIOUS FUNCTIONS               #######
############################################################

## Similarity with respect the count in each list
# option= "Dice" or "Jaccard"
def similiraty_coincidence(dict1, dict2, option,Dice_intersection__intensity):
    list1 = list(dict1.keys())
    list2 = list(dict2.keys())
    keys_intersection = list(set(list1).intersection(set(list2)))
    intersection = len(keys_intersection)
    union = len(set(list1).union(set(list2)))
    if option.lower() == "jaccard":
        results= float(intersection / union), keys_intersection
    elif option.lower() == "dice":
        results= float(Dice_intersection__intensity* intersection / (union + (Dice_intersection__intensity-1)*intersection)), keys_intersection
    elif option.lower() == "contained":
        results= float(intersection / min([len(list1), len(list2)])), keys_intersection
    else:
        raise ValueError("Please enter a correct similarity measure: Dice, Jaccard, contained")
    return results


## Function to adapt the dictionary to the fixed format used in the following functions
# The way of adaptation is different if the dictionary contains Entities or Related words
def adapt_dictionary(dictionary):
    ## Initialise empty dictionary to fill it with the adapted form
    dict_adapted = dict()
    ## If the dictionary is of Entities
    if isinstance(dictionary[next(iter(dictionary))],dict):
        for k, v in dictionary.items():
            dict_adapted[k] = v["FREQ"]
    ## If the dictionary is of Related Eords
    else:
        for k, v in dictionary.items():
            dict_adapted[k] = v[1]
    return (dict_adapted)


## Function to adjust two dictionarities with a matched set of their keys.
# The input parameters dict1 and dict1 are in the adapted form needed for the function
# The parameter mode is one of "union" to adjust each dictionary with all the keys in the
# union set (new keys will have an assigned absolute frecuency value of 0); or "intersection"
# to adjust each dictionary with all the keys in the intersection set (same absolute frecuency
# value as in the original one)
def match_dictionaries_keys(dict1, dict2, mode):
    ## Getting the list of values from the dictionaries
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())
    if mode.lower() == "union":
        keys_union = list(set(keys1).union(set(keys2)))
        ## Completing dict1 with he keys in dict2 that are not in dict1
        dict1_res = {key: dict1.get(key, 0) for key in keys_union}
        ## Completing dict2 with he keys in dict1 that are not in dict2
        # Same value for all: [0]
        dict2_res = {key: dict2.get(key, 0) for key in keys_union}
    elif mode.lower() == "intersection":
        keys_intersection = list(set(keys1).intersection(set(keys2)))
        dict1_res = {key: dict1[key] for key in keys_intersection}
        dict2_res = {key: dict2[key] for key in keys_intersection}
    else:
        raise ValueError("Please enter a correct mode option: union or intersection")
    return dict1_res, dict2_res


## Function to get the relative frecuency of the items of a dictionary,
# from their count frecuency
## The input parameter dictionary is in the adapted form needed for the function
def get_relative_frecuency(dictionary):
    ## Deepcopy of the input parameter to avoid changes in the original dictionary
    dict_aux = dict(dictionary)
    ## Computing the total count of occurrences
    list_values = list(dict_aux.values())
    total_count = np.sum(list_values)
    ## Computing the relative frecuency for each key
    for k, v in dict_aux.items():
        w = v / total_count
        dict_aux[k] = [v, w]
    return dict_aux


## Function to compute the similarity between tho arrays of relative frecuencies
def similarity_rfs(rf1, rf2, optionSimbSim, gamma=None):
    ## Pasamos rf1 y rf2  a array
    rf1 = np.array(rf1)
    rf2 = np.array(rf2)
    if optionSimbSim.lower() == "max":
        similarity= 1 - max(abs(rf1 - rf2))
    elif optionSimbSim.lower() == "gowda_diday":
        if np.sum([rf1, rf2]) == 0:
            similarity= 0
        else:
            numD1 = np.sum(abs(rf1 - rf2))
            denD1 = np.sum(np.maximum(rf1, rf2))
            D1 = numD1 / denD1
            numD2 = np.sum(rf1 + rf2 - np.minimum(rf1, rf2))
            denD2 = np.sum(rf1 + rf2)
            D2 = numD2 / denD2
            similarity= 1 - (D1 + D2) / 2

    elif optionSimbSim.lower() == "ichino_yaguchi":
        if (gamma is None) or not (0 <= gamma <= 0.5):
            raise ValueError("Please enter a value for gamma between 0 and 0.5")
        else:
            dis = np.sum(
                np.maximum(rf1, rf2) - np.minimum(rf1, rf2) + gamma * (2 * np.minimum(rf1, rf2) - rf1 - rf2)) / (
                              2 - 2 * gamma)
        similarity = 1 - dis

    else:
        raise ValueError("Please enter a correct similarity measure: max, gowda_diday or ichino_yaguchi")
    return similarity



## Function to finally compute the similarity relating to the distribution (in terms of relative frecuencies)
# of the keys between two dictionaries.
def similarity_distribution(dict1, dict2, mode, optionSimbSim, gamma=None):
    ## First we adapt the two dictionaries to the fixed format needed
    adapted_dict1 = adapt_dictionary(dict1)
    adapted_dict2 = adapt_dictionary(dict2)

    ## In order to compare the "bar charts" generated by dict1 and dict 2 (x asis: keys, y asis: relative frecuency),
    # we obtain the matched dictionaries from adapted_dict1 and adapted_dict2 with the union or intersection of their
    # keys
    dict1_aux, dict2_aux = match_dictionaries_keys(adapted_dict1, adapted_dict2, mode)

    ## Then we obtain the relative frecuency of words in dict1_aux and dict2_aux
    dict1_aux = get_relative_frecuency(dict1_aux)
    dict2_aux = get_relative_frecuency(dict2_aux)

    ## We sort both dictonarities by key alphabetically to compare index by index the list of relative frecuencies
    sorted_dict1_aux = OrderedDict(sorted(dict1_aux.items()))
    sorted_dict2_aux = OrderedDict(sorted(dict2_aux.items()))

    ## Obtaining the relative frecuencies of each of the sorted dictionary
    rf1 = [v[1] for v in list(sorted_dict1_aux.values())]
    # list(map(operator.itemgetter(1), list(sorted_dict1_aux.values())))
    rf2 = [v[1] for v in list(sorted_dict2_aux.values())]
    # list(map(operator.itemgetter(1), list(sorted_dict2_aux.values())))

    ## Disimilarity measure between rf1 and rf2
    similarity = similarity_rfs(rf1, rf2, optionSimbSim, gamma)
    return similarity


## Function to compute the "bar chart" as a whole of the common keys of a dictionarity
# by means of the count frecuencies
def common_bar_chart(dict1, dict2):
    ## First we adapt the two dictionaries to the fixed format needed
    adapted_dict1 = adapt_dictionary(dict1)
    adapted_dict2 = adapt_dictionary(dict2)
    ## We obtain the common keys in dict1 and dict2
    list1 = list(dict1.keys())
    list2 = list(dict2.keys())
    keys_intersection = list(set(list1).intersection(set(list2)))
    ## Create the common dictionary with the keys in the intersection and value is the sum values of dict1 and dict2
    common_dict = {key: (adapted_dict1[key] + adapted_dict2[key]) for key in keys_intersection}
    common_dict_rf = get_relative_frecuency(common_dict)
    return common_dict_rf


## Function to compute the average sentiment of the dictionary of one Entiny Name
def average_sentiment_EN(dictionary):
    ## First we adapt the dictionary to the format needed in the defined functions
    dictionary_adapted = adapt_dictionary(dictionary)
    ## Now we obtain the relative frecuencies of the keys
    dictionary_adapted_rf = get_relative_frecuency(dictionary_adapted)
    ## We extract those relative frecuencies
    rf = list(map(operator.itemgetter(1), list(dictionary_adapted_rf.values())))
    ## We extract the sentiments of the keys from the dictionary
    sents = list(map(operator.itemgetter(0), list(dictionary.values())))
    ## We compute the weighted average of sentiments
    average_sentiment = np.sum(np.array(rf) * np.array(sents))
    return average_sentiment


## Function to compute the average sentiment of the dictionary of the whole news
# Dictionary is the dictionary of a news
def average_sentiment_news(dictionary):
    ## Initialising empty list to store list of sentiments and frecuency along the whole text
    sents = []
    freq = []
    ## Itering through all the Entities in the dictionary
    for k_EN in list(dictionary):
        values_dictionary_EN_k = list(dictionary[k_EN]["RWORDS"].values())
        sents += [v[0] for v in values_dictionary_EN_k]
        freq += [v[1] for v in values_dictionary_EN_k]
    ## sents and freq to array
    sents = np.array(sents)
    freq = np.array(freq)
    ## Computing average sentiment
    average_sentiment = np.sum(sents * freq) / np.sum(freq)
    return (average_sentiment)


################################################################
#######        COMPUTING THE SIMILARITY MATRIX          ########
################################################################


def similarity(dictionary, component_selector, optionSimbSim,Dice_intersection__intensity, gamma):
    ## Checking if the length of parameter component_selector is correct
    if (len(component_selector) != 7):
        raise ValueError("The parameter component_selector must be of length 7")

    ## length of dictionary and news contained in it
    len_dictionary = len(dictionary)
    news_dictionary = list(dictionary.keys())

    ## We initialize the similarity matrix
    sim_matrix = np.zeros((len_dictionary, len_dictionary))

    ## We browse through all the news in (dictionary) and compare them peer to peer
    for i_news in range(len_dictionary):
        dict_news_i = dictionary[news_dictionary[i_news]]["ENs"]
        for j_news in np.arange(i_news + 1, len_dictionary):
            dict_news_j = dictionary[news_dictionary[j_news]]["ENs"]

            ##### We compute the first component of the similarity #####
            similarity_component1, ENs_keys_intersection = similiraty_coincidence(dict_news_i, dict_news_j, "Dice",Dice_intersection__intensity)

            ## If the news_i and news_j have any Entity Name in common ( ie, similarity_component1!=0,
            # then we compute the rest of the components of the proposed similarity.
            if similarity_component1 != 0:

                ##### We compute the second component of the similarity #####
                similarity_component2 = similarity_distribution(dict_news_i, dict_news_j, "intersection", optionSimbSim,
                                                                gamma)

                ##### We compute the seventh component of the similarity (for the simple version) #####
                average_sentiment_news_i = average_sentiment_news(dict_news_i)
                average_sentiment_news_j = average_sentiment_news(dict_news_j)
                similarity_component7 = 1 - abs(average_sentiment_news_i - average_sentiment_news_j) / 2

                ##### We compute the third, fourh and fith components of the proposed similarity #####
                ## We obtain the total "bar chart" of the common Entities in common
                # The weights to combine the similarity in each Entity Name in common
                # are its relative frecuency
                common_dict_news_i_j = common_bar_chart(dict_news_i, dict_news_j)
                ## We iterate through the common entity names in both news and compute the similarity in terms of
                # each one of the components proposed. Then, for each Entity name there is a similarity value (for every
                # component. Combination of them through the relative frecuencies of the common "bar chart".
                similarity_component3 = 0
                similarity_component4 = 0
                similarity_component5 = 0
                similarity_component6 = 0
                for k_EN in ENs_keys_intersection:
                    ## We obtain the dictionaries of the k Entity in the news i and j
                    dict_news_i_EN_k = dict_news_i[k_EN]["RWORDS"]
                    dict_news_j_EN_k = dict_news_j[k_EN]["RWORDS"]

                    ##### We compute the third component of the similarity in the Entity Name k #####
                    similarity_component3_EN_k, RWORDs_keys_intersection = similiraty_coincidence(dict_news_i_EN_k,
                                                                                                  dict_news_j_EN_k,
                                                                                                  "Dice",Dice_intersection__intensity)

                    ##### We compute the fourth component of the similarity in the Entity Name k  #####
                    ## If the news_i and news_j have in the Entity Name k have any Rword in common ( ie,
                    # similarity_component3_EN_k!=0, then we compute the similarity_component4_EN_k. Otherwise,
                    # similarity_component4_EN_k=0.
                    similarity_component4_EN_k = (similarity_component3_EN_k != 0) * similarity_distribution(
                        dict_news_i_EN_k,
                        dict_news_j_EN_k, "union",
                        optionSimbSim, gamma)

                    ##### We compute the fifth component of the similarity in the Entity Name k  #####
                    sent_dict_news_i_EN_k = average_sentiment_EN(dict_news_i_EN_k)
                    sent_dict_news_j_EN_k = average_sentiment_EN(dict_news_j_EN_k)
                    similarity_component5_EN_k = 1 - abs(sent_dict_news_i_EN_k - sent_dict_news_j_EN_k) / 2

                    ##### We compute the sixth component of the similarity in the Entity Name k  #####
                    ## If the news_i and news_j have in the Entity Name k have any Rword in common ( ie,
                    # similarity_component3_EN_k!=0), then we compute the sixth component.
                    if similarity_component3_EN_k > 0:

                        ## We obtain the total "bar chart" of the common Rwords in the Entity Name k
                        # The weights to combine the similarity in each Rword in common are its relative frecuency
                        common_dict_news_i_j_EN_k = common_bar_chart(dict_news_i_EN_k, dict_news_j_EN_k)
                        weights_EN_k_Rwords = [v[1] for v in list(common_dict_news_i_j_EN_k.values())]

                        ## We obtain the list of sentiment values in dict_news_i_EN_k and dict_news_j_EN_k for the common
                        # Rwords, that is, for the Rwords in RWORDs_keys_intersection
                        list_sents_dict_news_i_EN_k_Rwords = [dict_news_i_EN_k[w_Rword][0] for w_Rword in
                                                              RWORDs_keys_intersection]
                        list_sents_dict_news_j_EN_k_Rwords = [dict_news_j_EN_k[w_Rword][0] for w_Rword in
                                                              RWORDs_keys_intersection]

                        ## We obtain the list of similarity in sentimens in each Rword in the intersection
                        sim_sent_dict_news_i_j_EN_k_Rwords = [1 - abs(x - y) / 2 for x, y in
                                                              zip(list_sents_dict_news_i_EN_k_Rwords,
                                                                  list_sents_dict_news_j_EN_k_Rwords)]

                        ## The sixth similarity component in the Entity Name k is the average value of the sentiments'
                        # similarity, wwighted by the relative frecuency os the corresponding Rword in the common bar chart.
                        # Combination of them through the relative frecuencies of the common "bar chart".
                        similarity_component6_EN_k = np.sum(np.array(weights_EN_k_Rwords) *
                                                            np.array(sim_sent_dict_news_i_j_EN_k_Rwords))
                    else:
                        similarity_component6_EN_k = 0

                    ## We add these values corresponding to the Entity Name k to obtain the total similarities components
                    weight_EN_k = common_dict_news_i_j[k_EN][1]
                    similarity_component3 += weight_EN_k * similarity_component3_EN_k
                    similarity_component4 += weight_EN_k * similarity_component4_EN_k
                    similarity_component5 += weight_EN_k * similarity_component5_EN_k
                    similarity_component6 += weight_EN_k * similarity_component6_EN_k


            ## If the news_i and news_j have no Entity Name in common ( ie, similarity_component1==0,
            # then we the rest of the components of the proposed similarity is 0.
            else:
                similarity_component2 = 0
                similarity_component3 = 0
                similarity_component4 = 0
                similarity_component5 = 0
                similarity_component6 = 0
                similarity_component7 = 0

            ## Combination of every component
            total_weight_similarity_components = np.sum(component_selector)
            w_similarity_components = np.array(component_selector) / total_weight_similarity_components
            sim_matrix[i_news, j_news] = np.sum(w_similarity_components * np.array([similarity_component1,
                                                                                    similarity_component2,
                                                                                    similarity_component3,
                                                                                    similarity_component4,
                                                                                    similarity_component5,
                                                                                    similarity_component6,
                                                                                    similarity_component7]))
            sim_matrix[j_news, i_news] = sim_matrix[i_news, j_news]
        ## Completig the diagonal with 1s
        sim_matrix[i_news, i_news] = 1

    ## Imputing values -2.220446049250313e-16 with a 0
    sim_matrix[sim_matrix<0]=0

    dis_matrix = 1 - sim_matrix
    ## Imputing values -2.220446049250313e-16 with a 0
    dis_matrix[dis_matrix<0]= 0

    return dis_matrix


def export_csv(matrix, date):
    nombre_fichero_export_media = "MatrizMetrica_Dia_" + str(date) + ".csv"
    DF_matrix = pd.DataFrame(matrix)
    path = os.getcwd() + "\\data\\" + nombre_fichero_export_media
    DF_matrix.to_csv(path, sep=";", decimal=',')
    return True