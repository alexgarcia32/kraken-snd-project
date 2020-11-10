######################################################################################################
######################################################################################################
########      KRAKEN-FND EXPERIMENT ON CELEBRIIES DATA SET: MEAN OF RESULTS BY CELEBRITY      ########
######################################################################################################
######################################################################################################


import os
import sys
import pandas as pd

sys.path.append(os.path.abspath('../kraken-snd/knowledge-graph-builder/'))
sys.path.append(os.path.abspath('../kraken-snd/classifier/'))
from classifier.test_news import read_directory_dictionary, testing_news, measures_differents_threshold, measures_selected_threshold
import random

###########################################################################
########          Reading and exploring the dictionary             ########
###########################################################################

path = '../kraken-snd/classifier/experiments/overall/celebrityDataset/celebrity_nuevo'
celebrity_dictionary=read_directory_dictionary(path)

## Number of true news in the dictionary
n_true_news=len(celebrity_dictionary[0])

## Number of fake news in the dictionary
n_fake_news=len(celebrity_dictionary[1])



###################################################################################################
########          Creating the Knowledge-Graph from the celebrities dictionary             ########
###################################################################################################

# We select all the available news
d_true = { i : celebrity_dictionary[0][i] for i in range(0,n_true_news)} # 0 are true news
d_fake = { i : celebrity_dictionary[1][i] for i in range(0,n_fake_news)} # 1 are fake news



########################################################################
########          Creating testing news dictionaries            ########
########################################################################
# We use all the available true news to create the knowledge graph and the same news to test the true news
# due to the small sample size

## Random selection of fake news and true news to be tested
random.seed(2019)
# The fake testing news will also be all of the false news
fake_test_news=random.sample(range(0,n_fake_news), n_fake_news)
# The true testing news will also be all of the true news
true_test_news=random.sample(range(0,n_true_news), n_true_news)

## Selecting news to create the test news dictionaries
TRUE_test_news = list(celebrity_dictionary[0][i] for i in true_test_news)
FALSE_test_news = list(celebrity_dictionary[1][i] for i in fake_test_news)

###########################################################################################
########                          Exploring number of ENs                          ########
###########################################################################################

# Number of ENs appearing in true news
list_EN_true = []
for k,v in d_true.items():
    for i in d_true[k]["ENs"]:
        list_EN_true.append(i)

list_EN_true = list(set(list_EN_true)) # we get unique values
print(list_EN_true)
len(list_EN_true)

# We only want to study the celebrity ENs, not all the ENs
# We count the number of news in which each EN appears
# and select the minimum number of news we admit to select a celebrity
min_number_news = 5
list_EN_n = []
list_EN = []
n_times = 0
n_true_news = []
for i in range(0, len(list_EN_true) - 1):
    n_times = 0
    for k, v in d_true.items():
        if list_EN_true[i] in list(d_true[k]["ENs"]):
            n_times += 1
    if n_times >= min_number_news:
        list_EN.append(list_EN_true[i])
        n_true_news.append(n_times)
        #list_EN_n.append([list_EN_true[i], [n_times]])


index = list_EN
columns = ["n_true_news","n_fake_news","tp","fp","tn","fn","Precision","Recall","F1","Accuracy"]
df_celeb = pd.DataFrame(index=index, columns=columns)
df_celeb.index.name = "Celebrity"
df_celeb["n_true_news"] = n_true_news



## Number of fake news with the selected celebrity Entity Name
n_fake_news = []
for i in range(0, len(list_EN)):
    c_fake = 0
    for k,v in d_fake.items():
        if list_EN[i] in list(d_fake[k]["ENs"]):
            c_fake += 1
    #dic_result_celeb[list_EN[i]].append(c_fake)
    n_fake_news.append(c_fake)

df_celeb["n_fake_news"] = n_fake_news



## Fixing parameters
min_common_en = 1
component_selector = [1, 1, 1, 1, 1, 1, 0]
Dice_intersection__intensity = 4
### Loop for all celebrities

for i in range(0, len(list_EN)):
    print(i,list_EN[i])
    ### Filtering teh dictionaries by celebrity
    # Dictionary of true news filtered by celebrity
    d_celeb_true = dict()
    for k, v in d_true.items():
        for j in d_true[k]["ENs"]:
            if j == list_EN[i]:
                d_celeb_true[k] = v

    # Dictionary of fake news filtered by celebrity
    d_celeb_fake = dict()
    for k, v in d_fake.items():
        for j in d_fake[k]["ENs"]:
            if j == list_EN[i]:
                d_celeb_fake[k] = v

    ### Creating testing news dictionaries
    # The true tests news will be the ones that are in the knowledge graph
    TRUE_test_news = list(d_celeb_true.values())
    # The false tests news will be all in the dictionary of false news of Kim Kardashian
    FALSE_test_news = list(d_celeb_fake.values())

    ## Predictions
    threshold_prob_fake = 0  ## at the momment this parameter does not matter
    ## Obtaining probabilities for fake for the TRUE test news
    #min_common_en = 2 # we are more strict with true news
    fp, tn, predictions_Fake_True_news, predictions_Fake_value_True_news = testing_news(d_celeb_true, TRUE_test_news,
                                                                                        threshold_prob_fake,
                                                                                        min_common_en,
                                                                                        component_selector,
                                                                                        Dice_intersection__intensity)

    ## Obtaining probabilities for fake for the FALSE test news
    #min_common_en = 1
    tp, fn, predictions_Fake_False_news, predictions_Fake_value_False_news = testing_news(d_celeb_true, FALSE_test_news,
                                                                                          threshold_prob_fake,
                                                                                          min_common_en,
                                                                                          component_selector,
                                                                                          Dice_intersection__intensity)

    ## Measures for different threshold values
    list_results = measures_differents_threshold(predictions_Fake_value_True_news, predictions_Fake_value_False_news)
    list_threshold = list_results[0]
    list_accuracy = list_results[1]
    list_precision = list_results[2]
    list_recall = list_results[3]
    list_f1_score = list_results[4]
    list_fp_ratio = list_results[5]

    ## Finding Optimum threshold value only maximizing F1 score
    max_value = max(list_f1_score)
    max_index = list_f1_score.index(max_value)
    threshold_prob_fake = list_threshold[max_index]

    ## RESULTS FOR THE SELECTED FIXED THRESHOLD
    performance_measures, confusion_matrix = measures_selected_threshold(predictions_Fake_value_True_news,
                                                                         predictions_Fake_value_False_news,
                                                                         threshold_prob_fake)


    tn = confusion_matrix["Observed: NOT FAKE"][0]
    fp = confusion_matrix["Observed: NOT FAKE"][1]
    fn = confusion_matrix["Observed: FAKE"][0]
    tp = confusion_matrix["Observed: FAKE"][1]

    df_celeb.loc[list_EN[i], "tp"] = tp
    df_celeb.loc[list_EN[i], "fp"] = fp
    df_celeb.loc[list_EN[i], "tn"] = tn
    df_celeb.loc[list_EN[i], "fn"] = fn
    # Performance measures
    df_celeb.loc[list_EN[i], "Precision"] = performance_measures["Precision"]
    df_celeb.loc[list_EN[i], "Recall"] = performance_measures["Recall"]
    df_celeb.loc[list_EN[i], "F1"] = performance_measures["F1"]
    df_celeb.loc[list_EN[i], "Accuracy"] = performance_measures["Accuracy"]

    #dic_result_celeb[list_EN[i]].append(performance_measures)
    #dic_result_celeb[list_EN[i]].append(confusion_matrix)


df_celeb



