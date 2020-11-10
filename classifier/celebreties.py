############################################################################################
############################################################################################
########                KRAKEN-FND EXPERIMENT ON CELEBREIES DATA SET                ########
############################################################################################
############################################################################################


import os
import sys

sys.path.append(os.path.abspath('../kraken-snd/knowledge-graph-builder/'))
sys.path.append(os.path.abspath('../kraken-snd/classifier/'))
from classifier.test_news import read_directory_dictionary, testing_news, measures_differents_threshold, measures_selected_threshold
import random

###########################################################################
########          Reading and exploring the dictionary             ########
###########################################################################

path = '../kraken-snd/classifier/experiments/overall/celebrityDataset/E6_celebrity'
celebrity_dictionary=read_directory_dictionary(path)

## Number of true news in the dictionary
n_true_news=len(celebrity_dictionary[0])

## Number of fake news in the dictionary
n_fake_news=len(celebrity_dictionary[1])



###################################################################################################
########          Creating the Knowledge-Graph from the celebreties dictionary             ########
###################################################################################################

## Random selection of true news to build the knowlefge-graph dictionary
random.seed(2019)
# Selecting index news to form the knowlefge-graph dictionary: we select all
true_news_knowledge_graph=random.sample(range(0,n_true_news), n_true_news)

## Selecting the news in the Knowledge-Graph
d = { i : celebrity_dictionary[0][i] for i in true_news_knowledge_graph}


########################################################################
########          Creating testing news dictionaries            ########
########################################################################

## Random selection of fake news and true news to be tested
random.seed(2019)
# The fake testing news will also be all of the false news
fake_test_news=random.sample(range(0,n_fake_news), n_fake_news)
# The true testing news will also be all of the true news
true_test_news=random.sample(range(0,n_true_news), n_true_news)

## Selecting news to create the test news dictionaries
TRUE_test_news = list(celebrity_dictionary[0][i] for i in true_test_news)
FALSE_test_news = list(celebrity_dictionary[1][i] for i in fake_test_news)


########################################################################################
########          Finding optimum probability threshold for decision            ########
########################################################################################
#
# ## Fixing parameters
# min_common_en = 1
# component_selector = [0.5, 1, 1, 1, 1, 1, 0]
# Dice_intersection__intensity = 4
# threshold_prob_fake=0.01
# ## Obtaining probabilities for fake for the TRUE test news
# fp, tn, predictions_Fake_True_news, predictions_Fake_value_True_news = testing_news(d, TRUE_test_news,
#                                                         threshold_prob_fake, min_common_en,component_selector,Dice_intersection__intensity)
# n_true_test_news_valid=len(predictions_Fake_value_True_news)
# ## Obtaining probabilities for fake for the FALSE test news
# tp, fn, predictions_Fake_False_news, predictions_Fake_value_False_news = testing_news(d, FALSE_test_news,
#                                                         threshold_prob_fake, min_common_en,component_selector,Dice_intersection__intensity)
# n_fake_test_news_valid=len(predictions_Fake_value_False_news)
#
# ## RESULTS FOR A FIXED THRESHOLD
# performance_measures, confusion_matrix= measures_selected_threshold(predictions_Fake_value_True_news,predictions_Fake_value_False_news,threshold_prob_fake)




######################################################################################################################
######################################################################################################################
########                KRAKEN-FND EXPERIMENT ON CELEBREIES DATA SET FILTERED BY KIM KARDASHIAN               ########
######################################################################################################################
######################################################################################################################

###########################################################################################
########          Exploring number of news containing Kim Kardashian EN            ########
###########################################################################################

## Creating the dictionary of the false celebreties news
d_fake = { i : celebrity_dictionary[1][i] for i in fake_test_news}

## Number of true news with Kim Kardashian as Entity Name
c_true = 0
for k,v in d.items():
    for i in d[k]["ENs"]:
        if i == 'http://dbpedia.org/resource/Kim_Kardashian':
            c_true += 1
print(c_true)

## Number of fake news with Kim Kardashian as Entity Name
c_fake = 0
for k,v in d_fake.items():
    for i in d_fake[k]["ENs"]:
        if i == 'http://dbpedia.org/resource/Kim_Kardashian':
            c_fake += 1
print(c_fake)


#####################################################################################
########          Filtering the dictionaries by Kim Kardashian EN            ########
#####################################################################################

## Dictionary of true news filtered by kim kardashian
d_kim = dict()
for k,v in d.items():
    for i in d[k]["ENs"]:
        if i == 'http://dbpedia.org/resource/Kim_Kardashian':
            d_kim[k]=v

## Dictionary of fake news filtered by kim kardashian
d_kim_fake = dict()
for k,v in d_fake.items():
    for i in d_fake[k]["ENs"]:
        if i == 'http://dbpedia.org/resource/Kim_Kardashian':
            d_kim_fake[k]=v



########################################################################
########          Creating testing news dictionaries            ########
########################################################################

## The true tests news will be the ones that are in the knowledge graph
TRUE_test_news = list(d_kim.values())
## The false tests news will be all in the dictionary of false news of Kim Kardashian
FALSE_test_news = list(d_kim_fake.values())


########################################################################################
########          Finding optimum probability threshold for decision            ########
########################################################################################

## Fixing parameters
min_common_en = 1
component_selector = [1.5, 1, 1, 1, 1, 1, 0]
Dice_intersection__intensity = 4
threshold_prob_fake=0 ## at the momment this parameter does not matter

## Obtaining probabilities for fake for the TRUE test news
fp, tn, predictions_Fake_True_news, predictions_Fake_value_True_news = testing_news(d_kim, TRUE_test_news,
                                                        threshold_prob_fake, min_common_en,component_selector,Dice_intersection__intensity)

## Obtaining probabilities for fake for the FALSE test news
tp, fn, predictions_Fake_False_news, predictions_Fake_value_False_news = testing_news(d_kim, FALSE_test_news,
                                                        threshold_prob_fake, min_common_en,component_selector,Dice_intersection__intensity)


## Measures for different threshold values
list_results=measures_differents_threshold(predictions_Fake_value_True_news,predictions_Fake_value_False_news)
list_threshold=list_results[0]
list_accuracy=list_results[1]
list_precision=list_results[2]
list_recall=list_results[3]
list_f1_score=list_results[4]
list_fp_ratio=list_results[5]

## Finding Optimum threshold value only maximizing F1 score
max_value = max(list_f1_score)
max_index = list_f1_score.index(max_value)
threshold_prob_fake=list_threshold[max_index]

## RESULTS FOR THE SELECTED FIXED THRESHOLD
performance_measures, confusion_matrix= measures_selected_threshold(predictions_Fake_value_True_news,predictions_Fake_value_False_news,threshold_prob_fake)
performance_measures
print(confusion_matrix)




####################################################################################################
####################################################################################################
########          CHECKING THE PERCENTAGE IN COMMON WITH THE KB OF THE TEST NEWS            ########
####################################################################################################
####################################################################################################
#
# ## TRUE TEST NEWS
# common_entities=[]
# len_TRUE_test_news=len(TRUE_test_news)
# for i in range(len_TRUE_test_news):
#     common_entities_i = []
#     test_new=TRUE_test_news[i]
#     kb_TRUE_test_news=TRUE_test_news[:i]+TRUE_test_news[i+1:]
#     EN_test_new=list(test_new['ENs'])
#     for j in range(len_TRUE_test_news-1):
#         EN_kb_TRUE_test_news=list(kb_TRUE_test_news[j]['ENs'])
#         common_entities_i.append(len(list(set(EN_kb_TRUE_test_news) & set(EN_test_new))))
#     common_entities.append(len([k for k in common_entities_i if k > 1]))
# import numpy as np
# common_entities_true=np.array(common_entities )
# mean_true=np.mean(common_entities_true/18*100)
# median_true=np.median(common_entities_true/18*100)
# min_true=np.min(common_entities_true/18*100)
#
# ## FALSE TEST NEWS
# common_entities=[]
# len_FALSE_test_news=len(FALSE_test_news)
# len_TRUE_test_news=len(TRUE_test_news)
# EN_test_new = list(test_new['ENs'])
# for i in range(len_FALSE_test_news):
#     common_entities_i = []
#     test_new=FALSE_test_news[i]
#     EN_test_new=list(test_new['ENs'])
#     for j in range(len_TRUE_test_news):
#         EN_TRUE_test_news=list(TRUE_test_news[j]['ENs'])
#         common_entities_i.append(len(list(set(EN_TRUE_test_news) & set(EN_test_new))))
#     common_entities.append(len([k for k in common_entities_i if k > 1]))
# common_entities_false=np.array(common_entities )
# mean_false=np.mean(common_entities_false/18*100)
# median_false=np.median(common_entities_false/18*100)
# min_false=np.min(common_entities_false/18*100)
#
