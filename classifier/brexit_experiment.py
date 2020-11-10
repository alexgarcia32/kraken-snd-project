#########################################################################
########                Main to execute all codes                ########
#########################################################################

import os
import sys
import pandas as pd
import glob
sys.path.append(os.path.abspath('../kraken-snd/knowledge-graph-builder/'))
sys.path.append(os.path.abspath('../kraken-snd/classifier/'))
from fake_news.neo4j_conn import KnowledgeGraph
from classifier.test_news import read_directory_dictionary, testing_news, measures_differents_threshold, measures_selected_threshold, read_directory, min_var_f1_global
import dill as pickle


from lectura import previous_filter, read_fake
from filter_related_news import knowledge_filtered_fake
from similarity import similarity, similarity_or_distance_graph, export_csv
from classifier.ClusteringWithDistanceMatrix import dbscan_predict, dbscan_clustering, DBSCAN_parameters_epsilon_minsamples, prob_noise_points


#########################################################################
########                  LOAD KNOWLEDGE GRAPH                   ########
#########################################################################

# for brexit
# terminal: ssh -L 7687:10.100.18.78:7687 -N kraken@193.147.61.144
kg = KnowledgeGraph('bolt://localhost:7687', 'neo4j', 'neo4j')
date='2019-10-08'


## Reading the original graph before performing any filter
d_original = kg.get_sentiments_magic_method(date)
## deepcopy of the original dictionary to make changes in it
d = dict(d_original)





###################################################
###################################################
########          EXPERIMENT 3             ########
###################################################
###################################################

#### SELECTING DAY TO TEST
## path of test news to read
path = '../kraken-snd/classifier/experiments/E5_08oct_buenas/E5_08oct_final'
list_test_news=read_directory_dictionary(path)
## Processing the TRUE test news
TRUE_test_news = list_test_news[0]
## Processing the FALSE test news
FALSE_test_news = list_test_news[1]


#### FINDING OPTIMUM PROBABILITY THRESHOLD FOR DECISION
## Fixing parameters
min_common_en = 1
component_selector = [1, 1, 1, 1, 1, 1, 0]
Dice_intersection__intensity = 4

## Obtaining probabilities for fake for the TRUE test news
fp, tn, predictions_Fake_True_news, predictions_Fake_value_True_news = testing_news(d, TRUE_test_news,
                                                        0, min_common_en,component_selector,Dice_intersection__intensity)
## Obtaining probabilities for fake for the FALSE test news
tp, fn, predictions_Fake_False_news, predictions_Fake_value_False_news = testing_news(d, FALSE_test_news,
                                                        0, min_common_en,component_selector,Dice_intersection__intensity)

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

## Finding GLOBAL(for the three days) Optimum threshold value maximizing F1 score and minimizing variance
df_f1_scores_sorted, list_optimum_threshold, optimum_mean_f1_score, optimum_var_f1_score = min_var_f1_global()
suggested_optimum_threshold = max(list_optimum_threshold)

## RESULTS FOR A FIXED THRESHOLD
#threshold_prob_fake = suggested_optimum_threshold
performance_measures, confusion_matrix= measures_selected_threshold(predictions_Fake_value_True_news,predictions_Fake_value_False_news,threshold_prob_fake)



###################################################
###################################################
########          EXPERIMENT 1             ########
###################################################
###################################################

## path of test news to read
path = 'classifier/experiments/experimentos 1/exp1_final'
list_test_news=read_directory_dictionary(path)

## Fixing some parameters
threshold_prob_fake=0.3
min_common_en = 1
component_selector = [1, 1, 0, 0, 1, 1, 0]
Dice_intersection__intensity = 4

## Processing the TRUE test news
TRUE_test_news = list_test_news[0]
## Obtaining probabilities for fake for the TRUE test news
fp, tn, predictions_Fake_True_news, predictions_Fake_value_True_news = testing_news(d, TRUE_test_news,
                                                        threshold_prob_fake, min_common_en,component_selector,Dice_intersection__intensity)
## Processing the FALSE test news
FALSE_test_news = list_test_news[1]
## Obtaining probabilities for fake for the FALSE test news
tp, fn, predictions_Fake_False_news, predictions_Fake_value_False_news = testing_news(d, FALSE_test_news,
                                                        threshold_prob_fake, min_common_en,component_selector,Dice_intersection__intensity)




###################################################
###################################################
########          EXPERIMENT 2             ########
###################################################
###################################################

## path of test news to read
path = 'classifier/experiments/experimentos 2/exp2_final'
list_test_news=read_directory_dictionary(path)

## Fixing some parameters
threshold_prob_fake=0.3
min_common_en = 1
component_selector = [1, 1, 1, 1, 0, 0, 0]
Dice_intersection__intensity = 4

## Processing the TRUE test news
TRUE_test_news = list_test_news[0]
## Obtaining probabilities for fake for the TRUE test news
fp, tn, predictions_Fake_True_news, predictions_Fake_value_True_news = testing_news(d, TRUE_test_news,
                                                        threshold_prob_fake, min_common_en,component_selector,Dice_intersection__intensity)
## Processing the FALSE test news
FALSE_test_news = list_test_news[1]
## Obtaining probabilities for fake for the FALSE test news
tp, fn, predictions_Fake_False_news, predictions_Fake_value_False_news = testing_news(d, FALSE_test_news,
                                                        threshold_prob_fake, min_common_en,component_selector,Dice_intersection__intensity)

