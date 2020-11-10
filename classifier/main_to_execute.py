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

NEO4J_HOST = os.getenv('NEO4J_HOST', default='bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', default='neo4j')
NEO4J_PASS = os.getenv('NEO4J_PASS', default='')

kg = KnowledgeGraph(NEO4J_HOST, NEO4J_USER, NEO4J_PASS)
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
component_selector = [0.5, 1, 1, 1, 1, 1, 0]
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
#component_selector = [0.5, 1, 1, 1, 1, 1, 0]
#component_selector = [0.5, 1, 1, 1, 0, 0, 0]
component_selector = [0.5, 1, 0, 0, 1, 1, 0]
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
#component_selector = [0.5, 1, 1, 1, 1, 1, 0]
component_selector = [0.5, 1, 1, 1, 0, 0, 0]
#component_selector = [0.5, 1, 1, 1, 1, 1, 0]
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

# folder='classifier/experiments/prueba/'
# nNews=100
# a=read_directory(folder,100)
# test_news=a[0]
# component_selector = [0.5,1,0,0,1,1,0]
# min_common_en = 1
# Dice_intersection__intensity = 4
# threshold_prob_fake=0.3
# count_predictions_Fake, count_predictions_NotFake, predictions_Fake, predictions_Fake_value= \
# testing_news(d,test_news,threshold_prob_fake,min_common_en,component_selector,Dice_intersection__intensity)
# predictions_Fake_value



##############################################################################
########            VISUALIZATION OF A TEST NEWS' RESULTS             ########
##############################################################################

list_path = ['./experiments/E1_1.txt','./experiments/E1_2.txt',
             './experiments/E2_1.txt','./experiments/E3_1.txt',
             './experiments/E3_2.txt','./experiments/E3_1_texto_verdaderos.txt',
             './experiments/E4_borisdrunk.txt','./experiments/E4_borishulk.txt']

## Reading the test news and obtaining it dictionary form
filepath = list_path[1]
filepath = './experiments/E4_boris_verdadera_09oct.txt'
fake_news_dict = read_fake(filepath)

## Given a fake news, we filter the knowledge base graph for the specified date
#  IMPORTANT: Threshold indicating minimum number of EN in common admitted
min_common_en = 1
knowledge_filtered, error_size_KF = knowledge_filtered_fake(d,min_common_en,fake_news_dict)

## Appending the fake news to the filtered knowledge graph. The fake news is the last position
knowledge_filtered[fake_news_dict["SOURCE"]]= fake_news_dict

## It is necessary to apply a previous filter (regarding Entity Names and Related Words) to the
# obtained filtered knowledge graph including the fake news
rwords_news_min = 10
rwords_en_min = 1
knowledge_filtered = previous_filter(knowledge_filtered,rwords_news_min,rwords_en_min)

## Obtainaning the similarity and the dissimilariry matrixes with the selected componentes
#  IMPORTANT: Fixing the weights of each component of the similarity
component_selector = [1, 1,1,1, 1,1,0]
#  IMPORTANT: Fixing the intensity for the intersection in the Dice similarity formula
Dice_intersection__intensity=4
#  Fixing some basic parameters for the similarity measure
optionSimbSim="Ichino_yaguchi"
gamma=0.2 # In case of Ichino-Yaguchi similarity
# Similarity calculations
sim_matrix, dis_matrix=similarity(knowledge_filtered,component_selector,optionSimbSim,Dice_intersection__intensity,gamma)

# Exportation
#export_csv(sim_matrix,date)

## Automathic selection of parameters epsilon and min_samples in DBSCAN algorithm
epsilon, min_samples, error_parameters_DBSCAN =DBSCAN_parameters_epsilon_minsamples(dis_matrix)

## DBSAN algorithm results
graph_labels = None
graph_y_n = True
pca_x = 1
pca_y = 2
dbscan_labels, dbscan_n_clusters, dbscan_n_noise, var_exp, label_fake, \
prob_fake = dbscan_clustering(dis_matrix,epsilon,min_samples,graph_labels,graph_y_n,pca_x, pca_y)
prob_fake
print("prob= ",prob_fake)
print("epsilon= ",epsilon)
print("min_samples= ",min_samples)

noise_index = prob_noise_points(dis_matrix, epsilon, min_samples, dbscan_labels)

