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
date='2020-03-19'

## Reading the original graph before performing any filter
d_original = kg.get_sentiments_magic_method(date)
## deepcopy of the original dictionary to make changes in it
d = dict(d_original)




#########################################################################################
########                  CHECKING IF CORONAVIRUS IS AN ENTITY                   ########
#########################################################################################
# Check how many Fake news contain Coronavirus
coronavirus = ["http://dbpedia.org/resource/Coronavirus","http://dbpedia.org/resource/Coronaviridae",
               'http://dbpedia.org/resource/SARS_coronavirus','http://dbpedia.org/resource/Severe_acute_respiratory_syndrome',
               "http://dbpedia.org/resource/SARS_virus","http://dbpedia.org/resource/SARS-CoV",
               "http://dbpedia.org/resource/Severe_acute_respiratory_syndrom_coronavirus",
               "http://dbpedia.org/resource/Sars_virus",
               "http://dbpedia.org/resource/Severe_acute_respiratory_syndrome_coronavirus",
               "http://dbpedia.org/resource/SARS_CoV",
               "http://dbpedia.org/resource/SARS-associated_coronavirus"]


n_corona = 0
for k in range(len(FALSE_test_news)):
    if set(coronavirus) & set(list(FALSE_test_news[k]["ENs"])):
        n_corona+=1
# 12/28 old processing
# 28/28 new processing

# Check how many news in knowledge graph contain Coronavirus
datelist_all = pd.date_range(start="2020-02-18",end="2020-04-01",freq="D").strftime("%Y-%m-%d")
columns = ["percentage","n_total"]
df_n_covid = pd.DataFrame(index=datelist_all,columns=columns)
df_n_covid.index.name = "Date"
for i in range(0,len(datelist_all)):
    n_corona = 0
    date_loop = datelist_all[i]
    ## Reading the original graph before performing any filter
    d_original = kg.get_sentiments_magic_method(date_loop)
    ## deepcopy of the original dictionary to make changes in it
    d = dict(d_original)
    n_total = len(d)

    for news in list(d.keys()):
        #print(set(d[news]['ENs']))
        if set(coronavirus) & set(d[news]['ENs']):
            n_corona += 1

    df_n_covid.loc[date_loop,"percentage"] = n_corona/n_total
    df_n_covid.loc[date_loop, "n_total"] = n_total
    print(i)


print(df_n_covid)
df_n_covid.to_csv('percentage_covid_newprocessing.csv', index=True)
# % of news containing ENs about coronavirus per day

# Tenemos datos desde 2020-02-18 hasta 2020-03-19



#########################################################################################
########                           PREVIOUS FUNCTIONS                            ########
#########################################################################################

# We first adapt the fuction testing news
def testing_news_all_values(d, test_news, threshold_prob_fake, min_common_en, component_selector,Dice_intersection__intensity):
    ## Initialising the count for prediction "fake news"
    count_predictions_Fake = 0
    predictions_Fake = []
    predictions_Fake_value = []
    ## There must be at least n_tested_news tested news
    # Initialising the index in the list of test news to read
    len_test_news = len(test_news)
    # Initialising the count of valid_news_NotFake
    valid_tested_news = 0
    for i in range(len_test_news):
        ## Reading test news number i
        test_news_dict = test_news[i]

        ## Filtering the knowledge base graph through the test news
        knowledge_filtered, error_size_KF = knowledge_filtered_fake(d, min_common_en, test_news_dict)

        ## If there is not enough size in the filtered knowledge graph, pass to the next news.
        if error_size_KF == 1:
            predictions_Fake_value.append(float("nan"))
        ## Otherwise continue the process
        else:
            ## Appending the fake news to the filtered knowledge graph. The fake news is the last position
            knowledge_filtered[test_news_dict["SOURCE"]] = test_news_dict

            ## It is necessary to apply a previous filter (regarding Entity Names and Related Words) to the
            # obtained filtered knowledge graph including the fake news
            rwords_news_min = 10
            rwords_en_min = 1
            knowledge_filtered = previous_filter(knowledge_filtered, rwords_news_min, rwords_en_min)

            ## Obtainaning the similarity and the dissimilariry matrixes with the selected componentes
            #  Fixing some basic parameters for the similarity measure
            optionSimbSim = "Ichino_yaguchi"
            gamma = 0.2  # In case of Ichino-Yaguchi similarity
            # Similarity calculations
            dis_matrix = similarity(knowledge_filtered, component_selector, optionSimbSim,
                                    Dice_intersection__intensity, gamma)

            ## Automathic selection of parameters epsilon and min_samples in DBSCAN algorithm
            epsilon, min_samples, error_parameters_DBSCAN = DBSCAN_parameters_epsilon_minsamples(dis_matrix)
            ## If there is not enough size in yhr filtered knowledge graph, pass to the next news
            if error_parameters_DBSCAN == 1:
                predictions_Fake_value.append(float("nan"))
            ## Otherwise continue the process
            else:
                ## DBSAN algorithm results
                dbscan_labels, dbscan_n_clusters, dbscan_n_noise, var_exp, label_fake, \
                prob_fake = dbscan_clustering(dis_matrix, epsilon, min_samples, None, False)

                predictions_Fake_value.append(prob_fake)

                ## Deciding if the test news is a fake new or not and adding it to the count of
                # false predictions of NotFake
                if (prob_fake > threshold_prob_fake):
                    count_predictions_Fake += 1
                    predictions_Fake.append(i)

                valid_tested_news += 1

        print("i = ", i, ";     valid_tested_news = ", valid_tested_news)

    ## Computing the number of predictions "NotFake"
    count_predictions_NotFake = valid_tested_news - count_predictions_Fake

    return count_predictions_Fake, count_predictions_NotFake, predictions_Fake, predictions_Fake_value



#########################################################################################
######                           EXPERIMENT WITH FAKE NEWS                         ######
#########################################################################################

#####################################################
####               We load the news              ####
#####################################################
## path of test news to read
path = '../kraken-snd/classifier/experiments/covid/covid_falsas_nuevo'
list_test_news=read_directory_dictionary(path)

## Processing the TRUE test news
TRUE_test_news = list_test_news[0]
## Processing the FALSE test news
FALSE_test_news = list_test_news[1]


# Range of dates to evaluate fake news
datelist = pd.date_range(start="2020-02-18",end="2020-03-19",freq="1D").strftime("%Y-%m-%d")

# We create dataframe to save results
columns = datelist
name_fake = [FALSE_test_news[i]['fich'] for i in range(0,len(FALSE_test_news))]
df_covid_fake = pd.DataFrame(index=name_fake,columns=columns)
df_covid_fake.index.name = "FakeNews"


for i in range(0,len(datelist)):
     print(i)
     date_loop = datelist[i]

     ## Reading the original graph before performing any filter
     d_original = kg.get_sentiments_magic_method(date_loop)
     #d_original = kg.get_sentiments_magic_method(date_loop, delta_days=2)
     d = dict(d_original)

     min_common_en = 1
     component_selector = [1, 1, 1, 1, 1, 1, 0]
     Dice_intersection__intensity = 4

     ## Obtaining probabilities for fake for the FALSE test news
     tp, fn, predictions_Fake_False_news, predictions_Fake_value_False_news = testing_news_all_values(d, FALSE_test_news,
                                                                                           0, min_common_en,
                                                                                           component_selector,
                                                                                           Dice_intersection__intensity)

     df_covid_fake.loc[:,date_loop] = predictions_Fake_value_False_news


path_save=os.getcwd()+'/classifier/experiments/covid/results/prob_covid_fake_newProcessing_all_days_comp1is1.csv'
df_covid_fake.to_csv(path_save,na_rep='')


#########################################################################################
######                           EXPERIMENT WITH TRUE NEWS                         ######
#########################################################################################
## path of test news to read
path = '../kraken-snd/classifier/experiments/covid/verdaderas_carmensiemprecovid1530_nuevo' #verdaderas_carmensiemprecovid_nuevosdias_nuevo
list_test_news=read_directory_dictionary(path)
# verdaderas_carmensiemprecovid_nuevo son 10 noticias los días:
# '2020-02-18', '2020-02-24', '2020-03-01', '2020-03-07', '2020-03-13', '2020-03-19'
# verdaderas_carmensiemprecovid1530_nuevo los 6 dias de antes

# verdaderas_carmensiemprecovid_nuevosdias_nuevo son 10 noticias los días:
# '2020-02-18', '2020-02-25', '2020-03-03', '2020-03-10', '2020-03-17'

## Processing the TRUE test news
#TRUE_test_news = list_test_news
TRUE_test_news = [value for value in list_test_news.values()] #lo pongo así porque tienen otro formato

# Range of dates to evaluate true news
#datelist = pd.date_range(start="2020-02-18",end="2020-03-19",freq="6D").strftime("%Y-%m-%d") # verdaderas_carmensiemprecovid_nuevo
#datelist = pd.date_range(start="2020-02-18",end="2020-03-19",freq="7D").strftime("%Y-%m-%d") # verdaderas_carmensiemprecovid_nuevosdias_nuevo

# Datelist for verdaderas_carmensiemprecovid_nuevo, verdaderas_carmensiemprecovid1530_nuevo, verdaderas_marinasiemprecovid3045_nuevo
datelist = ['2020-02-18', '2020-02-24', '2020-03-01', '2020-03-07', '2020-03-13', '2020-03-19','original_date']
# Datelist for verdaderas_carmensiemprecovid_nuevosdias_nuevo
# datelist = ['2020-02-18', '2020-02-25', '2020-03-03', '2020-03-10', '2020-03-17','original_date']


# To check if Coronavirus is an EN in all the true news
# list_corona = []
# n_corona = 0
# for k in range(len(TRUE_test_news)):
#     if set(coronavirus) & set(list(TRUE_test_news[k]["ENs"])):
#         n_corona += 1
#         list_corona.append("1")
#     else:
#         list_corona.append("0")



### To delete the test true news from the kwoledge graph
#for i in TRUE_test_news.keys(): #we remove from d the test true news in TRUE_test_news
#     d.pop(i, None)


# We create dataframe to save results
columns = datelist
name_true = list_test_news.keys()
df_covid_true = pd.DataFrame(index=name_true,columns=columns)
df_covid_true.index.name = "TrueNews"


for i in range(0,len(datelist)-1):
     print(i)
     date_loop = datelist[i]

     ## Reading the original graph before performing any filter
     d_original = kg.get_sentiments_magic_method(date_loop)
     #d_original = kg.get_sentiments_magic_method(date_loop, delta_days=2)
     d = dict(d_original)
     print(len(d))

     for j in list_test_news.keys():
         if j in d:
             print(j)
             df_covid_true.loc[j, 'original_date'] = date_loop #we save the original date of the true news
             d.pop(j, None) # we remove from d the test true news in TRUE_test_news

     print(len(d))

     min_common_en = 1
     component_selector = [1, 1, 1, 1, 1, 1, 0]
     Dice_intersection__intensity = 4

     ## Obtaining probabilities for fake for the FALSE test news
     fp, tn, predictions_Fake_True_news, predictions_Fake_value_True_news = testing_news_all_values(d, TRUE_test_news,
                                                                                         0,
                                                                                         min_common_en,
                                                                                         component_selector,
                                                                                         Dice_intersection__intensity)

     df_covid_true.loc[:,date_loop] = predictions_Fake_value_True_news


path_save=os.getcwd()+'/classifier/experiments/covid/results/prob_covid_true_newProcessing_6days_comp1is1_other10more.csv' #prob_covid_true_newProcessing_5days_comp1is1.csv
df_covid_true.to_csv(path_save,na_rep='')



#########################################################################################
######                    EXPERIMENT WITH ONE TRUE NEWS FROM KB                    ######
#########################################################################################

NEO4J_HOST = os.getenv('NEO4J_HOST', default='bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', default='neo4j')
NEO4J_PASS = os.getenv('NEO4J_PASS', default='')

kg = KnowledgeGraph(NEO4J_HOST, NEO4J_USER, NEO4J_PASS)
date='2020-03-19'

## Reading the original graph before performing any filter
d_original = kg.get_sentiments_magic_method(date)
## deepcopy of the original dictionary to make changes in it
d = dict(d_original)


datelist = ['2020-02-18', '2020-02-24', '2020-03-01', '2020-03-07', '2020-03-13','2020-03-19','original_date']
news=['https://news.slashdot.org/story/20/03/19/0232245/what-happens-if-the-us-does-absolutely-nothing-to-combat-covid-19',
      'https://news.yahoo.com/italy-set-prolong-anti-coronavirus-084600570.html',
      'https://thenextweb.com/?p=1279424',
      'https://www.androidcentral.com/amazon-shuts-down-nyc-warehouse-after-worker-tests-positive-coronavirus',
      'https://www.businessinsider.com/novel-coronavirus-lockdown-preventive-isolation-protective-equipment-gear-2020-3',
      'https://www.businessinsider.com/coronavirus-could-make-florida-like-an-uber-italy-demographer-says-2020-3',
      'https://news.yahoo.com/asia-contained-coronavirus-home-now-110015510.html',
      'https://www.businessinsider.com/italy-surpasses-china-highest-coronavirus-death-toll-2020-3',
      'https://tech.slashdot.org/story/20/03/19/1830210/netflix-to-reduce-eu-bandwidth-by-25']

list_test_news={your_key: d[your_key] for your_key in news }
TRUE_test_news = [value for value in list_test_news.values()]


# We create dataframe to save results
columns = datelist
name_true = list_test_news.keys()
df_covid_true = pd.DataFrame(index=name_true,columns=columns)
df_covid_true.index.name = "TrueNews"

for i in range(0,len(datelist)-1):
     print(i)
     date_loop = datelist[i]

     ## Reading the original graph before performing any filter
     d_original = kg.get_sentiments_magic_method(date_loop)
     #d_original = kg.get_sentiments_magic_method(date_loop, delta_days=2)
     d = dict(d_original)
     print(len(d))

     for j in list_test_news.keys():
         if j in d:
             print(j)
             df_covid_true.loc[j, 'original_date'] = date_loop #we save the original date of the true news
             d.pop(j, None) # we remove from d the test true news in TRUE_test_news

     print(len(d))

     min_common_en = 1
     component_selector = [1, 1, 1, 1, 1, 1, 0]
     Dice_intersection__intensity = 4

     ## Obtaining probabilities for fake for the FALSE test news
     fp, tn, predictions_Fake_True_news, predictions_Fake_value_True_news = testing_news_all_values(d, TRUE_test_news,
                                                                                         0,
                                                                                         min_common_en,
                                                                                         component_selector,
                                                                                         Dice_intersection__intensity)

     df_covid_true.loc[:,date_loop] = predictions_Fake_value_True_news

     path_save = os.getcwd() + '/classifier/experiments/covid/results/prueba1.csv'  # prob_covid_true_newProcessing_5days_comp1is1.csv

