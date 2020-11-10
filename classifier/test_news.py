import glob
import os
import sys
sys.path.append(os.path.abspath('../../knowledge-graph-builder/'))
sys.path.append(os.path.abspath('/'))
from lectura import previous_filter, read_fake
import dill as pickle
from filter_related_news import knowledge_filtered_fake
from similarity import similarity
from ClusteringWithDistanceMatrix import dbscan_clustering, DBSCAN_parameters_epsilon_minsamples
import numpy as np
import pandas as pd
from sklearn.metrics import auc

## Function to read test news from a directory
def read_directory(folder,nNews):
    filepaths = glob.glob(os.path.join(folder, 'falsas', '*.txt'))
    print(len(filepaths))
    filepaths = glob.glob(os.path.join(folder, 'verdaderas','*.txt'))


    newsv = []
    lens = []
    k =0
    for fp in filepaths:
        print(k)
        with open(fp, 'r',encoding="utf8") as f:
            # Read the first line of the file
            r = f.read()
            a = read_fake(fp)
            a["fich"] = fp
            newsv.append(a)
            k += 1
            lens.append(len(r))
            if k > nNews:
                break
    newsf = []
    lens = []
    k = 0
    filepaths = glob.glob(os.path.join(folder, 'falsas', '*.txt'))
    for fp in filepaths:
        print(k)
        with open(fp, 'r', encoding="utf8") as f:
            # Read the first line of the file

            r = f.read()

            a = read_fake(fp)
            a["fich"] = fp
            newsf.append(a)
            k += 1
            lens.append(len(r))
            if k > nNews:
                break
    return (newsv,newsf)


## Function to create a dictionary with the test news and save it
def create_dictionary():
    a = read_directory("experiments/prueba",100)

    for i in a:
        for k in i:
            for k2,v in list(k["ENs"].items()):
                if k2 == "http://dbpedia.org/resource/Boris_Yeltsin":
                    k["ENs"]["http://dbpedia.org/resource/Boris_Johnson"] = k["ENs"].pop('http://dbpedia.org/resource/Boris_Yeltsin')

    with open('../experiments/E5_26sept_buenas/E5_26sept', 'wb') as fich:
        # Step 3
        pickle.dump(a, fich)

    b = read_directory("experiments/E5_02oct_buenas",100)
    for i in b:
        for k in i:
            for k2,v in list(k["ENs"].items()):
                if k2 == "http://dbpedia.org/resource/Boris_Yeltsin":
                    k["ENs"]["http://dbpedia.org/resource/Boris_Johnson"] = k["ENs"].pop('http://dbpedia.org/resource/Boris_Yeltsin')

    with open('../experiments/E5_02oct_buenas/E5_02oct', 'wb') as fich:
        # Step 3
        pickle.dump(b, fich)

    c = read_directory("experiments/E5_08oct_buenas",100)
    for i in c:
        for k in i:
            for k2,v in list(k["ENs"].items()):
                if k2 == "http://dbpedia.org/resource/Boris_Yeltsin":
                    k["ENs"]["http://dbpedia.org/resource/Boris_Johnson"] = k["ENs"].pop('http://dbpedia.org/resource/Boris_Yeltsin')
    with open('../experiments/E5_08oct_buenas/E5_08oct', 'wb') as fich:
        # Step 3
        pickle.dump(c, fich)


## Function to load the previous dictionary
def read_directory_dictionary(path):
    #cargar
    with open(path, 'rb') as fich:
        news = pickle.load(fich)
        return news


## Function to compute if the test news are fake or not
 # Param "test_news" is either the list of TRUE news to test or the FALSE news to test
def testing_news(d, test_news, threshold_prob_fake, min_common_en, component_selector,Dice_intersection__intensity):
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
            pass
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
                pass
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

def measures_differents_threshold(prob_fakes_values_T,prob_fakes_values_F):

    ## number of news processed
    n_news_True=len(prob_fakes_values_T)
    n_news_False=len(prob_fakes_values_F)

    ## Initialising some variables to save information
    list_threshold=[]
    list_accuracy=[]
    list_precision=[]
    list_recall=[]
    list_f1_score=[]
    list_fp_ratio=[]

    ## Looping through posible values of the threshold
    for t_aux in range(1,101):
        ## Computing threshold for considering fake
        threshold_prob_fake=t_aux/100
        list_threshold.append(threshold_prob_fake)

        ## Obtaining results for that threshold
        fp=len([p for p in prob_fakes_values_T if p >= threshold_prob_fake])
        tn=n_news_True-fp
        tp=len([p for p in prob_fakes_values_F if p >= threshold_prob_fake])
        fn=n_news_False-tp

        # Performance Measures
        if tp + tn + fp + fn == 0:
            accuracy = 0
        else:
            accuracy = (tp + tn) / (tp + tn + fp + fn)
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if recall == 0 and precision == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        # For curve ROC
        if fp + tn == 0:
            fp_ratio = 0
        else:
            fp_ratio = fp / (fp + tn)

        ## Appending these measures to a list
        list_accuracy.append(accuracy)
        list_precision.append(precision)
        list_recall.append(recall)
        list_f1_score.append(f1_score)
        list_fp_ratio.append(fp_ratio)

    r=[list_threshold,list_accuracy,list_precision,list_recall, list_f1_score, list_fp_ratio]

    return r



def measures_selected_threshold(prob_fakes_values_T,prob_fakes_values_F,threshold_prob_fake):

    ## number of news processed
    n_news_True=len(prob_fakes_values_T)
    n_news_False=len(prob_fakes_values_F)

    ## Obtaining results for that threshold
    fp = len([p for p in prob_fakes_values_T if p >= threshold_prob_fake])
    tn = n_news_True - fp
    tp = len([p for p in prob_fakes_values_F if p >= threshold_prob_fake])
    fn = n_news_False - tp

    # Performance Measures
    if tp + tn + fp + fn == 0:
        accuracy = 0
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if recall == 0 and precision == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    # For curve ROC
    if fp + tn == 0:
        fp_ratio = 0
    else:
        fp_ratio = fp / (fp + tn)

    keys=["Threshold","Precision","Recall","F1","Accuracy"]
    values=[threshold_prob_fake,precision,recall, f1_score, accuracy]
    results = dict(zip(keys, values))

    ## Confusion matrix
    confusion_matrix = np.array([tn, fn, fp, tp]).reshape(2, 2)
     # Naming rows and columns in order to understand the content of the matrix
    DF_confusion_matrix= pd.DataFrame(confusion_matrix, columns=["Observed: NOT FAKE","Observed: FAKE" ],
                      index=[ "Predicted: NOT FAKE","Predicted: FAKE"])

    return results, DF_confusion_matrix


## Function to calculate the threshold than maximize F1 score and minimize its variance among the three days
def min_var_f1_global():
    # We have already saved the list of threshold and, for the three days, the list of the F1 score
    df_f1_scores = pd.read_csv('../experiments/F1_score_3days_python.csv', header=0, sep=";", decimal=",")
    # We calculate variance and mean
    df_f1_scores['mean_f1_score'] = df_f1_scores[["Day1", "Day2", "Day3"]].mean(axis=1)
    df_f1_scores['var_f1_score'] = df_f1_scores[["Day1", "Day2", "Day3"]].var(axis=1)

    # We get unique values for mean and var and list of valid threshold for each pair (mean,var)
    values_pair = df_f1_scores.groupby('mean_f1_score').agg({'var_f1_score': 'mean', 'Threshold': list})
    values_pair['mean_f1_score'] = values_pair.index
    values_pair.reset_index(drop=True, inplace=True)
    values_pair['Mean-var'] = values_pair['mean_f1_score'] - values_pair['var_f1_score']
    cols = ['mean_f1_score', 'var_f1_score', 'Mean-var', 'Threshold']
    values_pair = values_pair[cols]
    # We sort Mean-var by descending order, the maximum value will determine the optimum threshold
    df_f1_scores_sorted = values_pair.sort_values(['Mean-var'], ascending=[False], inplace=False)

    # Optimum threshold
    list_optimum_threshold = df_f1_scores_sorted.iloc[0, 3]
    optimum_mean_f1_score = df_f1_scores_sorted.iloc[0, 0]
    optimum_var_f1_score = df_f1_scores_sorted.iloc[0, 1]

    return df_f1_scores_sorted, list_optimum_threshold, optimum_mean_f1_score, optimum_var_f1_score


