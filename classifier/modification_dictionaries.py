from classifier.test_news_new import read_directory_dictionary
import dill as pickle


##################################################################
########          DICTIONARY 26 SEPTEMBER 2019            ########
##################################################################
path = 'experiments/E5_26sept_buenas/E5_26sept_30_falsas'
list_test_news=read_directory_dictionary(path)
list_test_news_p=list(list_test_news)

## index to delete due to not processing of the news and in order to obtain balanced data
ix_true_delete_balance_data=[19,20,22,24,17,10]
ix_false_delete_balance_data=[40,43,1,9,48,44,46,4,14,23,2,5,7,8,11,13,18,20,21,26,28,29,31,33,35,37,39,42,45,50,51,36]
## Deleting True test news
for i in sorted(ix_true_delete_balance_data, reverse=True):
    del list_test_news_p[0][i]
## Deleting False test news
for i in sorted(ix_false_delete_balance_data, reverse=True):
    del list_test_news_p[1][i]

with open('experiments/E5_26sept_buenas/E5_26sept_final', 'wb') as fich:
    # Step 3
    pickle.dump(list_test_news_p, fich)



##################################################################
########           DICTIONARY 02 OCTOBER 2019             ########
##################################################################
path = 'experiments/E5_02oct_buenas/E5_02oct_30_falsas'
list_test_news=read_directory_dictionary(path)
list_test_news_p=list(list_test_news)

## index to delete due to not processing of the news and in order to obtain balanced data
ix_true_delete_not_processed=[0,1,2,3]
ix_false_delete_not_processed=[22,25,26,32,1,2,3,9,11,13,15,17,18,21,28,31,36,37,38]
## Deleting True test news
for i in sorted(ix_true_delete_not_processed, reverse=True):
    del list_test_news_p[0][i]
## Deleting False test news
for i in sorted(ix_false_delete_not_processed, reverse=True):
    del list_test_news_p[1][i]

with open('experiments/E5_02oct_buenas/E5_02oct_final', 'wb') as fich:
    # Step 3
    pickle.dump(list_test_news_p, fich)




##################################################################
########           DICTIONARY 08 OCTOBER 2019             ########
##################################################################
path = 'experiments/E5_08oct_buenas/E5_08oct_30_falsas'
list_test_news=read_directory_dictionary(path)
list_test_news_p=list(list_test_news)

## index to delete due to not processing of the news and in order to obtain balanced data
ix_true_delete_not_processed=[9,20]
ix_false_delete_not_processed=[5,6,7,12,14,19,25,26,27,35,36,38,40,34,32,9,13,31,24,17,29]
## Deleting True test news
for i in sorted(ix_true_delete_not_processed, reverse=True):
    del list_test_news_p[0][i]
## Deleting False test news
for i in sorted(ix_false_delete_not_processed, reverse=True):
    del list_test_news_p[1][i]

with open('experiments/E5_08oct_buenas/E5_08oct_final', 'wb') as fich:
    # Step 3
    pickle.dump(list_test_news_p, fich)

