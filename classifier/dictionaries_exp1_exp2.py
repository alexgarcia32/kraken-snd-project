import dill as pickle
from classifier.test_news import read_directory


###################################################
###################################################
########          EXPERIMENT 2             ########
###################################################
###################################################

folder='classifier/experiments/experimentos 2/'
nNews=100

exp2=read_directory(folder,nNews)
for i in exp2:
    for k in i:
        for k2, v in list(k["ENs"].items()):
            if (k2 == "http://dbpedia.org/resource/Boris_Yeltsin"):
                k["ENs"]["http://dbpedia.org/resource/Boris_Johnson"] = k["ENs"].pop('http://dbpedia.org/resource/Boris_Yeltsin')

with open('classifier/experiments/experimentos 2/exp2_final', 'wb') as fich:
# Step 3
    pickle.dump(exp2, fich)




###################################################
###################################################
########          EXPERIMENT 1             ########
###################################################
###################################################

folder='classifier/experiments/experimentos 1/auto_changed'
nNews=100

exp1=read_directory(folder,nNews)
for i in exp1:
    for k in i:
        for k2, v in list(k["ENs"].items()):
            if (k2 == "http://dbpedia.org/resource/Boris_Yeltsin"):
                k["ENs"]["http://dbpedia.org/resource/Boris_Johnson"] = k["ENs"].pop('http://dbpedia.org/resource/Boris_Yeltsin')

with open('classifier/experiments/experimentos 1/exp1_final', 'wb') as fich:
# Step 3
    pickle.dump(exp1, fich)