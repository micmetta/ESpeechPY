
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB



#filename_glass = "temp.xlsx"
filename_glass = "dataset_temp.xlsx"
#filename_glass = "dataset_DEFINITIVO.xlsx"
df_glass = pd.read_excel(filename_glass)

print(df_glass.shape)
print(df_glass.head())
print(df_glass.describe())




def get_train_test(df, y_col, x_cols, ratio):
    """
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 0.7)
    2. the column with the Y_values
    """
    #mask = np.random.rand(len(df)) &lt; ratio
    #mask = np.random.rand(len(df)); #la funzione np.random.rand restituisce un array di un certo numero di elementi
                                     #pari al valore intero passatogli come paramentro. Gli elementi dell'array sono
                                     #valori compresi tra 0 e 1
    #df_train = df[mask]
    #df_test = df[~mask]

    df_train, df_test = train_test_split(df, test_size=0.30)

    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values

    return df_train, df_test, X_train, Y_train, X_test, Y_test



y_col_glass = 'Emozione'
x_cols_glass = list(df_glass.columns.values)
x_cols_glass.remove(y_col_glass)

train_test_ratio = 0.7
df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df_glass, y_col_glass, x_cols_glass,train_test_ratio)

dict_classifiers = {

    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000,random_state=0),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=1000, random_state=0),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(),

}




def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers=5, verbose=True):
    """
    This method, takes as input the X, Y matrices of the Train and Test set.
    And fits them on all of the Classifiers specified in the dict_classifier.
    The trained models, and accuracies are saved in a dictionary. The reason to use a dictionary
    is because it is very easy to save the whole dictionary with the pickle module.
    Usually, the SVM, Random Forest and Gradient Boosting Classifier take quiet some time to train.
    So it is best to train them on a smaller dataset first and
    decide whether you want to comment them out or not based on the test accuracy score.
    """
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        #t_start = time.clock()
        t_start = time.perf_counter()
        classifier.fit(X_train, Y_train)
        #t_end = time.clock()
        t_end = time.perf_counter()
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)

        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score,
                                        'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models





def display_dict_models(dict_models, sort_by='test_score'):

    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]

    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls), 4)),
                       columns=['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0, len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]


    #display(df_.sort_values(by=sort_by, ascending=False))
    print(df_.sort_values(by=sort_by, ascending=False))





dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers=9)
display_dict_models(dict_models)








'''
print("Miglioramenti random forest: ")
#Provo a migliorare il random forest e vedo i risultati


GDB_params = {
    'n_estimators': [10, 100, 500, 1000],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 100, 1000, None],
    'min_samples_leaf':[1, 2, 4],
    'max_features':["sqrt", "log2", None,"auto"]

}


df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df_glass, y_col_glass, x_cols_glass, 0.7)



for n_est in GDB_params['n_estimators']:
    for crit in GDB_params['criterion']:
        for max_d in GDB_params['max_depth']:
            for min_samples in GDB_params['min_samples_leaf']:
                for max_feat in GDB_params['max_features']:
                    clf = RandomForestClassifier(n_estimators=n_est,criterion=crit, max_depth=max_d, min_samples_leaf=min_samples, max_features=max_feat)
                    clf.fit(X_train, Y_train)
                    train_score = clf.score(X_train, Y_train)
                    test_score = clf.score(X_test, Y_test)
                    print("For ({},{},{},{},{}) - train, test score: \t {:.5f} \t-\t {:.5f}".format(n_est,crit,max_d, min_samples,max_feat,train_score,test_score))
                                                                                      
'''



'''
filename_glass = "dataset_temp.xlsx"
df_glass = pd.read_excel(filename_glass)
print("MATRICE DI CORRELAZIONE")
correlation_matrix = df_glass.corr()
plt.figure(figsize=(10,8))
ax = sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True,fmt='.2f', cmap ='GnBu', cbar_kws={"shrink": .5}, robust=True)
plt.title('Correlation matrix between the features', fontsize=20)
plt.show()
'''








