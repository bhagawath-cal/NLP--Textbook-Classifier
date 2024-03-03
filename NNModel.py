import itertools
import numpy as np
from glob import glob
import pandas
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold, cross_validate
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pickle
import csv

class FFNN(): 
    def __init__(self):
        self.classifier = None

    def save_model(self, pkl_file):
        if self.classifier == None:
            raise Exception("No model initialized, you must fit a model to save it")
        pickle.dump(self.classifier, open(pkl_file, 'wb'))
            
    def load_model(self, pkl_file):
        self.classifier = pickle.load(open(pkl_file, 'rb'))


    def fit_vectorizer(self, train, ngram_range, max_df, min_df):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df, use_idf=True)
        X = self.vectorizer.fit_transform(train)
        return X.toarray() 
    

    def search_hyperparams(self, data, labels, hyperparams, KSplits, filename, save=False):
        """
        Input: 
            data - the raw training data, not vectorized yet. Should be a list of strings 
            labels - training labels that correspond to the training data sent in. should be a list of strings that are just ints. 
            hyperparams should be a pandas dataframe of the different hyperparameters
            hyperparams: 
                Vectorizer: 
                    list of ngram_range
                    list of max_df
                    list of min_df
                    list of max_features
                NN Model: 
                    list of hidden_layer_sizes (both number of layers and their sizes)
                    list of learning_rate
                    list of alpha (strength of L2 Regularizer)

        output: best hyperparamters. 

        Should log data for each training iteration. Log mean F1 and accuracy scores for each set of hyperparams.
        """
        fields = ['ngram_range', 'max_df', 'min_df', 'max_features', 'hidden_layer_sizes', 'learning_rate', 'alpha', 'accuracy', 'f1 macro', 'fit time']  
        rows = []

        best_hyperparamset = hyperparams.iloc[0]
        best_f1 = 0
        # https://docs.python.org/3/library/itertools.html#itertools.product
        # used itertools to make code cleaner. This is the same as iterating through each list in nested loops. Order is deterministic.
        if save:
            models = glob("models/*.pkl")
        for index, hyperparamset in hyperparams.iterrows():            
            modelfile = f"models/model{index}.pkl"
            
            # if save is true, ignore models already generated
            if save and modelfile in models: 
                print(f"{modelfile} already exists, skipping")
                continue

            print(f"training on hyperparamset {index} in hyperparams.csv")
            ngram_range = (hyperparamset['min_ngram'], hyperparamset['max_ngram'])
            max_df = hyperparamset['max_df']
            min_df = hyperparamset['min_df']
            max_features = hyperparamset['max_features']
            hidden_layer_sizes = json.loads(hyperparamset['hidden_layer_sizes'])
            learning_rate = hyperparamset['learning_rate']
            alpha = hyperparamset['alpha']
            if np.isnan(max_features):
                max_features = None

            # create vectorizer and classifier 
            vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_range, max_features=max_features, use_idf=True)
            classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate, alpha=alpha, solver='adam', tol=1e-3)

            # transform data 
            X = vectorizer.fit_transform(data)
            X = X.toarray()
            kfold = KFold(n_splits=KSplits)
            scoring = ['accuracy', 'f1_macro']
            scores = cross_validate(classifier, X, labels, cv=kfold, scoring=scoring, verbose=2, n_jobs=3)
            accuracy = np.mean(scores["test_accuracy"])
            f1_macro = np.mean(scores["test_f1_macro"])
            fit_time = np.mean(scores["fit_time"])

            print(f"accuracy: {accuracy}, f1 macro: {f1_macro}, fit time: {fit_time}")

            rows.append([ngram_range, max_df, min_df, max_features, hidden_layer_sizes, learning_rate, alpha, accuracy, f1_macro, fit_time])

            if f1_macro > best_f1:
                best_hyperparamset = hyperparamset

            if save:
                print("fitting model to save")
                self.fit_model(data, labels, hyperparamset)
                pickle.dump(self, open(modelfile, 'wb'))

        with open(filename, 'w') as csvfile: 
            csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL, escapechar='\\') 
            csvwriter.writerow(fields)
            csvwriter.writerows(rows)

        return best_hyperparamset




    def fit_model(self, data, labels, hyperparams):
        """
        Take the best hyperparams from the search, train the model with them
        hyperparams should be a pandas frame that is only 1 row long. If it is longer, it will take the first row.
        """
        ngram_range = (hyperparams['min_ngram'], hyperparams['max_ngram'])
        max_df = hyperparams["max_df"]
        min_df = hyperparams["min_df"]
        max_features = hyperparams["max_features"]
        hidden_layer_sizes = json.loads(hyperparams["hidden_layer_sizes"])
        learning_rate = hyperparams["learning_rate"]
        alpha = hyperparams["alpha"]

        # Set model parameters 
        X = self.fit_vectorizer(data, ngram_range, max_df, min_df)
        self.classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate, alpha=alpha, solver='adam', tol=1e-3)

        self.classifier.fit(X, labels)
        


    def predict(self, data):
        if self.classifier == None: 
            raise Exception("No model initialized, you must fit a model make predictions")
        X = self.vectorizer.transform(data)
        return self.classifier.predict(X)



    def test_model(self, test_data):
        pass




