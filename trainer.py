import itertools
import argparse
import numpy as np
import pandas
from tqdm import tqdm
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from NNModel import FFNN


def get_data(csv_file, test_size):
    """
    CSV file: filepath to CSV with data contained
    test_size: proportion of data set aside for test. 
    """
    dataframe = pandas.read_csv(csv_file)
    data = dataframe[['Chapter','Page', 'Sentence']]
    train, test = train_test_split(data, test_size=test_size, shuffle=True)
    return train, test

def get_hyperparams(index=0, fit=False):
    dataframe = pandas.read_csv("hyperparams.csv")
    hyperparams = dataframe[['min_ngram', 'max_ngram', 'max_df', 'min_df', 'max_features', 'hidden_layer_sizes', 'learning_rate', 'alpha']]
    
    if fit:
        hyperparams = hyperparams.iloc[index]

    return hyperparams

def main():

    # Check commandline args 
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-o", "--output", help = "the output file name for the data provided by the model, in csv format (should end in .csv). ignored it task is 'fit'. default='output.csv'", default="output.csv")
    argparser.add_argument("-t", "--task", default="fit", choices=["fit", "search", "both"], 
                           help="task for the trainer: \nIf fit, should provide a set of hyperparameters from the hyperparams.csv, and will print out the performance on the test data, and output model as --name. "+ 
                           "If search, model will be cross-validated on the entire dataset for all hyperparameter sets in hyperparams.csv, and will output data to file chosen with --output "+ 
                           "If both, will train and fit on best params from the entire hyperparams.csv, and then fit and output the model. default='fit'")
    argparser.add_argument("-p", "--params", type=int, help="an integer that represents the row in hyperparams.csv that contains the set of hyperparams to be fit on. ignored if task is not 'fit'. default=0", default=0)
    argparser.add_argument("-d", "--dataset", default="data.csv", help="data for model to be trained on. Default is data.csv")
    argparser.add_argument("-n", "--name", default="NNModel.pkl", help="output name for the pkl created by the fitting procedure. ignored if task is 'search' (should end in .pkl) default='NNModel.pkl'")
    argparser.add_argument('-s', "--save", action='store_true', help="if flag passed as argument, the training function in the model will fit and save the model for each set oh hyperparameters, and will skip over hyperparamsets that have already been tested. Will save time if the search process if interrupted but takes more time for each run. subsequent runs should be given a different --output file")
    args = argparser.parse_args()

    print(f"{args.output} {args.task} {args.params} {args.dataset} {args.name} {args.save}")
    
    hyperparams = get_hyperparams(fit=(args.task=="fit"), index=args.params)
    FitModel = False
    SearchModel = False
    
    if args.task == "fit" or args.task == "both":
        FitModel = True
        fit_hyperparams = hyperparams
    if args.task == "search" or args.task == "both":
        SearchModel = True

    print(hyperparams)

    train, test = get_data(args.dataset, 0.1)
    labels = train["Page"]
    model = FFNN()

    if SearchModel:
        fit_hyperparams = model.search_hyperparams(data=train["Sentence"].apply(lambda x: np.str_(x)), labels=train["Page"], hyperparams=hyperparams, KSplits=5, filename=args.output, save=args.save)
        print(f"best hyperparameters = \n{fit_hyperparams}")

    if FitModel:
        model.fit_model(data=train["Sentence"].apply(lambda x: np.str_(x)), labels=labels, hyperparams=fit_hyperparams)
        pickle.dump(model, open(args.name, 'wb'))
        y_pred = model.predict(test["Sentence"].apply(lambda x: np.str_(x)))
        y_true = test["Page"]
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"score: {f1:10.5f}")

    return


if __name__ == "__main__":
   main()


