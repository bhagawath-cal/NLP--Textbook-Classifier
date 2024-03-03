import csv
import string
import itertools



def main():
   
    fields = ['min_ngram', 'max_ngram', 'max_df', 'min_df', 'max_features', 'hidden_layer_sizes', 'learning_rate', 'alpha']  
    rows = []
    filename = "hyperparams_test.csv"

    # These worked well in the past
    rows.append([1, 1, 0.8, 0.0, "None", [100, 50], 0.01, 0.01])
    rows.append([1, 1, 0.8, 0.0, "None", [50, 50, 50], 0.01, 0.01])  
    rows.append([1, 1, 0.8, 0.0, "None", [400, 50], 0.01, 0.01])

    min_ngrams = [1]
    max_ngrams = [1, 2]
    max_df = [0.8]
    min_df = [0.0, 0.1]
    max_features_list = [None]
    hidden_layer_sizes_list = [[100, 75], [60, 70, 80], [75, 100, 125], [200, 250, 300], [120, 130, 140]]
    learning_rates = [0.01]
    alphas = [0.001, 0.01]

    for hyperparamset in itertools.product(min_ngrams, max_ngrams, max_df, min_df, max_features_list, hidden_layer_sizes_list, learning_rates, alphas):
        rows.append([hyperparamset[0], hyperparamset[1], hyperparamset[2], hyperparamset[3], hyperparamset[4], hyperparamset[5], hyperparamset[6], hyperparamset[7]])

    with open(filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL, escapechar='\\') 
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)   
    return


if __name__ == "__main__":
   main()