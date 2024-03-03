# Project Overview 
The goal of this project is to provide a sort of companion for students taking NLP classes like EECS 487. This companion functions like a chatbot, providing the user the ability 
to input text into its interface, and the model will attempt to match that text to a given textbook page from [Speech and Language Processing (3rd ed. draft) by Dan Jurafsky and James H. Martin](https://web.stanford.edu/~jurafsky/slp3/). The exact draft of the textbook that is being used for this model is included in here, as any subsequent draft would lead to inaccurate page numbering. This model uses the multi-layer perceptron classifier and the tf-idf vectorizer from scikit-learn. The installation process, requirements, and usage are outlined below.

# Installation
Running `./bin/install` will install all the required packages. 

In order to make the installation version-agnostic, a virtual environemtn is used and wxPython is installed without using precompiled wheels that are available for each version of Ubuntu (and other packages). This means that the installation, especially the wxwidgets package, can take a while to install. (it took around 30 minutes to install on my laptop, maybe 15 on my desktop. Your mileage may vary depending on your system) ([more information here](https://wxpython.org/blog/2017-08-17-builds-for-linux-with-pip/index.html)). Please be patient and do not exit while running install 

Lists of both of these are contained in the install directory and below, and fall into two categories: 

### Linux Packages (reqs.sys)
these are required for wxWidgets to run. These dependencies are for linux:

- dpkg-dev
- build-essential
- python3-dev
- freeglut3-dev
- libgl1-mesa-dev
- libglu1-mesa-dev
- libgstreamer-plugins-base1.0-dev
- libgtk-3-dev
- libjpeg-dev
- libnotify-dev
- libpng-dev
- libsdl2-dev
- libsm-dev
- libtiff-dev
- libwebkit2gtk-4.0-dev
- libxtst-dev

### Python Packages (requrements.txt)
these packages are required to run textParser.py, trainer.py, interface.py, and generate_hyperparams.py:

- nltk==3.8.1
- numpy==1.26.2
- pandas==2.1.4
- PyPDF2==3.0.1
- scikit_learn==1.3.0
- tqdm==4.66.1
- wxPython==4.2.1

before running this installation you should ensure you packages are all up-to-date by running `sudo apt-get update`. 


# Interface Usage
by running `./bin/run` after having run install, you can launch the interface module. This module is looking for `*.pkl` files in the models subdirectory, these are the 
neural networks that can be selected to be queried with a user's input. The default option that it looks for is NNModel.pkl (the one supplied here), but if this is not available it will look for other pkl files (like those that can be supploed by trainer.py, see below) 

You can input any text into the chatbot interface, and that text will be run as input into the MLPClassifier that has been trained on the textbook. 
A good example of usage might be taking text from one of the slides. The response the chatbot gives will include a page that it thinks the text from the slide best maps onto. 


# Trainer usage 
*This module must be run inside the virtual environment to ensure it has the right dependencies. Alternatively you could install all of the dependencies above to run them in your own environment*

### Features 
This module, contained in trainer.py, allows you to either fit an NN Model or perform a hyperparameter search using cross validation. The trained modules, contained within a .pkl file, are python classes that have the ability to predict based on text input passed into NNModel.predict()

### commandline arguments
    -o OUTPUT, --output OUTPUT

    the output file name for the data provided by the model, in csv format (should end in .csv).
    ignored it task is 'fit'. default='output.csv'



    -t {fit,search,both}, --task {fit,search,both}

    task for the trainer: If fit, should provide a set of hyperparameters from the
    hyperparams.csv, and will print out the performance on the test data, and output model as
    --name. If search, model will be cross-validated on the entire dataset for all hyperparameter
    sets in hyperparams.csv, and will output data to file chosen with --output If both, will train
    and fit on best params from the entire hyperparams.csv, and then fit and output the model.
    default='fit'



    -p PARAMS, --params PARAMS

    an integer that represents the row in hyperparams.csv that contains the set of hyperparams to
    be fit on. ignored if task is not 'fit'. default=0



    -d DATASET, --dataset DATASET

    data for model to be trained on. Default is data.csv



    -n NAME, --name NAME    

    output name for the pkl created by the fitting procedure. ignored if task is 'search' (should
    end in .pkl) default='NNModel.pkl'



    -s, --save    

    if flag passed as argument, the training function in the model will fit and save the model for
    each set oh hyperparameters, and will skip over hyperparamsets that have already been tested.
    Will save time if the search process if interrupted but takes more time for each run.
    subsequent runs should be given a different --output file



# Parser usage
*This module must be run inside the virtual environment to ensure it has the right dependencies. Alternatively you could install all of the dependencies above to run them in your own environment*

The parser was used to generate the dataset contained in data.csv. This shouldn't need to be run again unless that data is corrupted or lost somehow. The model parses all of the pdfs in the 'chapters' directory. These filenames are hardcoded in textParser.py, and so changing them is not recommended. Again, this module is useless if data already exists, but can be viewed to see the code-based parts of the parsing process. 

In addition to the parsing done in textParser.py, there was a lot of manual parsing that had to be done in particularly tricky pats of the pdf files. These were done in Adobe Acrobat, and consisted of removing some tricky characters, removing the bibliography and exercise pages from each chapter, and finding the corresponding starting page of each chapter in the full PDF (included here for ease of access, obtained from https://web.stanford.edu/~jurafsky/slp3/)


# Generating Hyperparameters 

This module (generate_hyperparms.py) is used to generate the CSV file that the training module reads from in order to train the modules. This file can be broken fairly easily if opened and saved from excel or other editors capable of opening csv files, so I would not recommend interacting with it that way (except for read-only purposes, just do not overwrite the file as generated)

This file takes the lists of different hyperparameters for the model and creates a cartesian product of all of them using itertools. The output of this is stored in hyperparams.csv. These different sets are then used for model training purposes. 

terminal output from the interface is automatically redirected to logs/interface.txt. 