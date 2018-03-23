Our code can be found on GitHub at https://github.com/ewells1/Information-Extraction/tree/master/Coref-Resolution. 

For our project, we moved all of the .v4_auto_conll files into just three folders: train, test, and dev. Organize_data.py was used to do this.

For our classifier, we have two main files. Read_data.py has all of the code for reading in the data and putting it into a usable format. Bi_lstm.py has all of the code for classification. 

To run our classifier, just run bi_lstm.py. Parameters are set in the main section at the bottom of the file.
