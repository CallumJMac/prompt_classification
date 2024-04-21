import string
import time
import pandas as pd
import joblib
import os
import joblib

# Import preprocessing libraries
import nltk
if not nltk.corpus.stopwords.fileids():
            nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# Import Build Pipeline and Feature Extractor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class PromptClassifier:
    """
    A text classification object for analyzing prompts.
    Attributes:
        est_name (str): Name of the classifier.
        model: Sklearn classifier object.
        pipeline: Sklearn pipeline object containing the model.

    Methods:
        train(x_train, x_test, y_train, y_test, results_df, save_model=True, save_metrics=True): Train the model and save performance metrics.
        predict_classification(text): Run inference on trained pipeline with raw text.
        preprocess(text): Preprocess the input text for classification.
        save_model_performance(y_test, y_predict, elapsed_time_train, elapsed_time_inf, results_df): Calculate evaluation metrics and save to a CSV file.
        save_model(): Save the sklearn pipeline for later use.
        load_model(model_path): Load a saved model from the specified path.
    """
    def __init__(self, est_name, est_obj):
        # Initialize model and pipeline
        self.est_name = est_name
        self.model = est_obj
        self.pipeline = Pipeline([('cv',CountVectorizer()), (f'{self.est_name}', self.model)])
        
    def train(self, x_train, x_test, y_train, y_test, results_df, save_model=True, save_metrics=True):
        """
        Train the model and save performance metrics for evaluation.
        Args:
            x_train (pd.Series): Cleaned text to train the model with.
            x_test (pd.Series): Cleaned text to evaluate the trained model.
            y_train (pd.Series): Str classification labels for training data.
            y_test (pd.Series): Str classification labels for test data.
            save_model (bool): If True then save model pipeline as a .pkl file. If False then nothing.
            save_metrics (bool): If True then save evaluation metrics as .csv file. If False then nothing.
            results_df: (pd.DataFrame): To be used to store evaluation metrics.
        Returns:
            None
        """
    
        # Record the start time for training
        start_time = time.time()

        # Preprocess text
        x_train = x_train.apply(self.preprocess)
        
        # Fit the model
        self.pipeline.fit(x_train, y_train)

        # Record the end time for training
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time_train = end_time - start_time

        # Measure inference time (crudely)
        start_time = time.time()
        # Use the model to predict unseen prompts
        y_predict = self.pipeline.predict(x_test)
        end_time = time.time()
        elapsed_time_inf = end_time - start_time
        
        # Calculate and save evaluation metrics
        if save_metrics == True:
            self.save_model_performance(y_test, y_predict, elapsed_time_train, elapsed_time_inf, results_df)
        
        if save_model == True:
            # Save the model for use later
            self.save_model()


    def predict_classification(self, text):
        """
        Run inference on trained pipeline with raw text.

        Args:
            text (str): A raw prompt that a user intends to be inputted to an LLM.
        Returns:
            prediction (str): The classification of the prompt. Inappropriate/In-scope, Malicious.
            pred_prob (float): The prediction probabilities for the classes that the text classifier was trained on.
        """
        # Preprocess text
        preprocessed_text = self.preprocess(text)
        # Make prediction using the model
        prediction = self.pipeline.predict([preprocessed_text])
        # Get prediction probability 
        pred_prob = self.pipeline.predict_proba([preprocessed_text])
        return prediction[0], pred_prob
    

    def preprocess(self, text):
        """
        Preprocesses the input text for classification.
        
        Args:
            text (str): The input text to be preprocessed.
            
        Returns:
            str: The preprocessed text.
        """
        # Tokenization
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
        # Remove punctuation
        tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in tokens]
        # Convert to lowercase
        tokens = [word.lower() for word in tokens]
        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)
    
    def save_model_performance(self, y_test, y_predict, elapsed_time_train, elapsed_time_inf, results_df):
        """
        Calculate evaluation metrics (accuracy, precision, recall, f1-score) and save metrics with training and inference times to a csv file.

        Args:
            y_test (str): Ground truth labels for unseen data.
            y_predict (str): Models predicted labels for unseen data.
            elapsed_time_train: The amount of time (s) taken to train the model.
            elapsed_time_inf: The amount of time (s) taken to run inference on a single prompt.
            results_df: (pd.DataFrame): To be used to store evaluation metrics.
        Returns:
            None
        """

        accuracy = accuracy_score(y_test, y_predict)
        precision = precision_score(y_test, y_predict, average='macro') #use macro as the classes are slightly unbalanced
        recall = recall_score(y_test, y_predict, average='macro')
        f1 = f1_score(y_test, y_predict, average='macro')

        results_df.loc[f'{self.est_name.replace("_", " ")}'] = [f"{accuracy:.2f}", f"{precision:.2f}", f"{recall:.2f}", f"{f1:.2f}", f"{elapsed_time_train:.2f}", f"{elapsed_time_inf:.2f}"]

        print(f"{self.est_name}, \t\t training time {elapsed_time_train:.2f}s")

        # Name index Classifier for use in web app 
        results_df = results_df.rename_axis("Classifier")
        results_df.to_csv("model_results.csv")
    
    def save_model(self):
        """
        Save the sklearn pipeline for later use.

        Args:
            None
        Returns:
            None
        """
        # Save the model for use later
        folder_path="models"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        pipeline_file = open(f"models/{self.est_name.lower()}.pkl", "wb")
        joblib.dump(self.pipeline, pipeline_file)
        pipeline_file.close()

    def load_model(self, model_path):
         """
        Load a saved model from the specified path.

        Args:
            model_path (str): Path to the saved model file.

        Returns:
            None
        """
         self.pipeline = joblib.load(open(f'{model_path}', "rb"))
