# Import EDA Pkgs
import pandas as pd
from sklearn.model_selection import train_test_split


class PromptDataLoader:
    """
    A class for loading and preprocessing data for prompt classification.

    Attributes:
        data_path (str): Path to the CSV file containing the dataset.

    Methods:
        load_data(): Load the data from a CSV file into a pandas DataFrame.
        train_test_split(test_size=0.1, random_state=69): Split the DataFrame into training and testing data.
    """
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load_data(self):
        """
        Load the data from a csv file to a pandas.DataFrame.

        Args:
            None
        Returns:
            None
        """
        self.df = pd.read_csv(self.data_path, delimiter='|')
    
    
    def train_test_split(self, test_size=0.1, random_state=69):
        """
        Split the DataFrame into training and testing data for inputting to the PromptClassifier object.
        
        Args:
            test_size (float): The proportion of dataset to be used for unseen test data to evaluate model performance [0, 1].
            random_state (int): For recreating results.
        Returns:
            x_train (pd.Series): Cleaned text to train the model with.
            x_test (pd.Series): Cleaned text to evaluate the trained model.
            y_train (pd.Series): Str classification labels for training data.
            y_test (pd.Series): Str classification labels for test data.
        
        """
        # Split DataFrame into prompts and labels
        x_prompts = self.df['query']
        y_labels = self.df['persona']

        #  Split Data
        x_train, x_test, y_train, y_test = train_test_split(x_prompts, y_labels, test_size=test_size, random_state=random_state)

        # Check number of training and testing samples
        print(f"# of Training Samples: {len(x_train)}")
        print(f"# of Testing Samples: {len(x_test)}")
        
        return x_train, x_test, y_train, y_test