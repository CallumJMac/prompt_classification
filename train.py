# Import Data Visualisation and EDA pkgs
import seaborn as sns
import pandas as pd

# Import classification models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Import Advai's Prompt Classification Modules
from promptify.model import PromptClassifier
from promptify.data_loader import PromptDataLoader

def main():
    # Define the relative path to the data
    DATA_PATH = "data/coding_challenge_data.csv"

    # Instatiate PromptDataLoader object
    dataloader = PromptDataLoader(DATA_PATH)

    # Loads the data from the data path
    dataloader.load_data()

    # Display the value counts of each class
    print(dataloader.df['persona'].value_counts())


    # Plot the value counts to visualise potential class imbalance
    sns.countplot(x='persona',data=dataloader.df)

    #  Split Data
    x_train, x_test, y_train, y_test = dataloader.train_test_split()

    # Model Training
    # Initialize estimators using their default parameters
    estimators = [
        ("K-Nearest_Neighbors", KNeighborsClassifier()),
        ("Support_Vector_Machine", svm.SVC(probability=True)),
        ("Logistic_Regression", LogisticRegression()),
        ("Gradient_Boosting_Classifier", GradientBoostingClassifier()),
        ("Random_Forest", RandomForestClassifier())
    ]

    # Prepare a DataFrame to keep track of the models' performance
    results = pd.DataFrame(columns=["Accuracy", "Precision", "Recall",
                                    "F1 Score", "Training Time (s)", "Inference Time (s)"])

    # Iterate through each estimator in the list
    for est_name, est_obj in estimators:
        # Initialize the NLP pipeline
        prompt_classifier = PromptClassifier(est_name, est_obj)

        # Train the model, Save the model weights & evaluation metrics
        prompt_classifier.train(x_train, x_test, y_train, y_test,results, save_model=True, save_metrics=True)

if __name__ == "__main__":
    main()