# Import Data Visualisation and EDA pkgs
import seaborn as sns
import pandas as pd

# Import classification models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Import Prompt Classification Modules
# import sys
# sys.path.insert(1, '/path/to/application/app/folder')
from promptify.model import PromptClassifier

def main():

    # Initialize data set location and file name
    data_file_path = "data_new/"
    data_file_name_train = "train-00000-of-00001-9564e8b05b4757ab"
    data_file_name_test = "test-00000-of-00001-701d16158af87368"
    data_file_ext = ".parquet"

    # Loading data set into a pandas DataFrame
    data_train = pd.read_parquet(data_file_path + data_file_name_train + data_file_ext)
    data_test = pd.read_parquet(data_file_path + data_file_name_test + data_file_ext)

    # Rename "text" column into "prompt"
    data_train.rename(columns={"text":"prompt"}, inplace=True)
    data_test.rename(columns={"text":"prompt"}, inplace=True)

    # Split DataFrame into prompts and labels
    x_train = data_train['prompt']
    y_train = data_train['label']
    x_test = data_test['prompt']
    y_test = data_test['label']


    # Check number of training and testing samples
    print(f"# of Training Samples: {len(x_train)}")
    print(f"# of Testing Samples: {len(x_test)}")

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