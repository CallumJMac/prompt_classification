# LLM Prompt Evaluation Challenge

The name of the solution provided is **promptify**.

This name was chosen as the solution as the objective is to take **prompt**s and class**ify**  them.

## Overview
We have been provided with labelled text data and therefore the simplest and easiest way to treat the challenge is as a **text classification problem**. The best practice for building machine learning systems is to start simple and iterate. Further, as the project requirements specify a lightweight and scalable solution, classic machine learning methods were used with the initial model being a logistic regression model. 

### Data Preprocessing

The following preprocessing steps were undertaken to prepare the text data to be inputted to our NLP pipeline. This was undertaken remove any data that is not semantically meaningful, and to reduce the size of the feature vector inputted to the model which makes the system more scalable and computationally efficient.

1. **Tokenization**: The text is tokenized using the word_tokenize() function from the NLTK library, splitting it into individual words or tokens.

2. **Stopword Removal**: Stopwords, which are common words that often do not carry significant meaning (e.g., "the," "is," "and"), are removed from the tokenized text. This step helps reduce noise in the data and focuses on more informative terms.

3. **Punctuation Removal**: Punctuation marks, such as periods, commas, and quotation marks, are removed from the tokenized text. This step helps standardize the representation of words and improves consistency in the data.

4. **Lowercasing**: All remaining tokens are converted to lowercase. This normalization step ensures that words with different cases (e.g., "Word" and "word") are treated as identical, reducing the vocabulary size and simplifying subsequent analyses.

5. **Stemming**: Reduces words to their base or root form to standardize text, reducing vocabulary size and improving generalization.

### Models used

The machine learning methods implemented include:
- **Support Vector Machine (SVM)**: SVM is a powerful supervised learning algorithm used for classification tasks. It works by finding the hyperplane that best separates the classes in the feature space. SVM is effective for three-class text classification because it can handle high-dimensional feature spaces efficiently. It works well with sparse data, making it suitable for text data represented as Count Vectorizer vectors. SVM also allows for the use of different kernel functions to capture complex relationships between features. Its ability to find the optimal decision boundary can be advantageous for distinguishing between three classes in text data.

- **Logistic Regression**: Logistic regression is a linear model used for binary classification tasks. It models the probability of a binary outcome using a logistic function. Despite being a binary classifier, logistic regression can be extended to handle multi-class classification using techniques like One-vs-Rest (OvR) or softmax regression. For three-class text classification problems, logistic regression can provide a simple and interpretable baseline model. It's computationally efficient and works well when the relationship between the features and the target variable is approximately linear.
- **Random Forest**: Random Forest is robust and flexible, capable of capturing complex relationships in the data. It works well with both numerical and categorical features, making it suitable for text classification problems where feature representations or word embeddings.
- **Gradient Boosting**: Gradient Boosting is an ensemble learning technique that builds an ensemble of weak learners (typically decision trees) sequentially, where each new model corrects the errors of the previous ones. It performs well with high-dimensional and sparse data, making it suitable for text data represented as feature vectors. Gradient Boosting often achieves high accuracy and generalization performance by sequentially improving the model's predictions. 
- **K-Nearest Neighbour**:  KNN is a simple and intuitive classification algorithm that works by finding the most similar instances (neighbors) to a given query instance based on a distance metric (e.g., Euclidean distance) in the feature space. It does not require training, making it particularly useful for online learning or when dealing with streaming data.

### (Optional) Streamlit Web App

A streamlit web application as been prototyped which allows real-time prompt input and classification to enhance user interaction. Streamlit was chosen as it allows researchers to rapidly prototype user interfaces when compared to traditional web application technologies such as JavaScript. If this project were to be developed further, it is recommended that the project should have an API built with a python library (like django, flask, or fastAPI) tp provide scalability, flexibility, and more control over the backend architecture, making it suitable for larger-scale deployments and production environments.

## Installation

### Create Virtual Environment

This guide will walk you through setting up and using a virtual environment for your project. Virtual environments are a great way to isolate project dependencies and avoid conflicts between different projects.

#### Prerequisites:

Before you begin, make sure you have the following installed on your system:

- Python (3.x recommended)
- virtualenv or venv (usually comes pre-installed with Python)

#### Steps:
1. **Clone your project repository** from the remote repository using Git. If you don't have a repository yet, create one and initialize it with your project files.
```
git clone <repository_url>
cd <project_directory>
```
2. **Create a Virtual Environment**:
Navigate to your project directory and create a new virtual environment using either virtualenv or venv. Replace <environment_name> with your desired environment name.

```
# Using venv (Python 3)
python3 -m venv <environment_name>
```

3. **Activate the Environment**: Activate the virtual environment you just created.
```
# On Windows
<environment_name>\Scripts\activate

# On Unix or MacOS
source <environment_name>/bin/activate
```

4. **Install Dependencies**: Install project dependencies using pip. You can do this by using the requirements.txt file. 
```
# Use the requirements.txt file
pip install -r requirements.txt
```

5. **Run the Project**: You can run the project within the activated virtual environment. All dependencies will be isolated to this environment, ensuring no conflicts with other projects.

6. **Deactivate the Environment**: When you're done working on your project, you can deactivate the virtual environment.
```
deactivate
```



## Example Usage

This section provides a basic guide on how to use the `promptify` (prompt-classify) package. Below are examples demonstrating how to import and use different components of the package.

### train.py

train.py is a Python script designed to train text classification models on a given dataset and evaluate their performance. It uses various machine learning algorithms and provides insights into the models' accuracy, precision, recall, F1 score, training time, and inference time.

**Run train.py to output all necessary files for running the streamlit web app.**

```
python train.py
```
This will output model `.pkl` files for running inference on unseen data, and also `model_results.csv` to allow users to inspect model performance. 

### `promptify` module

The promptify module is composed of various files and classes necessary for this text classification task. Including:

- `app.py`: Runs the streamlit app for the promptify solution on your machine locally. To get started execute the following instruction from the base project directory.
```
streamlit run promptify/app.py 
```
- `data_loader.py`: A class for loading and preprocessing data for prompt classification.
- `model.py`: A text classification object for analyzing prompts.
- `utils.py`: contains a dictionary `model_options` that configures various machine learning models along with their corresponding classes and paths to saved model files.


