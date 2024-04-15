from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

model_options = {
                'Logistic Regression': {
                    'model_class': LogisticRegression,
                    'model_path': "models/logistic_regression.pkl"
                },
                'K-Nearest Neighbor': {
                    'model_class': KNeighborsClassifier,
                    'model_path': "models/k-nearest_neighbors.pkl"
                },
                'Gradient Boosting': {
                    'model_class': GradientBoostingClassifier,
                    'model_path': "models/gradient_boosting_classifier.pkl"
                },
                'Support Vector Machine': {
                    'model_class': svm.SVC,
                    'model_path': "models/support_vector_machine.pkl"
                },
                'Random Forest': {
                    'model_class': RandomForestClassifier,
                    'model_path': "models/random_forest.pkl"
                }
            }