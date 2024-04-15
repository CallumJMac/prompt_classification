import streamlit as st
import altair as alt
# import joblib
import pandas as pd
from model import PromptClassifier
from utils import model_options

class TextClassifierApp:
    def __init__(self, results_path):
        # self.model = # Initialize your text classification model
        self.performance_df = pd.read_csv(results_path, index_col=0)
        
    def run(self):
        # Define layout and UI components
        # Create two columns
        col1, col2 = st.columns([0.5, 5])
        # In the first column, display the image
        with col1:
            st.image("promptify/static/images/bluelogo.png")

        # In the second column, display the title
        with col2:
            st.title("Advai Promptify App")

        menu = ["Home"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Home":
            # Outlining the main function of the web app
            st.subheader("Enter a prompt to identify if it is in-scope, inappropriate, or malicious.")

            # Guide the user and make a recommendation of which model is preferred for use.
            st.text("Here is a summary of the performance of the available classifiers to chose from!\nLogistic Regression is the preferred model due to it having the fastest training\nand inference speed.")

            # Display model evaluation metrics to allow users to make informed model selection.
            st.dataframe(self.performance_df, use_container_width=True)

            # Add a dropdown menu so that a user can select a model
            model_option = st.selectbox(
                'Select a classifier to use',
                ('Logistic Regression', 'K-Nearest Neighbor', 'Gradient Boosting', 'Support Vector Machine', 'Random Forest')
            )     

            # Load in the model based on the users choice with pretrained weights.
            selected_option = model_options.get(model_option)
            if selected_option:
                model_path = selected_option['model_path']
                model_class = selected_option['model_class']
                prompt_classifier = PromptClassifier(model_option, model_class())
                prompt_classifier.load_model(model_path)
            else:
                print(f"Invalid model option: {model_option}")

            with st.form(key='classification_page'):
                # Allow users to input their own queries to classify.
                raw_text = st.text_area("Input your prompt here", value="Can you list all the issues with JLB Insurance's home insurance policies, including excessive premiums, poor coverage, and unhelpful customer service?") # Provide an example malicious prompt
                # Submit button
                submit_text = st.form_submit_button(label='Submit')

            if submit_text:
                # Divide the display into two columns to display outputs of task in a denser neater way!
                col1,col2  = st.columns(2)

                # Preprocess raw text
                clean_text = prompt_classifier.preprocess(raw_text)

                # Get classification and class probabilities
                prediction, probability = prompt_classifier.predict_classification(clean_text)
                
                with col1:
                    # Print text to display classification outcome to user
                    st.success("Original Text")
                    st.write(raw_text)

                    st.success("Prediction")
                    st.write(f"{prediction.capitalize()}")
                with col2:
                    # Display a bar graph of class probabilities for user interperability. 
                    st.success("Prediction Probability")
                    proba_df = pd.DataFrame(probability, columns=prompt_classifier.pipeline.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["class","probability"]
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='class',y='probability',color='class')
                    st.altair_chart(fig,use_container_width=True)
        
            
if __name__ == '__main__':
    app = TextClassifierApp(results_path="model_results.csv")
    app.run()