from flask import Flask, jsonify, request
from promptify.model import PromptClassifier
from promptify.utils import model_options


app = Flask(__name__)

@app.post('/predict')
def predict():
    model_name = 'Logistic Regression'
    model_path = model_options[model_name]['model_path']
    prompt_classifier = PromptClassifier(model_name, model_path)
    data = request.json
    print(data)
    try:
        sample = data['text']
    except KeyError:
        return jsonify({'error': 'No text sent'})
    
    # sample = [sample]
    # Preprocess raw text
    clean_text = prompt_classifier.preprocess(sample)
    prediction, probability = prompt_classifier.predict_classification(sample)
    try:
        result = jsonify(prediction[0])
    except TypeError as e:
        result = jsonify({'error': str(e)})
    return result
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)