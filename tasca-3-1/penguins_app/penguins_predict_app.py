import pickle
from flask import Flask, jsonify, request
from penguins_predict_service import predict_single

app = Flask('penguins-predict')

with open('models/penguins-lr.pck', 'rb') as f:
    scaler_lr, dv_lr, model_lr = pickle.load(f)

with open('models/penguins-svm.pck', 'rb') as f:
    scaler_svm, dv_svm, model_svm = pickle.load(f)

with open('models/penguins-dt.pck', 'rb') as f:
    scaler_dt, dv_dt, model_dt = pickle.load(f)

with open('models/penguins-knn.pck', 'rb') as f:
    scaler_knn, dv_knn, model_knn = pickle.load(f)

@app.route('/predict/lr', methods=['POST'])
def predict_lr():
    penguin = request.get_json()
    species, prob = predict_single(penguin, scaler_lr, dv_lr, model_lr)
    return jsonify({'model': 'Logistic Regression', 'species': species, 'probability': prob})

@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    penguin = request.get_json()
    species, prob = predict_single(penguin, scaler_svm, dv_svm, model_svm)
    return jsonify({'model': 'SVM', 'species': species, 'probability': prob})

@app.route('/predict/dt', methods=['POST'])
def predict_dt():
    penguin = request.get_json()
    species, prob = predict_single(penguin, scaler_dt, dv_dt, model_dt)
    return jsonify({'model': 'Decision Tree', 'species': species, 'probability': prob})

@app.route('/predict/knn', methods=['POST'])
def predict_knn():
    penguin = request.get_json()
    species, prob = predict_single(penguin, scaler_knn, dv_knn, model_knn)
    return jsonify({'model': 'KNN', 'species': species, 'probability': prob})

if __name__ == '__main__':
    app.run(debug=True, port=8000)