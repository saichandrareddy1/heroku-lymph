from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle



app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def Hello():
       return render_template("index.html") 


@app.route('/predict',methods=['POST'])
def predict():

    feature1 = request.form['Feature1_Algo1']
    feature2 = request.form['Feature2_Algo1']
    feature3 = request.form['Feature3_Algo1']
    feature4 = request.form['Feature4_Algo1']

    li = [feature1, feature2, feature3, feature4]

    integer_ = [int(i) for i in li]
    final_features = [np.array(integer_)]
    predict_value = model.predict(final_features)

    if predict_value[0] == 0:
        output = "normal find"
    elif predict_value[0] == 1:
        output = "metastases"
    elif predict_value[0] == 2:
        output = "malign lymph"
    else:
        output = "fibrosis"

    return render_template('index.html',
    prediction_text_Algo1=f' patient was {output}')

@app.route('/predict2',methods=['POST'])
def predict2():

    feature1 = request.form['Feature1_Algo2']
    feature2 = request.form['Feature2_Algo2']
    feature3 = request.form['Feature3_Algo2']
    feature4 = request.form['Feature4_Algo2']

    li = [feature1, feature2, feature3, feature4]

    integer_ = [int(i) for i in li]
    final_features = [np.array(integer_)]
    predict_value = model.predict(final_features)

    if predict_value[0] == 0:
        output = "normal find"
    elif predict_value[0] == 1:
        output = "metastases"
    elif predict_value[0] == 2:
        output = "malign lymph"
    else:
        output = "fibrosis"

    return render_template('index.html',
    prediction_text_Algo2=f' patient was {output}')

@app.route('/predict3',methods=['POST'])
def predict3():

    feature1 = request.form['Feature1_Algo3']
    feature2 = request.form['Feature2_Algo3']
    feature3 = request.form['Feature3_Algo3']
    feature4 = request.form['Feature4_Algo3']

    li = [feature1, feature2, feature3, feature4]

    integer_ = [int(i) for i in li]
    final_features = [np.array(integer_)]
    predict_value = model.predict(final_features)

    if predict_value[0] == 0:
        output = "normal find"
    elif predict_value[0] == 1:
        output = "metastases"
    elif predict_value[0] == 2:
        output = "malign lymph"
    else:
        output = "fibrosis"

    return render_template('index.html',
    prediction_text_Algo3=f' patient was {output}')



if __name__ == "__main__":
    app.run(debug=True)