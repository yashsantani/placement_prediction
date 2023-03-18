from flask import Flask, request, jsonify
import numpy as np
import pickle
import sklearn

model = pickle.load(open('model_classifier.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    # gender = request.form.get('gender')
    # ssc_p = request.form.get('ssc_p')
    # ssc_b = request.form.get('ssc_b')
    # hsc_p = request.form.get('hsc_p')
    # hsc_b = request.form.get('hsc_b')
    # college_p = request.form.get('college_p')
    # backlogs = request.form.get('backlogs')
    # project = request.form.get('project')
    # dsa = request.form.get('dsa')
    # os = request.form.get('os')
    # networking = request.form.get('networking')
    # dbms = request.form.get('dbms')
    Gender = request.form.get('Gender')
    Age = request.form.get('Age')
    Stream = request.form.get('Stream')
    Internships = request.form.get('Internships')
    CGPA = request.form.get('CGPA')
    Hob = request.form.get('HistoryOfBacklogs')



    input_query = np.array([[Age,Gender,Stream,Internships,CGPA,Hob]])

    result = model.predict(input_query)[0]

    return jsonify({'placement': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
