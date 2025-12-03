from flask import Flask, request, jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# import ridge regression and  standard scaler pickle
ridge_model=pickle.load(open('saved_models/Ridge.pkl','rb'))
standard_scaler=pickle.load(open('saved_models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict():
    if request.method=="POST":
        Temprature=float(request.form.get('Temprature'))
        RH =float(request.form.get('RH'))
        Ws =float(request.form.get('Ws'))
        Rain =float(request.form.get('Rain'))
        FFMC =float(request.form.get('FFMC'))
        DMC =float(request.form.get('DMC'))
        ISI =float(request.form.get('ISI'))
        Classes =float(request.form.get('Classes'))
        Region =float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temprature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('predict.html',results=result[0])
    else:
        return render_template('predict.html')

if __name__=="__main__":
    app.run(debug=True)
    