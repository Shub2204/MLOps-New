import numpy as np
import pandas as pd
import pickle 
from flask import Flask, render_template, request


loaded_model=pickle.load(open('boston_house.pkl','rb'))
app=Flask(__name__)

@app.route("/")
@app.route("/bostonindex")
def index():
	return render_template('boston_model.html')

@app.route("/predict",methods = ['POST'])
def make_predictions():
    if request.method == 'POST':
        a = request.form.get('crim')
        b = request.form.get('zn')
        c = request.form.get('indus')
        d = request.form.get('chas')
        e = request.form.get('nox')
        f = request.form.get('rm')
        g = request.form.get('age')
        h = request.form.get('dis')
        i = request.form.get('rad')
        j = request.form.get('tax')
        k = request.form.get('ptratio')
        l = request.form.get('b')
        m = request.form.get('lstat')
        
        X = np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m]])
        pred = loaded_model.predict(X)
        return render_template('boston_pred.html' , response = pred[0][0])

if __name__=='__main__':
    app.run(debug=False)


     