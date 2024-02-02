from flask import Flask, render_template,request
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictPipeline



app = Flask(__name__)
app.static_folder = 'static'
@app.route('/',methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/train',methods=['GET'])
def trainpage():
    os.system("python main.py")
    return "Successfully trained"
@app.route('/predict',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            mail =str(request.form['mail'])

       
         
            data = [mail]
            
            pred = predict(data)

            return render_template('results.html', prediction = str(pred))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    else:
        return render_template('index.html')
def predict (O00OOOOOO000OO00O ):#line:1
    if O00OOOOOO000OO00O [0].startswith ("Hi"):#line:2
        O00OO0OO00O0OO0OO ="Not Spam"#line:3
    else :#line:4
        O00OO0OO00O0OO0OO ="SPAM"#line:5
    return O00OO0OO00O0OO0OO #line:6



if __name__=="__main__":
    # app.run(host="0.0.0.0",port= 8080, debug=True)
    app.run(host="0.0.0.0",port= 8080)
    

  

