from flask import Flask,render_template,request
import numpy as np
import pickle

app = Flask(__name__)

## load pickle model
model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    features = [np.array(float_feature)]
    prediction = model.predict(features)
    print(prediction)
    val = prediction[0]
    print(val)
    if val == True:
        return render_template('index.html' , value = "Startup will become successful in coming years")
    else:
        return render_template('index.html' , value = "Startup will not become successful in coming years")
        
        


if __name__ == '__main__':
    app.run(debug=True)
    
    
