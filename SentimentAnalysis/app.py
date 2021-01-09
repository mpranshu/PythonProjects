# -*- coding: utf-8 -*-
from flask import Flask,render_template,url_for,request
import pickle
import preprocessing

# load the model from disk
svc_model = pickle.load(open('svc_model.pkl', 'rb'))
document_model = pickle.load(open('inverse_document_frequency_model.pkl','rb'))


app = Flask(__name__)

'''
Modern web applications use meaningful URLs to help users. Users are more likely to like a page and come back if the page uses a meaningful URL they can remember and use to directly visit a page.

Use the route() decorator to bind a function to a URL.

@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello():
    return 'Hello, World'

'''
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/prediction',methods=['POST'])
def prediction():
    if request.method == 'POST':
        message = request.form['message']
        if(len(message)>2):
            text = [message]
            print("text:",text)
            data = preprocessing.text_Preprocessing(text)# call to function in preprocessing.py to stammer, tokenize and prune
            vect = document_model.transform(data)
            my_prediction = svc_model.predict(vect)
        else:
            my_prediction=3
        
    return render_template('home.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True,host='0.0.0.0')

#Source for data: https://www.kaggle.com/snap/amazon-fine-food-reviews?select=Reviews.csv
