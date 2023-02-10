import tensorflow as tf
from keras.models import load_model
from flask import Flask,request, render_template
import cv2
import numpy as np
import base64

app = Flask(__name__)

model=load_model('app1\ClassificationPeriodontitis10epd2Lpara.h5')


@app.route('/')
def hello_world():
    return render_template("Perio.html")

@app.route('/',methods=["GET","POST"])
def predict():
    
    file= request.files.get('image')
    if file is None:
        return 'No file input found in the request'
    
    file_bytes = file.read()
    if file_bytes is None:
        return 'Unable to read file data'

    try:
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except cv2.error:
        # Try saving the file to disk and reading it back using cv2.imread
        with open('temp.jpg', 'wb') as f:
            f.write(file_bytes)
        image = cv2.imread('temp.jpg')
        if image is None:
            return 'Unable to decode file as an image'
    
    imgC = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resize = tf.image.resize(imgC, (350,350))

    prediction=model.predict(np.expand_dims(resize/255,0))

    output='{0:.{1}f}'.format(prediction[0][0], 2)

    base64_string = base64.b64encode(file_bytes).decode('utf-8')

    if output>str(0.5):        
        return render_template('Perio.html',pred='Your Gum is SUSPECTED having Periodontitis.\nProbability of this is Periodontitis is {}'.format(output), base64image = base64_string)
    else:
        return render_template('Perio.html',pred='Your Gum is SAFE from Periodontitis.\n Probability of this is Periodontitis is {}'.format(output), base64image = base64_string)


if __name__ == '__main__':
    app.run()