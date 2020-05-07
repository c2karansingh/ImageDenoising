import tensorflow.keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__)

MODEL_PATH = 'models/final.h5'
model = tensorflow.keras.models.load_model(MODEL_PATH)

@app.route('/plot', methods=['GET', 'POST'])
def plot():    
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    
    if request.method == 'POST':
        # Get the image from post request
        image_file = request.files['file']
        filename = image_file.filename
        filepath = os.path.join('uploads', filename)
        image_file.save(filepath)
    
        noise_str = filepath
        img_clean = cv2.imread(noise_str)
        img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)/255
        img_test = cv2.imread(noise_str)
        img_test = img_test/255
        
        img_testb,img_testg,img_testr = cv2.split(img_test)
        
        img_testr = img_testr.reshape(1, img_clean.shape[0], img_clean.shape[1], 1)  
        img_testg = img_testg.reshape(1, img_clean.shape[0], img_clean.shape[1], 1)  
        img_testb = img_testb.reshape(1, img_clean.shape[0], img_clean.shape[1], 1)  
        
        y_predict_r = model.predict(img_testr)
        y_predict_g = model.predict(img_testg)
        y_predict_b = model.predict(img_testb)
        
        img_out_r = y_predict_r.reshape(img_clean.shape[0], img_clean.shape[1], 1)
        img_out_g = y_predict_g.reshape(img_clean.shape[0], img_clean.shape[1], 1)
        img_out_b = y_predict_b.reshape(img_clean.shape[0], img_clean.shape[1], 1)
        
        img_out = cv2.merge( (img_out_b,img_out_g,img_out_r) )
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        img_out = np.clip(img_out, 0, 1)
        
        print('saving at C:\\Python projects\\Image denoising\\static\\images'+ filename)
        plt.imsave('C:\\Python projects\\Image denoising\\static\\images\\'+ filename, img_out)

    return render_template('test.html', url ='/static/images/'+filename)

@app.route('/page', methods=['GET', 'POST'])
def page():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('page.html')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


if __name__ == '__main__':
    app.run()