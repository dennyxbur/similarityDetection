import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from fastai2.vision.all import *
from collections import OrderedDict
import logging
from model_utils import *


app = Flask(__name__)

UPLOAD_FOLDER = os.getcwd() + '/temp/'
ALLOW_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOW_EXTENSIONS


@app.route('/similarity', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if ('file_base' not in request.files) and ('file_inspect' not in request.files):
            flash('No file part')
            return redirect(request.url)
        basefile = request.files['file_base']
        inspectfile = request.files['file_inspect']
        if basefile.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if basefile and allowed_file(basefile.filename):
            filename = secure_filename(basefile.filename)
            filename_inspect = secure_filename(inspectfile.filename)
            if not os.path.isdir(app.config['UPLOAD_FOLDER']):
                os.mkdir(app.config['UPLOAD_FOLDER'])
            basefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            inspectfile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_inspect))

            img_base = PILImage.create(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_inspect = PILImage.create(os.path.join(app.config['UPLOAD_FOLDER'], filename_inspect))

            # learner = load_learner(os.getcwd() + '/arch_classifier/export.pkl', cpu=True)
            learner = load_learner(os.getcwd() + '/siamese.pkl', cpu=True)
            resp = learner.predict(SiameseImage(img_base, img_inspect))
            proba = resp[2]

            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename_inspect))
            jsonRet = {
                "arch style": str(proba)
            }
            return jsonRet
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file_base><br><br>
      <input type=file name=file_inspect><br><br>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return 'File uploaded'


if __name__ == '__main__':
    # print('OK')
    app.run(host='0.0.0.0', port=5531)
