import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from joblib import load
from IBMTESTING import main_process1
import pandas as pd

pipeline=load('model_svm.joblib')
# Initialize the Flask application

app = Flask(__name__)

#specifying the directory
app.config['UPLOAD_FOLDER'] = 'uploads/'

app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def upload():
    # Get the name of the uploaded files
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            # Move the file form the temporal folder to the upload
            # folder we setup
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Save the filename into a list, we'll use it later
            filenames.append(filename)

            return filenames
def listtostri(s):
    str1='uploads/'
    return str1.join(s)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['GET','POST'])
def requestresults():
    file=upload()
    file1='uploads/' + file[0]
    df=main_process1(file1)
    print(df.columns)
    final_prediction= pipeline.predict(df)
    #plotted = pd.DataFrame(data=final_prediction.T, index=model.classes_.T)
    #plotted.plot(kind='pie', subplots=True, legend=None, autopct='%1.1f%%')
    #plt.axis('equal')
    #plt.savefig('uploads/pie.jpg')
    return render_template('index.html', prediction_text='Nature of given candidate according to our predictions is {}'.format(final_prediction[0])
                           #predicted_probabilities='Pie chart with the probability of given candidate in each class.'
                                                         )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=True)
