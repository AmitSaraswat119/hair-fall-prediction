from flask import Flask, render_template, request
from joblib import load
from tensorflow.keras.models import load_model
from flask import send_from_directory
import numpy as np

app = Flask(__name__)

model = load_model('hair_loss_prediction_model.h5')
scaler = load('scaler/scaler.joblib')

def make_prediction(data_point):
    categorical_cols = ['Genetics', 'Hormonal_Changes', 'Medical_Conditions', 
                        'Medications_&_Treatments', 'Nutritional_Deficiencies', 
                        'Stress', 'Poor_Hair_Care_Habits', 'Environmental_Factors', 
                        'Smoking', 'Weight_Loss']
    encoded_data_point = []
    for col in categorical_cols:
        encoder = load(f'encoders/{col}_encoder.joblib')
        encoded_value = encoder.transform([[data_point[col]]])
        encoded_data_point.extend(encoded_value[0])
    numerical_data = [data_point['Age']]
    scaled_data_point = np.concatenate((numerical_data, encoded_data_point))
    scaled_data_point = scaler.transform(np.array([scaled_data_point]).reshape(1, -1))
    prediction = model.predict(scaled_data_point)
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    form_data = request.form.to_dict()
    prediction = make_prediction(form_data)
    if(prediction[0][0] > 0.5):
        prediction = "High Chances of Hair fall"
    else:
        prediction =  "Low chances of Hair fall"
    
    return render_template('index.html', prediction=prediction)

@app.route('/data_report')
def data_report():
    graphs = [
        'age_density.png',
        'qq_plot_age.png',
        'medical_condition_count.png',
        'medication_treatment.png',
        'nutritional_deficiencies.png',
        'age_density_hair_loss.png'
    ]
    return render_template('data_report.html', graphs=graphs)

@app.route('/download/<path:filename>')
def download(filename):
    return send_from_directory('static/images', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True , port=5000)
