from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

#app = Flask(__name__)
app = Flask(__name__, template_folder='templates')
# Load the pre-trained model
filename = 'finalized_model_forest.sav'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        Age = int(request.form['Age'])
        Monthly_Rate = int(request.form['Monthly_Rate'])
        Environment_Satisfaction = int(request.form['Environment_Satisfaction'])
        Job_Involvement = int(request.form['Job_Involvement'])
        Standard_Hours = int(request.form['Standard_Hours'])
        Performance_Rate = int(request.form['Performance_Rate'])

        # Format the input data into a numpy array
        input_data = np.array([[Age,Monthly_Rate, Environment_Satisfaction, Job_Involvement, Standard_Hours, Performance_Rate  ]])

        # Make prediction using the loaded model
        prediction = model.predict(input_data)
        print(prediction)
        # Map prediction to result
        result = 'This employee will Attrition or quit his job in the future Detected' if prediction[0] == 1 else 'This employee wont Attrition or quit his job in the future'
        #result=prediction
        print(result)

        return render_template('output.html', result=result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

