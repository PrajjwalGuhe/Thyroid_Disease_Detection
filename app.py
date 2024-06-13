from src.ThyroidDiseaseDetection.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.ThyroidDiseaseDetection.logger import logging
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    # Processing form data and making predictions
    else:
        # Define list of required keys
        required_keys = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 
                         'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 
                         'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 
                         'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'T3_measured', 
                         'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 
                         'referral_source', 'age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
        
        logging.info("Received form data: %s", request.form)
        
        # Check if all required keys are present
        if not all(key in request.form for key in required_keys):
            missing_keys = [key for key in required_keys if key not in request.form]
            return jsonify({"status": "error", "message": f"Missing keys: {', '.join(missing_keys)}"}), 400

        # Extract form data
        data = {
            'sex': request.form['sex'],
            'on_thyroxine': request.form['on_thyroxine'],
            'query_on_thyroxine': request.form['query_on_thyroxine'],
            'on_antithyroid_medication': request.form['on_antithyroid_medication'],
            'sick': request.form['sick'],
            'pregnant': request.form['pregnant'],
            'thyroid_surgery': request.form['thyroid_surgery'],
            'I131_treatment': request.form['I131_treatment'],
            'query_hypothyroid': request.form['query_hypothyroid'],
            'query_hyperthyroid': request.form['query_hyperthyroid'],
            'lithium': request.form['lithium'],
            'goitre': request.form['goitre'],
            'tumor': request.form['tumor'],
            'hypopituitary': request.form['hypopituitary'],
            'psych': request.form['psych'],
            'TSH_measured': request.form['TSH_measured'],
            'T3_measured': request.form['T3_measured'],
            'TT4_measured': request.form['TT4_measured'],
            'T4U_measured': request.form['T4U_measured'],
            'FTI_measured': request.form['FTI_measured'],
            'TBG_measured': request.form['TBG_measured'],
            'referral_source': request.form['referral_source'],
            'age': float(request.form['age']),
            'TSH': float(request.form['TSH']),
            'T3': float(request.form['T3']),
            'TT4': float(request.form['TT4']),
            'T4U': float(request.form['T4U']),
            'FTI': float(request.form['FTI'])
        }

        # Process the data for prediction
        custom_data = CustomData(data)
        data_df = custom_data.get_data_as_dataframe()

        pipeline = PredictPipeline()
        prediction = pipeline.predict(data_df)

        print("Raw predictions:", prediction)

        hyper_threshold = 0.7
        hypo_threshold = 0.3

        if prediction[0] >= hyper_threshold:
            result = "Hyperthyroid"
        elif prediction[0] <= hypo_threshold:
            result = "Hypothyroid"
        else:
            result = "No thyroid condition detected"

        print("Predicted result:", result) 

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
