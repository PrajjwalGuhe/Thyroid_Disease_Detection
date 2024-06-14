from src.ThyroidDiseaseDetection.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.ThyroidDiseaseDetection.logger import logging
from flask import Flask, request, render_template, jsonify
import pymongo
from config import MONGO_URI, DB_NAME

app = Flask(__name__)

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
predictions_collection = db["predictions"]

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    # Process form data and make predictions
    else:
        # Extract data from index.html form
        name = request.args.get('name')
        number = request.args.get('number')
        
        # Validate and process the rest of the form data
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
        
        # Extract the remaining form data
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
            prediction_result = "Hyperthyroid"
        elif prediction[0] <= hypo_threshold:
            prediction_result = "Hypothyroid"
        else:
            prediction_result = "No thyroid condition detected"
        
        print("Predicted result:", prediction_result)
        
        # Save results to MongoDB
        prediction_record = {
            'name': name,
            'number': number,
            'input_data': {
                'sex': data['sex'],
                'on_thyroxine': data['on_thyroxine'],
                'query_on_thyroxine': data['query_on_thyroxine'],
                'on_antithyroid_medication': data['on_antithyroid_medication'],
                'sick': data['sick'],
                'pregnant': data['pregnant'],
                'thyroid_surgery': data['thyroid_surgery'],
                'I131_treatment': data['I131_treatment'],
                'query_hypothyroid': data['query_hypothyroid'],
                'query_hyperthyroid': data['query_hyperthyroid'],
                'lithium': data['lithium'],
                'goitre': data['goitre'],
                'tumor': data['tumor'],
                'hypopituitary': data['hypopituitary'],
                'psych': data['psych'],
                'TSH_measured': data['TSH_measured'],
                'T3_measured': data['T3_measured'],
                'TT4_measured': data['TT4_measured'],
                'T4U_measured': data['T4U_measured'],
                'FTI_measured': data['FTI_measured'],
                'TBG_measured': data['TBG_measured'],
                'referral_source': data['referral_source'],
                'age': data['age'],
                'TSH': data['TSH'],
                'T3': data['T3'],
                'TT4': data['TT4'],
                'T4U': data['T4U'],
                'FTI': data['FTI']
            },
            'prediction': prediction_result,
            'prediction_score': float(prediction[0])
        }
        
        db_result = predictions_collection.insert_one(prediction_record)
        print(f"Inserted prediction document ID: {db_result.inserted_id}")
        
        return render_template('result.html', name=name, number=number, prediction=prediction_result, score=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
