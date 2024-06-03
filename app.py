    
from src.ThyroidDiseaseDetection.pipelines.prediction_pipeline import CustomData,PredictPipeline

from flask import Flask,request,render_template,jsonify


app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    else:
        data = CustomData(
            sex=request.form['sex'],
            on_thyroxine=request.form['on_thyroxine'],
            query_on_thyroxine=request.form['query_on_thyroxine'],
            on_antithyroid_medication=request.form['on_antithyroid_medication'],
            sick=request.form['sick'],
            pregnant=request.form['pregnant'],
            thyroid_surgery=request.form['thyroid_surgery'],
            I131_treatment=request.form['I131_treatment'],
            query_hypothyroid=request.form['query_hypothyroid'],
            query_hyperthyroid=request.form['query_hyperthyroid'],
            lithium=request.form['lithium'],
            goitre=request.form['goitre'],
            tumor=request.form['tumor'],
            hypopituitary=request.form['hypopituitary'],
            psych=request.form['psych'],
            TSH_measured=request.form['TSH_measured'],
            T3_measured=request.form['T3_measured'],
            TT4_measured=request.form['TT4_measured'],
            T4U_measured=request.form['T4U_measured'],
            FTI_measured=request.form['FTI_measured'],
            TBG_measured=request.form['TBG_measured'],
            referral_source=request.form['referral_source'],
            age=request.form['age'],
            TSH=request.form['TSH'],
            T3=request.form['T3'],
            TT4=request.form['TT4'],
            T4U=request.form['T4U'],
            FTI=request.form['FTI']
        )

         # this is my final data
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.predict(final_data)
        
        result=round(pred[0],2)
        
        return render_template("result.html",final_result=result)

#execution begin
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)