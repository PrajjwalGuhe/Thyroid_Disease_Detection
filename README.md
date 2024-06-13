# Thyroid_Disease_Detection

!!! README file is to be update !!!

In the app.py file, the @app.route("/predict", methods=["GET", "POST"]) decorator defines a route for both GET and POST requests. When a GET request is made to /predict, the form.html template is rendered, allowing users to fill out the form.
When the form is submitted (POST request), the Flask application retrieves the form data using request.form, creates a dictionary data with the form values, and then creates a CustomData object from it. The CustomData class (not shown in the provided code) is likely responsible for preprocessing the data and converting it into a DataFrame format.
Next, a PredictPipeline object is created, and the predict method is called with the DataFrame to obtain the prediction. Finally, the result.html template is rendered, passing the prediction as a parameter.
