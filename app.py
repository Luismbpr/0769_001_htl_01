import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask, render_template, request

app = Flask(__name__)

loaded_model = joblib.load(MODEL_OUTPUT_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        ##If index.html is sending a 'POST'
        ## It first sends a lead_time id="lead_time". Typecast into Integer (just in case it was a string)
        lead_time = int(request.form["lead_time"])
        no_of_special_requests = int(request.form["no_of_special_requests"])
        avg_price_per_room = float(request.form["avg_price_per_room"])
        arrival_month = int(request.form["arrival_month"])
        arrival_date = int(request.form["arrival_date"])
        market_segment_type = int(request.form["market_segment_type"])
        no_of_week_nights = int(request.form["no_of_week_nights"])
        no_of_weekend_nights = int(request.form["no_of_weekend_nights"])
        type_of_meal_plan = int(request.form["type_of_meal_plan"])
        room_type_reserved = int(request.form["room_type_reserved"])
        
        ## Convert into NumPy Array
        features = np.array([[lead_time,no_of_special_requests,avg_price_per_room,arrival_month,arrival_date,market_segment_type,no_of_week_nights,no_of_weekend_nights,type_of_meal_plan,room_type_reserved]])

        # Load model and make prediction of the features
        prediction = loaded_model.predict(features)
        
        ## render_template Helps Show the result in the html file
        ## When printing prediction it shows array([1]), so need to use prediction[0] to get only the [1]
        return render_template('index.html', prediction=prediction[0])
    
    ## return render_template if this condition is not true if request.method == "POST"
    ## Will show index.html (it should always be running) but without any prediction
    return render_template("index.html", prediction=None)

## For GCP Cloud Deployment it should be in port=8080
## Flask by default runs at port=5000
if __name__=="__main__":
    app.run(host='0.0.0.0' , port=8080)
    #app.run(host='0.0.0.0' , port=5000)