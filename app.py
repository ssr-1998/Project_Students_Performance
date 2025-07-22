import sys
from src.logger import logging
from src.exception import CustomException
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application


# Route for a Home Page
@app.route('/')
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        raise CustomException(e, sys)


@app.route('/predict_data', methods=["GET", "POST"])
def predict_datapoint():
    try:
        if request.method == "GET":
            return render_template("home.html")
        else:
            logging.info("Fetching Input Features from the Web Page")
            data = CustomData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("ethnicity"),
                parental_level_of_education=request.form.get("parental_level_of_education"),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=request.form.get("reading_score"),
                writing_score=request.form.get("writing_score")
            )
            pred_df = data.get_data_as_data_frame()

            predict_pipeline_obj = PredictPipeline()
            preds = predict_pipeline_obj.predict(pred_df)
            result = round(preds[0], 2)

            if result > 100:
                logging.info("Math Score ({}) > 100 -> Converted to 100".format(str(result)))
                result = 100

            return render_template("home.html", results=result)
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)  # Running at 127.0.0.1:5000
