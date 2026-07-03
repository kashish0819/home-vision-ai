from flask import Flask, render_template, request, send_file
import os
import uuid
import sys
from pathlib import Path

# =====================================================
# PROJECT PATH
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =====================================================
# IMPORTS
# =====================================================

from ml.predict_fusion import predict_fusion
from database.database import (
    insert_prediction,
    get_all_predictions,
    get_prediction
)
from utils.pdf_report import generate_pdf
from database.database import (
    insert_prediction,
    load_data
)

from flask import send_file

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =====================================================
# STATE GEO SCORE
# =====================================================

state_geo = {
    "Gujarat": 0.82,
    "Maharashtra": 0.76,
    "Rajasthan": 0.85,
    "Delhi": 0.58,
    "Karnataka": 0.68,
    "Tamil Nadu": 0.63,
    "Kerala": 0.48,
    "Punjab": 0.61,
    "Other": 0.75
}

# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# PREDICTION PAGE
# =====================================================

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


# =====================================================
# PREDICT
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    global last_prediction

    print("=" * 60)
    print("STEP 1 : Entered /predict route")
    print("=" * 60)

    try:

        patient_name = request.form.get("patient_name")
        age = int(request.form.get("age"))
        gender = request.form.get("gender")

        hb = float(request.form.get("hb"))
        mch = float(request.form.get("mch"))
        mchc = float(request.form.get("mchc"))
        mcv = float(request.form.get("mcv"))

        state = request.form.get("state")
        district = request.form.get("district")

        print("STEP 2 : Form Data Read Successfully")

        gender_value = 1 if gender == "Male" else 0
        geo_score = state_geo.get(state, 0.75)

        image = request.files["image"]

        filename = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        image.save(image_path)

        print("STEP 3 : Image Saved")
        print(image_path)

        print("STEP 4 : Calling predict_fusion()")

        result = predict_fusion(
            image_path,
            gender_value,
            hb,
            mch,
            mchc,
            mcv,
            geo_score
        )

        print("STEP 5 : AI Prediction Finished")
        print(result)

        prediction = result["label"]
        confidence = result["confidence"]

        print("STEP 6 : Saving Database")
        report_id = insert_prediction(
    patient_name,
    age,
    gender,
    state,
    district,
    image_path,
    hb,
    mch,
    mchc,
    mcv,
    prediction,
    prediction,
    confidence
)
        


        print("STEP 7 : Database Saved")

        last_prediction = {
            "patient": patient_name,
            "age": age,
            "gender": gender,
            "state": state,
            "district": district,
            "hb": hb,
            "mch": mch,
            "mchc": mchc,
            "mcv": mcv,
            "prediction": prediction,
            "confidence": confidence * 100
        }

        print("STEP 8 : Rendering Result Page")

        return render_template(
            "result.html",
            patient=patient_name,
            age=age,
            gender=gender,
            hb=hb,
            mch=mch,
            mchc=mchc,
            mcv=mcv,
            state=state,
            district=district,
            prediction=prediction,
            report_id=report_id,
            confidence=round(confidence * 100, 2),
            result=result
        )

    except Exception as e:

        print("=" * 60)
        print("ERROR OCCURRED")
        print(e)
        print("=" * 60)

        return f"""
        <h2>Application Error</h2>
        <pre>{e}</pre>
        """
    
last_prediction = {}

@app.route("/download-report/<int:report_id>")
def download_report(report_id):

    data = get_prediction(report_id)

    if data is None:
        return "Report not found", 404

    pdf_name = f"HemoVision_Report_{report_id}.pdf"

    generate_pdf(
        pdf_name,
        data[1],    # patient
        data[2],    # age
        data[3],    # gender
        data[4],    # state
        data[5],    # district
        data[7],    # hemoglobin
        data[8],    # mch
        data[9],    # mchc
        data[10],   # mcv
        data[11],   # prediction
        data[13] * 100
    )

    return send_file(pdf_name, as_attachment=True)


# =====================================================
# ANALYTICS
# =====================================================

@app.route("/analytics")
def analytics():

    df = load_data()

    total_patients = len(df)

    anemia_cases = len(df[df["prediction"] == "Anemia"])

    normal_cases = len(df[df["prediction"] == "Normal"])

    if total_patients > 0:
        average_confidence = round(df["confidence"].mean() * 100, 2)
    else:
        average_confidence = 0

    return render_template(
        "analytics.html",
        total_patients=total_patients,
        anemia_cases=anemia_cases,
        normal_cases=normal_cases,
        average_confidence=average_confidence,
        records=df.to_dict(orient="records")
    )

# =====================================================
# REPORTS
# =====================================================

@app.route("/reports")
def reports():

    reports = get_all_predictions()

    return render_template(
        "reports.html",
        reports=reports
    )

# =====================================================
# ABOUT
# =====================================================

@app.route("/about")
def about():
    return render_template("about.html")


# =====================================================
# CONTACT
# =====================================================

@app.route("/contact")
def contact():
    return render_template("contact.html")


# =====================================================
# RUN
# =====================================================

print(app.url_map)

if __name__ == "__main__":
    app.run(debug=True)