
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import joblib
import numpy as np
import os
import re

from werkzeug.utils import secure_filename
from pypdf import PdfReader

from PIL import Image, ImageEnhance, ImageOps
import pytesseract


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# TESSERACT OCR
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    print("✅ Tesseract OCR found!")
    print("Path:", TESSERACT_PATH)

    try:
        print(
            "Tesseract Version:",
            pytesseract.get_tesseract_version()
        )
    except Exception as e:
        print("⚠️ Tesseract found but could not start.")
        print(e)

else:

    print("⚠️ Tesseract OCR not found.")
    print("Expected path:", TESSERACT_PATH)


# ============================================================
# UPLOAD SETTINGS
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)

ALLOWED_EXTENSIONS = {
    "pdf",
    "jpg",
    "jpeg",
    "png"
}


# ============================================================
# LOAD ML MODELS
# ============================================================

print("\n===================================")
print("      LOADING AI HEALTHCARE MODEL")
print("===================================")

try:

    # Lightweight KNN model
    # Size: approximately 5.86 MB

    knn_model = joblib.load(
        os.path.join(
            BASE_DIR,
            "knn_model_light.pkl"
        )
    )

    scaler = joblib.load(
        os.path.join(
            BASE_DIR,
            "scaler.pkl"
        )
    )

    encoder = joblib.load(
        os.path.join(
            BASE_DIR,
            "label_encoder.pkl"
        )
    )

    print("✅ ML models loaded successfully!")
    print("✅ Lightweight KNN model loaded!")
    print("Model: knn_model_light.pkl")

except Exception as e:

    print("\n❌ MODEL LOADING ERROR")
    print("-----------------------------------")
    print(e)
    print("-----------------------------------")

    raise


# ============================================================
# SYMPTOM COLUMNS
# ============================================================

symptom_columns = list(
    scaler.feature_names_in_
)

print(
    "Symptoms:",
    len(symptom_columns)
)

print(
    "Diseases:",
    len(encoder.classes_)
)


# ============================================================
# SYMPTOM ALIASES
# ============================================================

symptom_aliases = {

    "shortness of breath": [
        "difficulty breathing",
        "trouble breathing",
        "breathing problem",
        "breathlessness",
        "hard to breathe",
        "shortness of breath"
    ],

    "abdominal pain": [
        "stomach pain",
        "stomach ache",
        "belly pain",
        "pain in stomach",
        "abdominal pain"
    ],

    "sharp chest pain": [
        "chest pain",
        "pain in chest",
        "chest hurts",
        "pain around chest",
        "sharp chest pain"
    ],

    "cough": [
        "coughing",
        "cough"
    ],

    "dizziness": [
        "feeling dizzy",
        "dizzy",
        "lightheaded",
        "light headed",
        "dizziness"
    ],

    "sore throat": [
        "throat pain",
        "painful throat",
        "sore throat"
    ],

    "headache": [
        "head pain",
        "pain in head",
        "head ache",
        "headache"
    ],

    "vomiting": [
        "vomit",
        "vomiting",
        "throwing up"
    ],

    "fever": [
        "high temperature",
        "temperature",
        "fever"
    ],

    "nausea": [
        "feeling sick",
        "sick feeling",
        "nauseous",
        "nausea"
    ],

    "fatigue": [
        "tired",
        "very tired",
        "weakness",
        "feeling weak",
        "fatigue"
    ],

    "joint pain": [
        "pain in joints",
        "joint ache",
        "joints hurt",
        "joint pain"
    ],

    "back pain": [
        "pain in back",
        "back ache",
        "back hurts",
        "back pain"
    ]
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# EXTRACT SYMPTOMS
# ============================================================

def extract_symptoms(text):

    text = clean_text(text)

    detected = set()

    # --------------------------------------------------------
    # Dataset symptom matching
    # --------------------------------------------------------

    for symptom in symptom_columns:

        symptom_lower = clean_text(
            symptom
        )

        if symptom_lower in text:

            detected.add(
                symptom
            )

    # --------------------------------------------------------
    # Alias matching
    # --------------------------------------------------------

    for symptom, aliases in symptom_aliases.items():

        if symptom not in symptom_columns:
            continue

        for alias in aliases:

            alias = clean_text(
                alias
            )

            if alias in text:

                detected.add(
                    symptom
                )

                break

    return sorted(
        list(detected)
    )


# ============================================================
# CREATE FEATURE VECTOR
# ============================================================

def create_feature_vector(
    detected_symptoms
):

    vector = [

        1 if symptom in detected_symptoms
        else 0

        for symptom in symptom_columns

    ]

    return np.array(
        vector,
        dtype=np.float64
    ).reshape(
        1,
        -1
    )


# ============================================================
# PREDICT FROM SYMPTOMS
# ============================================================

def predict_from_symptoms(
    detected_symptoms
):

    if not detected_symptoms:

        return None

    # Create feature vector

    feature_vector = create_feature_vector(
        detected_symptoms
    )

    # Scale

    scaled_features = scaler.transform(
        feature_vector
    )

    # KNN prediction

    prediction = knn_model.predict(
        scaled_features
    )

    # Convert encoded value to disease

    disease = encoder.inverse_transform(
        prediction
    )[0]

    return str(
        disease
    )


# ============================================================
# DISEASE PREDICTION
# ============================================================

def predict_disease(text):

    detected_symptoms = extract_symptoms(
        text
    )

    if not detected_symptoms:

        return {

            "success": False,

            "message":
                "No recognizable symptoms were detected. "
                "Try describing your symptoms in simple language.",

            "symptoms": [],

            "disease": None

        }

    disease = predict_from_symptoms(
        detected_symptoms
    )

    return {

        "success": True,

        "message":
            "Symptoms analyzed successfully.",

        "symptoms":
            detected_symptoms,

        "disease":
            disease

    }


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    if "." not in filename:

        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file_path):

    text = ""

    try:

        reader = PdfReader(
            file_path
        )

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += (
                    page_text
                    + "\n"
                )

    except Exception as e:

        print(
            "❌ PDF extraction error:",
            e
        )

        return ""

    return text.strip()


# ============================================================
# IMAGE OCR
# ============================================================

def extract_image_text(file_path):

    try:

        print("\n===================================")
        print("          OCR STARTED")
        print("===================================")

        print(
            "Image:",
            file_path
        )

        # Check Tesseract

        if not os.path.exists(
            TESSERACT_PATH
        ):

            print(
                "❌ Tesseract executable not found!"
            )

            return ""

        # Open image

        image = Image.open(
            file_path
        )

        print(
            "✅ Image loaded"
        )

        print(
            "Original size:",
            image.size
        )

        # RGB

        image = image.convert(
            "RGB"
        )

        # Upscale

        image = image.resize(
            (
                image.width * 3,
                image.height * 3
            ),
            Image.Resampling.LANCZOS
        )

        print(
            "Upscaled size:",
            image.size
        )

        # Grayscale

        gray = ImageOps.grayscale(
            image
        )

        # Contrast

        gray = ImageEnhance.Contrast(
            gray
        ).enhance(
            2.5
        )

        # Sharpness

        gray = ImageEnhance.Sharpness(
            gray
        ).enhance(
            2
        )

        # ----------------------------------------------------
        # OCR ATTEMPT 1
        # ----------------------------------------------------

        print(
            "OCR attempt 1..."
        )

        text1 = pytesseract.image_to_string(
            gray,
            config="--psm 6"
        )

        print(
            "Characters:",
            len(text1)
        )

        # ----------------------------------------------------
        # OCR ATTEMPT 2
        # ----------------------------------------------------

        print(
            "OCR attempt 2..."
        )

        text2 = pytesseract.image_to_string(
            gray,
            config="--psm 11"
        )

        print(
            "Characters:",
            len(text2)
        )

        # Choose better OCR result

        if len(
            text2.strip()
        ) > len(
            text1.strip()
        ):

            text = text2

        else:

            text = text1

        text = text.strip()

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        print("-----------------------------------")

        if text:

            print(
                "✅ OCR SUCCESS"
            )

            print(
                "Extracted characters:",
                len(text)
            )

            print(
                "\nExtracted text:"
            )

            print(
                text[:3000]
            )

        else:

            print(
                "❌ OCR returned EMPTY TEXT"
            )

        print(
            "===================================\n"
        )

        return text

    except pytesseract.TesseractNotFoundError:

        print(
            "❌ Tesseract executable not found."
        )

        return ""

    except Exception as e:

        print(
            "❌ OCR ERROR:"
        )

        print(
            type(e).__name__
        )

        print(
            e
        )

        return ""


# ============================================================
# REPORT TEXT EXTRACTION
# ============================================================

def extract_report_text(file_path):

    extension = file_path.rsplit(
        ".",
        1
    )[1].lower()

    print(
        "Report type:",
        extension
    )

    # PDF

    if extension == "pdf":

        return extract_pdf_text(
            file_path
        )

    # IMAGE

    if extension in {
        "jpg",
        "jpeg",
        "png"
    }:

        return extract_image_text(
            file_path
        )

    return ""


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# SYMPTOM PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({

                "success": False,

                "message":
                    "No data received."

            }), 400

        symptoms_text = data.get(
            "symptoms",
            ""
        )

        if not isinstance(
            symptoms_text,
            str
        ):

            symptoms_text = str(
                symptoms_text
            )

        symptoms_text = symptoms_text.strip()

        if not symptoms_text:

            return jsonify({

                "success": False,

                "message":
                    "Please enter your symptoms."

            }), 400

        result = predict_disease(
            symptoms_text
        )

        return jsonify(
            result
        )

    except Exception as e:

        print(
            "❌ Prediction error:",
            e
        )

        return jsonify({

            "success": False,

            "message":
                "Prediction failed.",

            "error":
                str(e)

        }), 500


# ============================================================
# MEDICAL REPORT ANALYSIS API
# ============================================================

@app.route(
    "/analyze-report",
    methods=["POST"]
)
def analyze_report():

    file_path = None

    try:

        # ----------------------------------------------------
        # CHECK FILE
        # ----------------------------------------------------

        if "report" not in request.files:

            return jsonify({

                "success": False,

                "message":
                    "No medical report uploaded."

            }), 400

        file = request.files[
            "report"
        ]

        if not file.filename:

            return jsonify({

                "success": False,

                "message":
                    "Please select a medical report."

            }), 400

        if not allowed_file(
            file.filename
        ):

            return jsonify({

                "success": False,

                "message":
                    "Only PDF, JPG, JPEG and PNG "
                    "files are supported."

            }), 400

        # ----------------------------------------------------
        # SAVE TEMPORARY FILE
        # ----------------------------------------------------

        filename = secure_filename(
            file.filename
        )

        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(
            file_path
        )

        print("\n===================================")
        print("       MEDICAL REPORT ANALYSIS")
        print("===================================")

        print(
            "File:",
            filename
        )

        # ----------------------------------------------------
        # EXTRACT TEXT
        # ----------------------------------------------------

        report_text = extract_report_text(
            file_path
        )

        if not report_text:

            return jsonify({

                "success": False,

                "message":
                    "OCR could not extract readable text "
                    "from this report. Please upload a "
                    "clear medical report image.",

                "symptoms": [],

                "disease": None

            }), 400

        # ----------------------------------------------------
        # LIMIT TEXT
        # ----------------------------------------------------

        report_text = report_text[
            :15000
        ]

        # ----------------------------------------------------
        # DETECT SYMPTOMS
        # ----------------------------------------------------

        detected_symptoms = extract_symptoms(
            report_text
        )

        # ----------------------------------------------------
        # PREDICT DISEASE
        # ----------------------------------------------------

        disease = None

        if detected_symptoms:

            disease = predict_from_symptoms(
                detected_symptoms
            )

        # ----------------------------------------------------
        # ANALYSIS MESSAGE
        # ----------------------------------------------------

        if disease:

            analysis_message = (

                f"AI detected "
                f"{len(detected_symptoms)} "
                f"recognizable symptom(s). "
                f"Possible condition: "
                f"{disease}."

            )

        else:

            analysis_message = (

                "The report was successfully read, "
                "but no recognizable symptoms from "
                "the trained dataset were detected."

            )

        # ----------------------------------------------------
        # TERMINAL OUTPUT
        # ----------------------------------------------------

        print(
            "Detected symptoms:",
            detected_symptoms
        )

        print(
            "Predicted disease:",
            disease
        )

        print(
            "===================================\n"
        )

        # ----------------------------------------------------
        # JSON RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "message":
                analysis_message,

            "disease":
                disease,

            "symptoms":
                detected_symptoms,

            "report_text":
                report_text,

            "text_length":
                len(report_text)

        })

    except Exception as e:

        print(
            "❌ REPORT ANALYSIS ERROR:"
        )

        print(
            type(e).__name__
        )

        print(
            e
        )

        return jsonify({

            "success": False,

            "message":
                "Medical report analysis failed.",

            "error":
                str(e)

        }), 500

    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY FILE
        # ----------------------------------------------------

        if (
            file_path
            and os.path.exists(
                file_path
            )
        ):

            try:

                os.remove(
                    file_path
                )

            except Exception:

                pass


# ============================================================
# FILE TOO LARGE ERROR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "success": False,

        "message":
            "File is too large. "
            "Maximum size is 10 MB."

    }), 413


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print("      AI HEALTHCARE ASSISTANT")
    print("===================================")

    print(
        "Symptoms:",
        len(symptom_columns)
    )

    print(
        "Diseases:",
        len(encoder.classes_)
    )

    print(
        "PDF Analysis: ENABLED"
    )

    print(
        "Image OCR: ENABLED"
    )

    print(
        "Lightweight KNN Model: ENABLED"
    )

    print(
        "Model: knn_model_light.pkl"
    )

    print(
        "Server: http://127.0.0.1:5000"
    )

    print(
        "===================================\n"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
