import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import re
import shutil
import tempfile
import warnings

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

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}


# ============================================================
# TESSERACT OCR
# Works on Windows and Linux/Render when Tesseract is installed
# ============================================================

def find_tesseract():
    env_path = os.environ.get("TESSERACT_CMD")

    if env_path and os.path.isfile(env_path):
        return env_path

    found = shutil.which("tesseract")

    if found:
        return found

    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]

    for path in windows_paths:
        if os.path.isfile(path):
            return path

    return None


TESSERACT_PATH = find_tesseract()
OCR_AVAILABLE = False

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    try:
        version = pytesseract.get_tesseract_version()

        OCR_AVAILABLE = True

        print("✅ Tesseract OCR found!")
        print("Path:", TESSERACT_PATH)
        print("Tesseract Version:", version)

    except Exception as e:
        print("⚠️ Tesseract was found but could not start:", e)

else:
    print("⚠️ Tesseract OCR not installed/found.")
    print(
        "OCR will work locally if Tesseract is installed, "
        "or on Render if Tesseract is installed."
    )


# ============================================================
# ML MODELS
# Lazy loading reduces startup memory.
# n_jobs=1 prevents unnecessary parallel memory usage on Render.
# ============================================================

knn_model = None
scaler = None
encoder = None
symptom_columns = []
model_loaded = False


def load_models():

    global knn_model
    global scaler
    global encoder
    global symptom_columns
    global model_loaded

    if model_loaded:
        return

    print("\n===================================")
    print("      LOADING AI HEALTHCARE MODEL")
    print("===================================")

    model_path = os.path.join(
        BASE_DIR,
        "knn_model_light.pkl"
    )

    scaler_path = os.path.join(
        BASE_DIR,
        "scaler.pkl"
    )

    encoder_path = os.path.join(
        BASE_DIR,
        "label_encoder.pkl"
    )

    try:

        # Do NOT use mmap_mode here.
        # knn_model_light.pkl was saved with compression,
        # and compressed joblib arrays cannot be memory-mapped.

        knn_model = joblib.load(
            model_path
        )

        scaler = joblib.load(
            scaler_path
        )

        encoder = joblib.load(
            encoder_path
        )

        # The original lightweight model was created with n_jobs=-1.
        # One job is much safer for Render's small memory limit.

        if hasattr(knn_model, "n_jobs"):
            knn_model.n_jobs = 1

        symptom_columns = list(
            getattr(
                scaler,
                "feature_names_in_",
                []
            )
        )

        if not symptom_columns:
            raise RuntimeError(
                "scaler.pkl does not contain feature_names_in_."
            )

        model_loaded = True

        print("✅ ML models loaded successfully!")
        print("✅ Lightweight KNN model loaded!")
        print("Model: knn_model_light.pkl")
        print("Symptoms:", len(symptom_columns))
        print("Diseases:", len(encoder.classes_))

    except Exception as e:

        knn_model = None
        scaler = None
        encoder = None
        symptom_columns = []
        model_loaded = False

        print("\n❌ MODEL LOADING ERROR")
        print("-----------------------------------")
        print(
            type(e).__name__,
            ":",
            e
        )
        print("-----------------------------------")

        raise


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

    load_models()

    text = clean_text(text)

    detected = set()

    # Dataset symptom matching

    for symptom in symptom_columns:

        symptom_lower = clean_text(
            symptom
        )

        if (
            symptom_lower
            and symptom_lower in text
        ):
            detected.add(
                symptom
            )

    # Alias matching

    for symptom, aliases in symptom_aliases.items():

        if symptom not in symptom_columns:
            continue

        for alias in aliases:

            if clean_text(alias) in text:

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

    load_models()

    vector = [

        1 if symptom in detected_symptoms
        else 0

        for symptom in symptom_columns
    ]

    # float32 uses less memory than float64

    return np.asarray(
        vector,
        dtype=np.float32
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

    load_models()

    if not detected_symptoms:
        return None

    feature_vector = create_feature_vector(
        detected_symptoms
    )

    # Your scaler was fitted using feature names.
    # The original project passed a NumPy array, which generated
    # a harmless sklearn warning. Suppress only that warning.

    with warnings.catch_warnings():

        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names"
        )

        scaled_features = scaler.transform(
            feature_vector
        )

    # Keep prediction single-threaded.

    if hasattr(knn_model, "n_jobs"):
        knn_model.n_jobs = 1

    prediction = knn_model.predict(
        scaled_features
    )

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

    text_parts = []

    try:

        reader = PdfReader(
            file_path
        )

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text_parts.append(
                    page_text
                )

    except Exception as e:

        print(
            "❌ PDF extraction error:",
            e
        )

        return ""

    return "\n".join(
        text_parts
    ).strip()


# ============================================================
# IMAGE OCR
# ============================================================

def extract_image_text(file_path):

    if not OCR_AVAILABLE:

        print(
            "⚠️ OCR requested, "
            "but Tesseract is not available."
        )

        return ""

    try:

        print("\n===================================")
        print("          OCR STARTED")
        print("===================================")

        print(
            "Image:",
            file_path
        )

        with Image.open(
            file_path
        ) as original:

            image = original.convert(
                "RGB"
            )

            print(
                "Original size:",
                image.size
            )

            # Prevent very large images from causing
            # a memory spike on Render.

            max_side = 1800

            scale = min(
                1.0,
                max_side / max(
                    image.size
                )
            )

            if scale < 1.0:

                new_size = (

                    max(
                        1,
                        int(
                            image.width * scale
                        )
                    ),

                    max(
                        1,
                        int(
                            image.height * scale
                        )
                    )
                )

                image = image.resize(
                    new_size,
                    Image.Resampling.LANCZOS
                )

            # Moderate upscale for small images only.

            if max(
                image.size
            ) < 1200:

                image = image.resize(

                    (
                        image.width * 2,
                        image.height * 2
                    ),

                    Image.Resampling.LANCZOS
                )

            gray = ImageOps.grayscale(
                image
            )

            gray = ImageEnhance.Contrast(
                gray
            ).enhance(
                2.0
            )

            gray = ImageEnhance.Sharpness(
                gray
            ).enhance(
                1.5
            )

            # OCR attempt 1

            print(
                "OCR attempt 1..."
            )

            text1 = pytesseract.image_to_string(
                gray,
                config="--psm 6"
            )

            # OCR attempt 2

            print(
                "OCR attempt 2..."
            )

            text2 = pytesseract.image_to_string(
                gray,
                config="--psm 11"
            )

            if len(
                text2.strip()
            ) > len(
                text1.strip()
            ):

                text = text2

            else:

                text = text1

            text = text.strip()

        if text:

            print(
                "✅ OCR SUCCESS"
            )

            print(
                "Extracted characters:",
                len(text)
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
            "❌ OCR ERROR:",
            type(e).__name__,
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

    if extension == "pdf":

        return extract_pdf_text(
            file_path
        )

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

        print(
            "🔎 Prediction request received"
        )

        print(
            "Symptoms text:",
            symptoms_text[:500]
        )

        result = predict_disease(
            symptoms_text
        )

        print(
            "✅ Prediction completed:",
            result.get("disease")
        )

        return jsonify(
            result
        )

    except Exception as e:

        print(
            "❌ Prediction error:",
            type(e).__name__,
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

        filename = secure_filename(
            file.filename
        )

        # Unique temporary file prevents
        # simultaneous requests from overwriting files.

        suffix = os.path.splitext(
            filename
        )[1].lower()

        fd, file_path = tempfile.mkstemp(

            prefix="healthai_",

            suffix=suffix,

            dir=UPLOAD_FOLDER
        )

        os.close(fd)

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

        report_text = extract_report_text(
            file_path
        )

        if not report_text:

            if suffix == ".pdf":

                message = (
                    "No readable text could be extracted "
                    "from this PDF. If it is a scanned PDF, "
                    "upload a clear image version or use a "
                    "PDF containing selectable text."
                )

            elif not OCR_AVAILABLE:

                message = (
                    "Tesseract OCR is not installed on "
                    "this server. Install Tesseract on "
                    "the deployment environment to "
                    "analyze image reports."
                )

            else:

                message = (
                    "OCR could not extract readable text "
                    "from this report. Please upload a "
                    "clear medical report image."
                )

            return jsonify({

                "success": False,

                "message":
                    message,

                "symptoms": [],

                "disease": None

            }), 400

        # Keep response reasonably small.

        report_text = report_text[
            :15000
        ]

        detected_symptoms = extract_symptoms(
            report_text
        )

        disease = None

        if detected_symptoms:

            disease = predict_from_symptoms(
                detected_symptoms
            )

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
            "❌ REPORT ANALYSIS ERROR:",
            type(e).__name__,
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
# RUN FLASK LOCALLY
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print("      AI HEALTHCARE ASSISTANT")
    print("===================================")

    print(
        "PDF Analysis: ENABLED"
    )

    print(
        "Image OCR:",
        "ENABLED"
        if OCR_AVAILABLE
        else
        "AVAILABLE WHEN TESSERACT IS INSTALLED"
    )

    print(
        "Lightweight KNN Model: ENABLED (lazy loaded)"
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

        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),

        debug=False,

        use_reloader=False
    )