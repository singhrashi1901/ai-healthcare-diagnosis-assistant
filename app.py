import os
import re
import shutil
import tempfile
import warnings
import gc

# Keep native numerical libraries single-threaded.
# This is important on small Render instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import joblib
import numpy as np
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
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}


# ============================================================
# TESSERACT OCR
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
        print("OK: Tesseract OCR found")
        print("Path:", TESSERACT_PATH)
        print("Tesseract Version:", version)
    except Exception as e:
        print("WARNING: Tesseract found but could not start:", e)
else:
    print("WARNING: Tesseract OCR is not installed/found.")
    print("Image OCR will work locally when Tesseract is installed.")


# ============================================================
# ML MODEL
# ============================================================

knn_model = None
scaler = None
encoder = None
symptom_columns = []
model_loaded = False


def load_models():
    global knn_model, scaler, encoder, symptom_columns, model_loaded

    if model_loaded:
        return

    print("\n===================================")
    print("      LOADING AI HEALTHCARE MODEL")
    print("===================================")

    model_path = os.path.join(BASE_DIR, "knn_model_light.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")

    try:
        knn_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)

        # Never allow KNN to create extra worker threads.
        if hasattr(knn_model, "n_jobs"):
            knn_model.n_jobs = 1

        symptom_columns = list(
            getattr(scaler, "feature_names_in_", [])
        )

        if not symptom_columns:
            raise RuntimeError(
                "scaler.pkl does not contain feature_names_in_."
            )

        model_loaded = True

        fit_x = getattr(knn_model, "_fit_X", None)
        training_rows = fit_x.shape[0] if fit_x is not None else "unknown"

        print("OK: ML models loaded successfully")
        print("OK: Lightweight KNN model loaded")
        print("Model: knn_model_light.pkl")
        print("Symptoms:", len(symptom_columns))
        print("Diseases:", len(encoder.classes_))
        print("Training rows:", training_rows)

    except Exception as e:
        knn_model = None
        scaler = None
        encoder = None
        symptom_columns = []
        model_loaded = False

        print("\nMODEL LOADING ERROR")
        print(type(e).__name__, ":", e)
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
        "shortness of breath",
    ],
    "abdominal pain": [
        "stomach pain",
        "stomach ache",
        "belly pain",
        "pain in stomach",
        "abdominal pain",
    ],
    "sharp chest pain": [
        "chest pain",
        "pain in chest",
        "chest hurts",
        "pain around chest",
        "sharp chest pain",
    ],
    "cough": [
        "coughing",
        "cough",
    ],
    "dizziness": [
        "feeling dizzy",
        "dizzy",
        "lightheaded",
        "light headed",
        "dizziness",
    ],
    "sore throat": [
        "throat pain",
        "painful throat",
        "sore throat",
    ],
    "headache": [
        "head pain",
        "pain in head",
        "head ache",
        "headache",
    ],
    "vomiting": [
        "vomit",
        "vomiting",
        "throwing up",
    ],
    "fever": [
        "high temperature",
        "temperature",
        "fever",
    ],
    "nausea": [
        "feeling sick",
        "sick feeling",
        "nauseous",
        "nausea",
    ],
    "fatigue": [
        "tired",
        "very tired",
        "weakness",
        "feeling weak",
        "fatigue",
    ],
    "joint pain": [
        "pain in joints",
        "joint ache",
        "joints hurt",
        "joint pain",
    ],
    "back pain": [
        "pain in back",
        "back ache",
        "back hurts",
        "back pain",
    ],
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# EXTRACT SYMPTOMS
# ============================================================

def extract_symptoms(text):
    load_models()

    text = clean_text(text)
    detected = set()

    # Direct dataset symptom matching
    for symptom in symptom_columns:
        symptom_lower = clean_text(symptom)

        if symptom_lower and symptom_lower in text:
            detected.add(symptom)

    # Natural-language aliases
    for symptom, aliases in symptom_aliases.items():

        if symptom not in symptom_columns:
            continue

        for alias in aliases:
            if clean_text(alias) in text:
                detected.add(symptom)
                break

    return sorted(detected)


# ============================================================
# CREATE FEATURE VECTOR
# ============================================================

def create_feature_vector(detected_symptoms):
    load_models()

    vector = [
        1 if symptom in detected_symptoms else 0
        for symptom in symptom_columns
    ]

    return np.asarray(vector, dtype=np.float32).reshape(1, -1)


# ============================================================
# MEMORY-SAFE KNN PREDICTION
#
# IMPORTANT:
# sklearn KNN predict() can create a large temporary distance
# array. On Render Free, that can kill the worker with status 137.
#
# This implementation calculates distances in small chunks and
# keeps only the nearest neighbours. Prediction logic remains KNN.
# ============================================================

def memory_safe_knn_predict(model, X):
    train_X = getattr(model, "_fit_X", None)
    train_y = getattr(model, "_y", None)

    if train_X is None or train_y is None:
        raise RuntimeError("KNN model does not contain fitted training data.")

    query = np.asarray(X, dtype=np.float32).reshape(1, -1)

    n_samples = train_X.shape[0]
    k = min(int(getattr(model, "n_neighbors", 5)), n_samples)

    metric = getattr(model, "metric", "minkowski")
    p = getattr(model, "p", 2)
    weights = getattr(model, "weights", "uniform")

    # Keep this small so Render does not create a large temporary array.
    CHUNK_SIZE = 512

    best_distances = np.full(k, np.inf, dtype=np.float32)
    best_indices = np.full(k, -1, dtype=np.int64)

    for start in range(0, n_samples, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_samples)

        chunk = np.asarray(
            train_X[start:end],
            dtype=np.float32
        )

        diff = chunk - query

        if metric in ("euclidean", "l2"):
            distances = np.sqrt(
                np.sum(diff * diff, axis=1, dtype=np.float32)
            )
        elif metric in ("manhattan", "cityblock", "l1"):
            distances = np.sum(
                np.abs(diff),
                axis=1,
                dtype=np.float32
            )
        elif metric == "minkowski":
            if p == 1:
                distances = np.sum(
                    np.abs(diff),
                    axis=1,
                    dtype=np.float32
                )
            elif p == 2:
                distances = np.sqrt(
                    np.sum(diff * diff, axis=1, dtype=np.float32)
                )
            else:
                distances = np.power(
                    np.sum(
                        np.power(np.abs(diff), p),
                        axis=1,
                        dtype=np.float32
                    ),
                    1.0 / p
                )
        else:
            # The project's KNN is normally Euclidean/Minkowski.
            # Fall back safely to Euclidean for an unknown metric.
            distances = np.sqrt(
                np.sum(diff * diff, axis=1, dtype=np.float32)
            )

        local_k = min(k, len(distances))

        local_positions = np.argpartition(
            distances,
            local_k - 1
        )[:local_k]

        local_distances = distances[local_positions]
        local_indices = (
            local_positions.astype(np.int64) + start
        )

        combined_distances = np.concatenate(
            [best_distances, local_distances]
        )
        combined_indices = np.concatenate(
            [best_indices, local_indices]
        )

        keep = np.argpartition(
            combined_distances,
            k - 1
        )[:k]

        best_distances = combined_distances[keep]
        best_indices = combined_indices[keep]

        # Release chunk temporaries before next loop.
        del diff
        del distances
        del local_positions
        del local_distances
        del local_indices
        del combined_distances
        del combined_indices
        del keep

    # Remove invalid entries if any.
    valid = best_indices >= 0
    best_indices = best_indices[valid]
    best_distances = best_distances[valid]

    neighbour_labels = np.asarray(
        train_y[best_indices]
    )

    if weights in ("distance", "distance_weight"):

        zero_distance = best_distances == 0

        if np.any(zero_distance):
            zero_labels = neighbour_labels[zero_distance]

            values, counts = np.unique(
                zero_labels,
                return_counts=True
            )

            predicted_label = values[
                np.argmax(counts)
            ]

        else:
            safe_distances = np.maximum(
                best_distances,
                1e-12
            )

            neighbour_weights = 1.0 / safe_distances

            values = np.unique(neighbour_labels)

            scores = []

            for value in values:
                scores.append(
                    neighbour_weights[
                        neighbour_labels == value
                    ].sum()
                )

            predicted_label = values[
                int(np.argmax(scores))
            ]

    else:
        # Uniform KNN vote.
        values, counts = np.unique(
            neighbour_labels,
            return_counts=True
        )

        predicted_label = values[
            np.argmax(counts)
        ]

    return predicted_label


# ============================================================
# PREDICT FROM SYMPTOMS
# ============================================================

def predict_from_symptoms(detected_symptoms):
    load_models()

    if not detected_symptoms:
        return None

    feature_vector = create_feature_vector(
        detected_symptoms
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names"
        )

        scaled_features = scaler.transform(
            feature_vector
        )

    # Keep prediction completely single-threaded.
    if hasattr(knn_model, "n_jobs"):
        knn_model.n_jobs = 1

    predicted_label = memory_safe_knn_predict(
        knn_model,
        scaled_features
    )

    disease = encoder.inverse_transform(
        np.asarray([predicted_label])
    )[0]

    # Explicitly release request arrays.
    del feature_vector
    del scaled_features
    gc.collect()

    return str(disease)


# ============================================================
# DISEASE PREDICTION
# ============================================================

def predict_disease(text):
    detected_symptoms = extract_symptoms(text)

    if not detected_symptoms:
        return {
            "success": False,
            "message": (
                "No recognizable symptoms were detected. "
                "Try describing your symptoms in simple language."
            ),
            "symptoms": [],
            "disease": None,
        }

    disease = predict_from_symptoms(
        detected_symptoms
    )

    return {
        "success": True,
        "message": "Symptoms analyzed successfully.",
        "symptoms": detected_symptoms,
        "disease": disease,
    }


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):
    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file_path):
    text_parts = []

    try:
        reader = PdfReader(file_path)

        # Keep PDF processing bounded on small instances.
        max_pages = min(len(reader.pages), 30)

        for index in range(max_pages):

            page_text = reader.pages[index].extract_text()

            if page_text:
                text_parts.append(page_text)

    except Exception as e:
        print(
            "PDF extraction error:",
            type(e).__name__,
            e
        )
        return ""

    text = "\n".join(text_parts).strip()

    return text[:15000]


# ============================================================
# IMAGE OCR
# ============================================================

def extract_image_text(file_path):

    if not OCR_AVAILABLE:
        print(
            "OCR requested, but Tesseract is not available."
        )
        return ""

    try:
        print("\n===================================")
        print("          OCR STARTED")
        print("===================================")

        with Image.open(file_path) as original:

            image = original.convert("RGB")

            # Never allow huge images to create a Render memory spike.
            max_side = 1400

            largest_side = max(
                image.width,
                image.height
            )

            if largest_side > max_side:

                scale = max_side / largest_side

                new_size = (
                    max(1, int(image.width * scale)),
                    max(1, int(image.height * scale))
                )

                image = image.resize(
                    new_size,
                    Image.Resampling.LANCZOS
                )

            # Do NOT upscale small images.
            # Upscaling was one of the unnecessary memory costs.
            gray = ImageOps.grayscale(image)

            gray = ImageEnhance.Contrast(
                gray
            ).enhance(1.8)

            gray = ImageEnhance.Sharpness(
                gray
            ).enhance(1.3)

            # One OCR pass is enough for the deployed version.
            text = pytesseract.image_to_string(
                gray,
                config="--psm 6"
            ).strip()

            del gray
            del image
            gc.collect()

        if text:
            print("OCR SUCCESS")
            print("Extracted characters:", len(text))
        else:
            print("OCR returned EMPTY TEXT")

        return text[:15000]

    except pytesseract.TesseractNotFoundError:
        print("Tesseract executable not found.")
        return ""

    except Exception as e:
        print(
            "OCR ERROR:",
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
        return extract_pdf_text(file_path)

    if extension in {"jpg", "jpeg", "png"}:
        return extract_image_text(file_path)

    return ""


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "ocr_available": OCR_AVAILABLE,
        "service": "HealthAI",
    })


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# SYMPTOM PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(
            silent=True
        )

        if not data:
            return jsonify({
                "success": False,
                "message": "No data received.",
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
                "message": "Please enter your symptoms.",
            }), 400

        print("\nPREDICTION REQUEST")
        print("Symptoms:", symptoms_text[:500])

        result = predict_disease(
            symptoms_text
        )

        print(
            "Prediction completed:",
            result.get("disease")
        )

        return jsonify(result)

    except MemoryError:
        gc.collect()

        return jsonify({
            "success": False,
            "message": (
                "The prediction used too much memory. "
                "Please try fewer symptoms."
            ),
        }), 503

    except Exception as e:

        print(
            "PREDICTION ERROR:",
            type(e).__name__,
            e
        )

        gc.collect()

        return jsonify({
            "success": False,
            "message": "Prediction failed.",
            "error": str(e),
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
                "message": "No medical report uploaded.",
            }), 400

        file = request.files["report"]

        if not file.filename:
            return jsonify({
                "success": False,
                "message": "Please select a medical report.",
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "message": (
                    "Only PDF, JPG, JPEG and PNG "
                    "files are supported."
                ),
            }), 400

        filename = secure_filename(
            file.filename
        )

        suffix = os.path.splitext(
            filename
        )[1].lower()

        fd, file_path = tempfile.mkstemp(
            prefix="healthai_",
            suffix=suffix,
            dir=UPLOAD_FOLDER
        )

        os.close(fd)

        file.save(file_path)

        print("\n===================================")
        print("       MEDICAL REPORT ANALYSIS")
        print("===================================")
        print("File:", filename)

        report_text = extract_report_text(
            file_path
        )

        if not report_text:

            if suffix == ".pdf":

                message = (
                    "No readable text could be extracted "
                    "from this PDF. If it is a scanned PDF, "
                    "upload a clear image version or a PDF "
                    "containing selectable text."
                )

            elif not OCR_AVAILABLE:

                message = (
                    "Tesseract OCR is not installed on "
                    "this server. Image OCR works locally "
                    "when Tesseract is installed."
                )

            else:

                message = (
                    "OCR could not extract readable text. "
                    "Please upload a clear medical report image."
                )

            return jsonify({
                "success": False,
                "message": message,
                "symptoms": [],
                "disease": None,
            }), 400

        report_text = report_text[:15000]

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
                f"AI detected {len(detected_symptoms)} "
                f"recognizable symptom(s). "
                f"Possible condition: {disease}."
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

        gc.collect()

        return jsonify({
            "success": True,
            "message": analysis_message,
            "disease": disease,
            "symptoms": detected_symptoms,
            "report_text": report_text,
            "text_length": len(report_text),
        })

    except MemoryError:

        gc.collect()

        return jsonify({
            "success": False,
            "message": (
                "The report is too large for the available "
                "memory. Please upload a smaller report."
            ),
        }), 503

    except Exception as e:

        print(
            "REPORT ANALYSIS ERROR:",
            type(e).__name__,
            e
        )

        gc.collect()

        return jsonify({
            "success": False,
            "message": "Medical report analysis failed.",
            "error": str(e),
        }), 500

    finally:

        if (
            file_path
            and os.path.exists(file_path)
        ):
            try:
                os.remove(file_path)
            except Exception:
                pass

        gc.collect()


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "message": (
            "File is too large. "
            "Maximum size is 8 MB."
        ),
    }), 413


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print("      AI HEALTHCARE ASSISTANT")
    print("===================================")

    print("PDF Analysis: ENABLED")

    print(
        "Image OCR:",
        "ENABLED"
        if OCR_AVAILABLE
        else
        "AVAILABLE WHEN TESSERACT IS INSTALLED"
    )

    print(
        "Memory-safe KNN prediction: ENABLED"
    )

    print(
        "Model: knn_model_light.pkl"
    )

    print(
        "Server: http://127.0.0.1:5000"
    )

    print("===================================\n")

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