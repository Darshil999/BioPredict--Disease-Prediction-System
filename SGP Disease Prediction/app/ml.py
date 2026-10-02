"""Model loading, preprocessing and prediction insights for the three predictors.

Each predictor declares its model inputs in the exact order the model was trained
on. The forms in ``forms.py`` only collect and validate values; this module turns
the cleaned values into a feature vector and asks the model for a prediction.
"""

import logging
import pickle
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Feature:
    field: str  # form field name
    label: str  # human-readable label shown in the input summary
    model_name: str = ""  # column name stored in the model (if it stores one)
    unit: str = ""


@dataclass(frozen=True)
class Predictor:
    key: str
    name: str
    icon: str
    model_file: str
    features: tuple
    model_type: str
    description: str
    # Model output label shown as "Elevated predicted risk"; the other class is
    # shown as "Lower predicted risk".
    elevated_label: int = 1
    # True when the original preprocessing artifact is missing and the app uses
    # parameters reconstructed from the model itself (see LUNG_SCALER_*).
    reconstructed_preprocessing: bool = False


# Label semantics for the heart model cannot be verified from the pickle (it
# stores no training labels or metadata). The app keeps the original project's
# mapping, label 1 = elevated, which is the documented convention of the common
# heart.csv versions of the UCI Cleveland data. Note: on the features whose
# coding is unambiguous across dataset versions (sex, exang, oldpeak, thalach),
# the coefficients associate label 1 with the clinically lower-risk values,
# consistent with the known target inversion in some heart.csv copies. Confirm
# against the training data before relying on the mapping.
HEART = Predictor(
    key="heart",
    name="Heart Disease",
    icon="images/heart-icon.png",
    model_file="your_model.pkl",
    model_type="Logistic Regression",
    description="Screens 13 clinical indicators such as blood pressure, cholesterol and exercise test results.",
    features=(
        Feature("age", "Age", "age", "years"),
        Feature("sex", "Sex", "sex"),
        Feature("cp", "Chest pain type", "cp"),
        Feature("trestbps", "Resting blood pressure", "trestbps", "mm Hg"),
        Feature("chol", "Serum cholesterol", "chol", "mg/dL"),
        Feature("fbs", "Fasting blood sugar > 120 mg/dL", "fbs"),
        Feature("restecg", "Resting ECG result", "restecg"),
        Feature("thalach", "Max heart rate achieved", "thalach", "bpm"),
        Feature("exang", "Exercise-induced angina", "exang"),
        Feature("oldpeak", "ST depression (oldpeak)", "oldpeak"),
        Feature("slope", "Slope of peak exercise ST segment", "slope"),
        Feature("ca", "Major vessels coloured by fluoroscopy", "ca"),
        Feature("thal", "Thalassemia (thal)", "thal"),
    ),
)

DIABETES = Predictor(
    key="diabetes",
    name="Diabetes",
    icon="images/diabetes-icon.png",
    model_file="diabetes_model.pkl",
    model_type="Support Vector Classifier (linear kernel)",
    description="Uses 8 measurements including glucose, BMI, insulin and diabetes pedigree function.",
    features=(
        Feature("pregnancies", "Pregnancies", "Pregnancies"),
        Feature("glucose", "Glucose", "Glucose", "mg/dL"),
        Feature("blood_pressure", "Diastolic blood pressure", "BloodPressure", "mm Hg"),
        Feature("skin_thickness", "Triceps skin-fold thickness", "SkinThickness", "mm"),
        Feature("insulin", "Insulin", "Insulin", "µU/mL"),
        Feature("bmi", "BMI", "BMI", "kg/m²"),
        Feature("dpf", "Diabetes pedigree function", "DiabetesPedigreeFunction"),
        Feature("age", "Age", "Age", "years"),
    ),
)

# The lung cancer model was saved without column names. The order below is the
# column order of the survey lung cancer dataset and of the original view code.
LUNG_CANCER = Predictor(
    key="lung_cancer",
    name="Lung Cancer",
    icon="images/lung-icon.png",
    model_file="lung_cancer.pkl",
    model_type="Random Forest",
    reconstructed_preprocessing=True,
    description="Combines age, gender and 13 yes/no lifestyle and symptom indicators.",
    features=(
        Feature("gender", "Gender"),
        Feature("age", "Age", unit="years"),
        Feature("smoking", "Smoking"),
        Feature("yellow_fingers", "Yellow fingers"),
        Feature("anxiety", "Anxiety"),
        Feature("peer_pressure", "Peer pressure"),
        Feature("chronic_disease", "Chronic disease"),
        Feature("fatigue", "Fatigue"),
        Feature("allergy", "Allergy"),
        Feature("wheezing", "Wheezing"),
        Feature("alcohol_consuming", "Alcohol consumption"),
        Feature("coughing", "Coughing"),
        Feature("shortness_of_breath", "Shortness of breath"),
        Feature("swallowing_difficulty", "Swallowing difficulty"),
        Feature("chest_pain", "Chest pain"),
    ),
)

PREDICTORS = {p.key: p for p in (HEART, DIABETES, LUNG_CANCER)}

# The lung cancer Random Forest was trained on StandardScaler-transformed inputs,
# but the scaler itself was not saved with the model. Its parameters were
# recovered from the model's split thresholds (see tests.LungScalerTests):
# - each binary feature splits once, at (0.5 - p) / sqrt(p(1 - p)), which gives
#   its training mean p exactly (all are multiples of 1/220, i.e. 220 rows);
# - all age splits fall on a half-year grid with spacing 0.5 / std, which gives
#   the age std exactly and the mean only up to a whole-year offset
#   (61.25 / 62.25 / 63.25 all fit; 62.25 is used). Across every possible input,
#   the alternatives change the predicted class for 0.25% of inputs.
# Not recoverable from the model and taken from the original project code:
# the column order below index 1, Gender coded Male=1, and Yes coded above No.
# This is legacy preprocessing, not the original scaler artifact; re-exporting
# the model as Pipeline(StandardScaler, RandomForest) would remove it.
_LUNG_BINARY_MEANS = np.array([108, 116, 127, 108, 110, 114, 146, 124, 123, 125, 127, 137, 101, 124]) / 220
LUNG_SCALER_MEAN = np.insert(_LUNG_BINARY_MEANS, 1, 62.25)
LUNG_SCALER_SCALE = np.insert(np.sqrt(_LUNG_BINARY_MEANS * (1 - _LUNG_BINARY_MEANS)), 1, 8.0355035)


@dataclass
class PredictionResult:
    predictor: Predictor
    elevated: bool
    confidence: float | None  # predicted-class probability in [0, 1], if supported
    top_features: list  # [(label, share_of_total_importance)], may be empty


@lru_cache(maxsize=None)
def load_model(predictor_key):
    """Load a model once per process and check it matches the declared inputs."""
    predictor = PREDICTORS[predictor_key]
    with open(MODEL_DIR / predictor.model_file, "rb") as fh:
        model = pickle.load(fh)

    expected = [f.model_name for f in predictor.features]
    stored = getattr(model, "feature_names_in_", None)
    if stored is not None and list(stored) != expected:
        raise ValueError(f"{predictor.model_file} expects {list(stored)}, app sends {expected}")
    if model.n_features_in_ != len(predictor.features):
        raise ValueError(f"{predictor.model_file} expects {model.n_features_in_} features")
    return model


def build_feature_vector(predictor, cleaned_data):
    """Return the 1 x n_features array the model expects, in training order."""
    row = np.array([[float(cleaned_data[f.field]) for f in predictor.features]])
    if predictor is LUNG_CANCER:
        row = (row - LUNG_SCALER_MEAN) / LUNG_SCALER_SCALE
    return row


def predict(predictor, cleaned_data):
    model = load_model(predictor.key)
    X = build_feature_vector(predictor, cleaned_data)

    with warnings.catch_warnings():
        # Models fitted on DataFrames warn when given plain arrays; the column
        # order is verified in load_model instead.
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        label = int(model.predict(X)[0])
        confidence = None
        # SVC exposes predict_proba only when trained with probability=True.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(proba[list(model.classes_).index(label)])

    return PredictionResult(
        predictor=predictor,
        elevated=label == predictor.elevated_label,
        confidence=confidence,
        top_features=top_global_features(predictor),
    )


def top_global_features(predictor, n=3):
    """Top-n global feature importances for tree models.

    Linear model coefficients are not used: the heart and diabetes models were
    trained on unscaled inputs, so coefficient sizes are not comparable across
    features measured in different units.
    """
    importances = getattr(load_model(predictor.key), "feature_importances_", None)
    if importances is None:
        return []
    order = np.argsort(importances)[::-1][:n]
    return [(predictor.features[i].label, float(importances[i])) for i in order]
