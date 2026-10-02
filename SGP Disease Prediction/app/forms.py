from django import forms

from .ml import DIABETES, HEART, LUNG_CANCER

YES_NO = [(1, "Yes"), (0, "No")]


def coded(n, prefix="Type"):
    """Choices for categorical inputs whose dataset codes have no verified label."""
    return [(i, f"{prefix} {i}") for i in range(n)]


def integer(label, min_value=0, max_value=None, help_text=""):
    return forms.IntegerField(label=label, min_value=min_value, max_value=max_value, help_text=help_text)


def decimal(label, min_value=0, help_text=""):
    return forms.FloatField(label=label, min_value=min_value, help_text=help_text)


def select(label, choices, help_text=""):
    return forms.TypedChoiceField(
        label=label, choices=[("", "Select…")] + choices, coerce=int, help_text=help_text
    )


def toggle(label, choices=YES_NO, help_text=""):
    return forms.TypedChoiceField(
        label=label, choices=choices, coerce=int, widget=forms.RadioSelect, help_text=help_text
    )


def age():
    return integer("Age", min_value=1, max_value=120)


class PredictorForm(forms.Form):
    """Base form: groups fields for display and attaches units from the predictor spec."""

    predictor = None
    fieldsets = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            widget = field.widget
            if isinstance(widget, forms.RadioSelect):
                widget.attrs['class'] = 'btn-check'
            elif isinstance(widget, forms.Select):
                widget.attrs['class'] = 'form-select'
            else:
                widget.attrs['class'] = 'form-control'

    def grouped_fields(self):
        units = {f.field: f.unit for f in self.predictor.features}
        groups = []
        for title, names in self.fieldsets:
            for name in names:
                if name in self.errors and not isinstance(self.fields[name].widget, forms.RadioSelect):
                    self.fields[name].widget.attrs['class'] += ' is-invalid'
            groups.append((title, [(self[name], units[name]) for name in names]))
        return groups

    def summary(self):
        """Human-readable (label, value) pairs for the submitted inputs, in model order."""
        rows = []
        for feature in self.predictor.features:
            field = self.fields[feature.field]
            value = self.cleaned_data[feature.field]
            if isinstance(field, forms.TypedChoiceField):
                value = dict(field.choices)[value]
            elif feature.unit:
                value = f"{value:g} {feature.unit}"
            else:
                value = f"{value:g}"
            rows.append((feature.label, value))
        return rows


class HeartDiseaseForm(PredictorForm):
    predictor = HEART
    fieldsets = (
        ("Profile", ["age", "sex"]),
        ("Vitals & blood work", ["trestbps", "chol", "fbs"]),
        ("Symptoms & exercise test", ["cp", "exang", "thalach", "oldpeak", "slope"]),
        ("Diagnostic results", ["restecg", "ca", "thal"]),
    )

    age = age()
    sex = select("Sex", [(1, "Male"), (0, "Female")])
    cp = select("Chest pain type", coded(4), "Dataset category code (0–3).")
    trestbps = integer("Resting blood pressure", min_value=1)
    chol = integer("Serum cholesterol", min_value=1)
    fbs = toggle("Fasting blood sugar > 120 mg/dL")
    restecg = select("Resting ECG result", coded(3), "Dataset category code (0–2).")
    thalach = integer("Max heart rate achieved", min_value=1)
    exang = toggle("Exercise-induced angina")
    oldpeak = decimal("ST depression (oldpeak)", help_text="ST depression induced by exercise relative to rest.")
    slope = select("Slope of peak exercise ST segment", coded(3), "Dataset category code (0–2).")
    ca = select("Major vessels coloured by fluoroscopy", coded(5, "Count:"), "0–4, as encoded in the dataset.")
    thal = select("Thalassemia (thal)", coded(4), "Dataset category code (0–3).")


class DiabetesForm(PredictorForm):
    predictor = DIABETES
    fieldsets = (
        ("Profile", ["pregnancies", "age", "bmi", "dpf"]),
        ("Clinical measurements", ["glucose", "blood_pressure", "insulin", "skin_thickness"]),
    )

    pregnancies = integer("Pregnancies", help_text="Number of times pregnant.")
    glucose = integer("Glucose", min_value=1, help_text="Plasma glucose concentration.")
    blood_pressure = integer("Diastolic blood pressure", min_value=1)
    skin_thickness = integer("Triceps skin-fold thickness")
    insulin = integer("Insulin", help_text="2-hour serum insulin.")
    bmi = decimal("BMI", min_value=0.1, help_text="Body mass index.")
    dpf = decimal("Diabetes pedigree function", help_text="Score summarising family history of diabetes.")
    age = age()


class LungCancerForm(PredictorForm):
    predictor = LUNG_CANCER
    fieldsets = (
        ("Profile", ["gender", "age"]),
        ("Lifestyle & history", ["smoking", "alcohol_consuming", "peer_pressure", "chronic_disease"]),
        (
            "Symptoms",
            [
                "yellow_fingers", "anxiety", "fatigue", "allergy", "wheezing",
                "coughing", "shortness_of_breath", "swallowing_difficulty", "chest_pain",
            ],
        ),
    )

    gender = toggle("Gender", [(1, "Male"), (0, "Female")])
    age = age()
    smoking = toggle("Smoking")
    yellow_fingers = toggle("Yellow fingers")
    anxiety = toggle("Anxiety")
    peer_pressure = toggle("Peer pressure")
    chronic_disease = toggle("Chronic disease")
    fatigue = toggle("Fatigue")
    allergy = toggle("Allergy")
    wheezing = toggle("Wheezing")
    alcohol_consuming = toggle("Alcohol consumption")
    coughing = toggle("Coughing")
    shortness_of_breath = toggle("Shortness of breath")
    swallowing_difficulty = toggle("Swallowing difficulty")
    chest_pain = toggle("Chest pain")
