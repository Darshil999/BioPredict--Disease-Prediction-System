# BioPredict

BioPredict is a Django web application that serves three pre-trained scikit-learn models for
health risk screening: heart disease, diabetes and lung cancer. Users fill in a validated form,
the inputs are passed to the model in the exact order it was trained on, and the app returns a
**Prediction Insights** page with the predicted risk class, the model's confidence (where the model
supports it), the features the model relies on most (where that can be reported reliably) and a
readable summary of the submitted inputs.

> BioPredict is an educational project. Its output is not a medical diagnosis.

## Features

- **Prediction Insights** – predicted class shown as *Elevated* / *Lower predicted risk*, with
  model confidence from `predict_proba()` when the model supports it. Nothing is estimated or
  invented when it doesn't.
- **Lightweight model transparency** – top global feature importances for the tree-based model.
- **Server-side validation** – required fields, numeric checks, allowed categorical choices and
  inline error messages. Missing values are rejected instead of being silently treated as `0`.
- **Model-input safety checks** – on load, each model's stored column names are compared against
  the order the app sends; mismatches fail loudly.
- **Models cached per process** – each `.pkl` is loaded once, not on every request.
- Responsive Bootstrap 5 UI with a persistent dark mode.
- Test suite covering feature order, predictions, validation and page rendering.

## Supported predictions

| Predictor     | Model                                  | Inputs | Confidence | Feature importance |
|---------------|----------------------------------------|--------|------------|--------------------|
| Heart Disease | Logistic Regression                    | 13     | Yes        | No (see below)     |
| Diabetes      | Support Vector Classifier, linear kernel | 8    | No         | No (see below)     |
| Lung Cancer   | Random Forest                          | 15     | Yes        | Yes                |

- The diabetes SVC was trained with `probability=False`, so it has no `predict_proba()`. The app
  shows the class label only.
- The heart and diabetes models are linear models trained on unscaled inputs. Their coefficients
  are not comparable across features in different units, so they are not presented as importances.
- The lung cancer Random Forest reports impurity-based global importances. These describe the model
  overall, not an individual prediction.

## Tech stack

Python · Django 5.1 · scikit-learn 1.5.2 · NumPy · Bootstrap 5.3 · Bootstrap Icons

## Architecture

```
Browser form ──POST──▶ views._predictor_view
                         │
                         ├─ forms.py   validate & clean inputs (types, ranges, choices)
                         ├─ ml.py      build feature vector in training order
                         │             (+ standard scaling for the lung model)
                         │             cached model ─▶ predict / predict_proba / importances
                         └─ result.html  Prediction Insights page
```

- `app/ml.py` – one declarative spec per predictor (model file, inputs in training order, units),
  cached model loading, explicit preprocessing, and insight extraction.
- `app/forms.py` – one Django form per predictor with validation and field grouping.
- `app/views.py` – a single shared GET/POST view used by all three predictors.

## Local setup

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd BioPredict--Disease-Prediction-System
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Running the server

Debug mode is opt-in through environment variables. For local development:

```powershell
# Windows PowerShell
cd "SGP Disease Prediction"
$env:DJANGO_DEBUG = "True"
python manage.py runserver
```

```bash
# macOS/Linux
cd "SGP Disease Prediction"
DJANGO_DEBUG=True python manage.py runserver
```

Open http://127.0.0.1:8000/.

Run the tests with `python manage.py test app` (same environment variable).

In production, set `DJANGO_SECRET_KEY`, keep `DJANGO_DEBUG` unset, and set `DJANGO_ALLOWED_HOSTS`
(comma-separated).

## Project structure

```
SGP Disease Prediction/
├── manage.py
├── model/                 # Django project settings & root URLs
├── app/
│   ├── ml.py              # predictor specs, model loading, preprocessing, insights
│   ├── forms.py           # validated input forms
│   ├── views.py           # shared predictor view
│   ├── urls.py
│   ├── tests.py
│   ├── your_model.pkl     # heart disease model
│   ├── diabetes_model.pkl
│   ├── lung_cancer.pkl
│   └── static/            # css/, js/ (dark mode), images/
└── templates/             # base, home, predictor, result
```

## Known model caveats

**Heart disease – label semantics not verified.** The pickle stores no training labels or dataset
metadata. The app keeps the original project's mapping (class `1` = elevated predicted risk), the
documented convention of common `heart.csv` versions of the UCI Cleveland data. However, on the
features whose coding is unambiguous across dataset versions (sex, exercise-induced angina,
oldpeak, max heart rate), the model's coefficients associate class `1` with the clinically
lower-risk values. That is consistent with the target inversion reported in some `heart.csv`
copies. Without the training data this cannot be confirmed either way. The model also stopped at
lbfgs's 100-iteration limit (`n_iter_ == max_iter`).

**Lung cancer – reconstructed legacy preprocessing.** The Random Forest was trained on
standardized (`StandardScaler`) inputs, but the original scaler artifact was not saved and no
training data or notebook is available. The app uses scaling parameters reconstructed from the
model's own split thresholds (documented in `app/ml.py`):

- binary-feature means and the age standard deviation reproduce every split threshold in the model
  (`LungScalerTests`);
- the age mean is known only up to a whole-year offset, which changes the predicted class for about
  0.25% of possible inputs;
- column order, `Male = 1` and `Yes > No` coding come from the original project code, not the model.

Predictions for this model should therefore be treated as approximate. The proper fix is to
re-export the model from its training code as a single `Pipeline(StandardScaler, RandomForest)`.

## ML disclaimer

These models were trained on small public datasets and are for demonstration only. Predictions
and confidence values are model outputs, not clinically validated probabilities, and must not be
used for diagnosis or treatment decisions. Consult a qualified clinician about any health concern.

## Future improvements

- Re-export each model as a scikit-learn `Pipeline` (including preprocessing) with saved metadata.
- Add the training notebooks and documented evaluation results to the repository.
- Per-prediction explanations for the linear models using standardized coefficients.
