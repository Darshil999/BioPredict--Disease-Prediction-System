import warnings
from unittest import mock

import numpy as np
from django.test import SimpleTestCase
from django.urls import reverse

from .forms import DiabetesForm, HeartDiseaseForm, LungCancerForm
from .ml import (
    DIABETES, HEART, LUNG_CANCER, LUNG_SCALER_MEAN, LUNG_SCALER_SCALE,
    build_feature_vector, load_model, predict,
)

HEART_INPUT = {
    'age': '54', 'sex': '1', 'cp': '0', 'trestbps': '130', 'chol': '250', 'fbs': '0', 'restecg': '1',
    'thalach': '150', 'exang': '1', 'oldpeak': '1.4', 'slope': '1', 'ca': '1', 'thal': '2',
}
DIABETES_INPUT = {
    'pregnancies': '2', 'glucose': '148', 'blood_pressure': '72', 'skin_thickness': '35',
    'insulin': '0', 'bmi': '33.6', 'dpf': '0.627', 'age': '50',
}
LUNG_INPUT = {f.field: '1' for f in LUNG_CANCER.features} | {'age': '65', 'allergy': '0'}


def raw_model_output(predictor, row):
    model = load_model(predictor.key)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X = np.array([row], dtype=float)
        proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
        return int(model.predict(X)[0]), proba


class FeatureOrderTests(SimpleTestCase):
    def test_declared_order_matches_stored_model_columns(self):
        for predictor in (HEART, DIABETES):
            model = load_model(predictor.key)
            self.assertEqual([f.model_name for f in predictor.features], list(model.feature_names_in_))

    def test_lung_model_input_count(self):
        self.assertEqual(load_model(LUNG_CANCER.key).n_features_in_, len(LUNG_CANCER.features))

    def test_forms_cover_every_model_input(self):
        for form_class in (HeartDiseaseForm, DiabetesForm, LungCancerForm):
            features = [f.field for f in form_class.predictor.features]
            grouped = [name for _, names in form_class.fieldsets for name in names]
            self.assertCountEqual(features, form_class.base_fields)
            self.assertCountEqual(features, grouped)

    def test_feature_vector_follows_training_order(self):
        form = HeartDiseaseForm(HEART_INPUT)
        self.assertTrue(form.is_valid(), form.errors)
        row = build_feature_vector(HEART, form.cleaned_data)[0]
        # Heart form field names equal the model's column names.
        expected = [float(HEART_INPUT[name]) for name in load_model(HEART.key).feature_names_in_]
        np.testing.assert_array_equal(row, expected)


class PredictionTests(SimpleTestCase):
    def test_heart_matches_direct_model_call(self):
        form = HeartDiseaseForm(HEART_INPUT)
        self.assertTrue(form.is_valid())
        result = predict(HEART, form.cleaned_data)
        label, proba = raw_model_output(HEART, [float(HEART_INPUT[f.field]) for f in HEART.features])
        self.assertEqual(result.elevated, label == 1)
        self.assertAlmostEqual(result.confidence, proba[label])
        self.assertEqual(result.top_features, [])

    def test_diabetes_has_no_confidence(self):
        form = DiabetesForm(DIABETES_INPUT)
        self.assertTrue(form.is_valid())
        result = predict(DIABETES, form.cleaned_data)
        label, proba = raw_model_output(DIABETES, [float(DIABETES_INPUT[f.field]) for f in DIABETES.features])
        self.assertIsNone(proba)
        self.assertIsNone(result.confidence)
        self.assertEqual(result.elevated, label == 1)

    def test_lung_uses_scaled_inputs_and_reports_importances(self):
        form = LungCancerForm(LUNG_INPUT)
        self.assertTrue(form.is_valid(), form.errors)
        result = predict(LUNG_CANCER, form.cleaned_data)
        raw = np.array([float(LUNG_INPUT[f.field]) for f in LUNG_CANCER.features])
        label, proba = raw_model_output(LUNG_CANCER, (raw - LUNG_SCALER_MEAN) / LUNG_SCALER_SCALE)
        self.assertEqual(result.elevated, label == 1)
        self.assertAlmostEqual(result.confidence, proba[label])
        self.assertEqual([name for name, _ in result.top_features], ['Age', 'Allergy', 'Yellow fingers'])


class ClassMappingTests(SimpleTestCase):
    """The app's chosen label -> display mapping and predicted-class confidence.

    These tests pin the application's mapping; they do not establish what the
    labels mean medically (see the notes in ml.py and the README).
    """

    def test_chosen_elevated_labels(self):
        self.assertEqual(HEART.elevated_label, 1)
        self.assertEqual(DIABETES.elevated_label, 1)
        self.assertEqual(LUNG_CANCER.elevated_label, 1)

    def test_both_classes_map_and_report_predicted_class_probability(self):
        heart_a = HEART_INPUT | {'sex': '0', 'thalach': '170', 'exang': '0', 'oldpeak': '0'}
        heart_b = HEART_INPUT | {'sex': '1', 'thalach': '110', 'exang': '1', 'oldpeak': '3'}
        lung_a = {f.field: '0' for f in LUNG_CANCER.features} | {'age': '65'}
        cases = [
            (HEART, HeartDiseaseForm, [heart_a, heart_b]),
            (LUNG_CANCER, LungCancerForm, [lung_a, LUNG_INPUT]),
        ]
        for predictor, form_class, inputs in cases:
            model = load_model(predictor.key)
            seen = set()
            for data in inputs:
                form = form_class(data)
                self.assertTrue(form.is_valid(), form.errors)
                X = build_feature_vector(predictor, form.cleaned_data)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    label = int(model.predict(X)[0])
                    proba = model.predict_proba(X)[0]
                result = predict(predictor, form.cleaned_data)
                self.assertEqual(result.elevated, label == predictor.elevated_label)
                self.assertAlmostEqual(result.confidence, proba[list(model.classes_).index(label)])
                self.assertGreaterEqual(result.confidence, 0.5)
                seen.add(label)
            self.assertEqual(seen, {0, 1}, predictor.key)

    def test_confidence_follows_classes_order_not_column_position(self):
        class FakeModel:
            classes_ = np.array([1, 0])  # deliberately reversed column order

            def __init__(self, label, proba):
                self.label, self.proba = label, proba

            def predict(self, X):
                return np.array([self.label])

            def predict_proba(self, X):
                return np.array([self.proba])

        form = HeartDiseaseForm(HEART_INPUT)
        self.assertTrue(form.is_valid())
        for label, proba, confidence in [(1, [0.8, 0.2], 0.8), (0, [0.3, 0.7], 0.7)]:
            with mock.patch('app.ml.load_model', return_value=FakeModel(label, proba)):
                result = predict(HEART, form.cleaned_data)
            self.assertEqual(result.elevated, label == 1)
            self.assertAlmostEqual(result.confidence, confidence)


class LungPreprocessingTests(SimpleTestCase):
    def test_only_lung_uses_reconstructed_preprocessing(self):
        self.assertTrue(LUNG_CANCER.reconstructed_preprocessing)
        self.assertFalse(HEART.reconstructed_preprocessing)
        self.assertFalse(DIABETES.reconstructed_preprocessing)

    def test_lung_inputs_are_standardized_in_training_order(self):
        data = {f.field: 0 for f in LUNG_CANCER.features} | {'gender': 1, 'age': 70}
        row = build_feature_vector(LUNG_CANCER, data)[0]
        raw = np.array([data[f.field] for f in LUNG_CANCER.features], dtype=float)
        np.testing.assert_allclose(row, (raw - LUNG_SCALER_MEAN) / LUNG_SCALER_SCALE)
        self.assertAlmostEqual(row[1], (70 - 62.25) / 8.0355035)
        # "Yes" (1) must land above "No" (0) for every binary feature.
        yes = build_feature_vector(LUNG_CANCER, {f.field: 1 for f in LUNG_CANCER.features})[0]
        self.assertTrue(all(yes[i] > row[i] for i in range(2, 15)))

    def test_heart_and_diabetes_inputs_are_passed_unscaled(self):
        for predictor, data in ((HEART, HEART_INPUT), (DIABETES, DIABETES_INPUT)):
            row = build_feature_vector(predictor, {k: float(v) for k, v in data.items()})[0]
            np.testing.assert_array_equal(row, [float(data[f.field]) for f in predictor.features])

    def test_reconstruction_note_shown_only_for_lung(self):
        note = 'input scaling is reconstructed'
        self.assertContains(self.client.post(reverse('lung_cancer'), LUNG_INPUT), note)
        self.assertNotContains(self.client.post(reverse('heart'), HEART_INPUT), note)


class LungScalerTests(SimpleTestCase):
    """The reconstructed scaler must reproduce the model's own split thresholds."""

    def thresholds(self, feature):
        model = load_model(LUNG_CANCER.key)
        return np.array([
            t for tree in model.estimators_
            for f, t in zip(tree.tree_.feature, tree.tree_.threshold) if f == feature
        ])

    def test_binary_features_split_exactly_between_no_and_yes(self):
        for i in [0] + list(range(2, 15)):
            midpoint = (0.5 - LUNG_SCALER_MEAN[i]) / LUNG_SCALER_SCALE[i]
            np.testing.assert_allclose(self.thresholds(i), midpoint, atol=1e-6)

    def test_age_splits_fall_on_half_year_grid(self):
        ages = self.thresholds(1) * LUNG_SCALER_SCALE[1] + LUNG_SCALER_MEAN[1]
        np.testing.assert_allclose(ages * 2, np.round(ages * 2), atol=1e-4)
        self.assertTrue(30 < ages.min() and ages.max() < 90)


class ViewTests(SimpleTestCase):
    def test_pages_render(self):
        for name in ('home', 'heart', 'diabetes', 'lung_cancer'):
            response = self.client.get(reverse(name))
            self.assertEqual(response.status_code, 200, name)

    def test_valid_submissions_show_insights(self):
        cases = [('heart', HEART_INPUT, True), ('diabetes', DIABETES_INPUT, False), ('lung_cancer', LUNG_INPUT, True)]
        for name, data, has_confidence in cases:
            response = self.client.post(reverse(name), data)
            self.assertTemplateUsed(response, 'result.html')
            self.assertContains(response, 'predicted risk')
            self.assertContains(response, 'Inputs submitted')
            self.assertEqual('for the predicted class' in response.content.decode(), has_confidence, name)

    def test_summary_uses_readable_labels(self):
        response = self.client.post(reverse('lung_cancer'), LUNG_INPUT)
        self.assertContains(response, '<dd class="mb-0">Male</dd>', html=True)
        self.assertContains(response, '<dd class="mb-0">65 years</dd>', html=True)

    def test_missing_fields_are_rejected_not_zeroed(self):
        data = dict(HEART_INPUT)
        del data['chol']
        response = self.client.post(reverse('heart'), data)
        self.assertTemplateUsed(response, 'predictor.html')
        self.assertContains(response, 'This field is required.')

    def test_invalid_values_show_inline_errors(self):
        cases = [
            ('heart', HEART_INPUT, 'age', 'abc', 'Enter a whole number.'),
            ('heart', HEART_INPUT, 'cp', '9', 'Select a valid choice.'),
            ('diabetes', DIABETES_INPUT, 'bmi', 'fat', 'Enter a number.'),
            ('diabetes', DIABETES_INPUT, 'glucose', '-5', 'greater than or equal to 1'),
            ('lung_cancer', LUNG_INPUT, 'smoking', 'Yes', 'Select a valid choice.'),
        ]
        for name, valid, field, value, message in cases:
            response = self.client.post(reverse(name), valid | {field: value})
            self.assertTemplateUsed(response, 'predictor.html')
            self.assertContains(response, message)
            self.assertContains(response, 'is-invalid' if field != 'smoking' else 'invalid-feedback')

    def test_empty_submission_does_not_crash(self):
        for name in ('heart', 'diabetes', 'lung_cancer'):
            response = self.client.post(reverse(name), {})
            self.assertEqual(response.status_code, 200)
            self.assertTemplateUsed(response, 'predictor.html')
