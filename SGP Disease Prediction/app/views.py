import logging

from django.shortcuts import render

from .forms import DiabetesForm, HeartDiseaseForm, LungCancerForm
from .ml import PREDICTORS, predict

logger = logging.getLogger(__name__)


def home(request):
    return render(request, 'home.html', {'predictors': PREDICTORS.values()})


def _predictor_view(request, form_class):
    """Show the predictor form on GET; validate, predict and show insights on POST."""
    predictor = form_class.predictor
    form = form_class(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        try:
            result = predict(predictor, form.cleaned_data)
        except Exception:
            logger.exception("Prediction failed for %s", predictor.key)
            form.add_error(None, "The model could not produce a prediction. Please try again later.")
        else:
            return render(request, 'result.html', {
                'predictor': predictor,
                'result': result,
                'summary': form.summary(),
            })

    return render(request, 'predictor.html', {'predictor': predictor, 'form': form})


def heart(request):
    return _predictor_view(request, HeartDiseaseForm)


def diabetes(request):
    return _predictor_view(request, DiabetesForm)


def lung_cancer(request):
    return _predictor_view(request, LungCancerForm)
