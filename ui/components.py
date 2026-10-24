"""Small functions that return HTML for the app. No machine learning here.

Every string is built without leading spaces or blank lines, because
Streamlit's markdown would treat indented lines as a code block.
"""
from html import escape

from ui.labels import field_label

ARC_LENGTH = 251.33  # length of the half-circle gauge path


def hero_html(metrics):
    """Top banner with a few real facts about the trained model."""
    chips = []
    if metrics:
        chips.append(f"Model: {escape(str(metrics.get('selected_model', 'n/a')))}")
        if "test" in metrics:
            chips.append(f"Test ROC-AUC: {metrics['test']['roc_auc']:.2f}")
        if "dataset" in metrics:
            chips.append(f"{metrics['dataset']['rows']} loans in the dataset")
    chip_html = "".join(f'<span class="chip">{chip}</span>' for chip in chips)
    return (
        '<div class="hero fade-in">'
        '<div class="hero-eyebrow">CodeAlpha ML Internship · Task 1</div>'
        "<h1>Credit Risk Assessment</h1>"
        "<p>Enter a loan request and applicant profile to see a model-based risk score. "
        "Built to learn how credit scoring models work.</p>"
        f'<div class="chips">{chip_html}</div>'
        "</div>"
    )


def empty_state_html():
    return (
        '<div class="card empty fade-in delay-1">'
        '<div class="empty-icon">📊</div>'
        "<h3>No score yet</h3>"
        "<p>Fill in the form and press <b>Assess risk</b>. "
        "Your result will appear here.</p>"
        "</div>"
    )


def gauge_html(score, band):
    """Half-circle gauge that sweeps from empty to the score."""
    percent = round(score * 100)
    offset = ARC_LENGTH * (1 - score)
    tone = band["tone"]
    arc = "M 20 100 A 80 80 0 0 1 180 100"
    return (
        '<div class="card result fade-in">'
        '<div class="eyebrow">Estimated risk score</div>'
        f'<svg class="gauge" viewBox="0 0 200 120" role="img" aria-label="Risk score {percent} percent">'
        f'<path class="gauge-track" d="{arc}"/>'
        f'<path class="gauge-fill tone-{tone}" d="{arc}" style="stroke-dashoffset:{offset:.2f}"/>'
        f'<text x="100" y="94" text-anchor="middle" class="gauge-number">{percent}%</text>'
        "</svg>"
        f'<div class="band tone-{tone}">{escape(band["label"])}</div>'
        f'<p class="muted">{escape(band["message"])}</p>'
        "</div>"
    )


def factor_bars_html(features):
    """Bars showing which inputs matter most across all applicants."""
    if not features:
        return ""
    top = max(item["importance"] for item in features)
    rows = "".join(
        '<div class="factor">'
        f'<div class="factor-name">{escape(field_label(item["feature"]))}</div>'
        f'<div class="bar"><span style="--w:{item["importance"] / top * 100:.0f}%"></span></div>'
        "</div>"
        for item in features
    )
    return (
        '<div class="card factors fade-in delay-2">'
        '<div class="eyebrow">What the model pays attention to</div>'
        '<p class="muted small">The overall pattern across all applicants. '
        "It does not explain this single result.</p>"
        f"{rows}"
        "</div>"
    )


def disclaimer_html():
    return (
        '<div class="disclaimer fade-in delay-3">'
        "<b>Educational demo.</b> This score comes from a small public dataset "
        "(1,000 loans from one German bank, collected decades ago). "
        "It must not be used to approve or reject real loans. "
        "Real credit decisions need legal compliance, fairness checks and human review."
        "</div>"
    )


def footer_html():
    return (
        '<div class="footer">CodeAlpha Machine Learning Internship · '
        "Task 1: Credit Scoring Model · Educational project</div>"
    )
