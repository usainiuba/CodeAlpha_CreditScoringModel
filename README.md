Credit Scoring Model
A small project that predicts whether a loan applicant is a good or bad credit risk, and shows the result in a simple web app.
I built it for Task 1 of the CodeAlpha Machine Learning Internship. The goal was to try a few classic models on real credit data, compare them honestly, and keep the best one.
---
What it does
Downloads the German Credit dataset (1,000 past loans, 20 details about each).
Cleans the data and adds a couple of useful features (like the monthly payment).
Trains and compares three models: Logistic Regression, Decision Tree and Random Forest.
Tests the best one on loans it has never seen.
Lets you type in an applicant's details in a web app and get a risk score.
---
The data
I used the German Credit dataset (UCI Statlog), loaded through OpenML as credit-g.
1,000 loans, 20 features, about 30% of them are "bad risk".
You don't need to download anything by hand. The first time you train, the script fetches it and saves a copy in data/. That folder is not stored in this repository, so you only need internet on the first run.
A note on fairness: I removed personal status (sex and marital status) and foreign worker before training. These shouldn't decide whether someone gets a loan.
---
How I tested it
800 loans were used for training and 200 were kept aside for the final test.
The three models were compared with 5-fold cross-validation on the 800 training loans.
The best model was then tested once on the 200 held-back loans.
"Bad risk" is the positive class. All models use class_weight="balanced" because bad loans are the minority.
Comparison (5-fold cross-validation)
The ± number is the standard deviation across the five folds.
Model ROC-AUC Precision Recall F1
Logistic Regression 0.761 ± 0.056 0.480 ± 0.038 0.654 ± 0.082 0.553 ± 0.054
Decision Tree 0.687 ± 0.059 0.448 ± 0.034 0.654 ± 0.148 0.527 ± 0.072
Random Forest 0.798 ± 0.041 0.551 ± 0.037 0.646 ± 0.071 0.594 ± 0.048
I picked Random Forest because it had the best ROC-AUC.
Final test (200 unseen loans, threshold 0.5)
Metric Value
ROC-AUC 0.802
Precision (bad risk) 0.583
Recall (bad risk) 0.700
F1 (bad risk) 0.636
Accuracy 0.760
Confusion matrix:
Predicted good Predicted bad
Actually good 110 30
Actually bad 18 42
In plain words: out of 60 bad loans the model caught 42 and missed 18. Out of 140 good loans it wrongly flagged 30 as risky.
Accuracy alone can fool you here. A model that always said "good" would already be right 70% of the time, which is why I also look at precision, recall and ROC-AUC.
What the model pays attention to
Measured with permutation importance (how much ROC-AUC drops when a feature is shuffled):
Checking account status (by far the biggest, 0.127)
Credit history (0.015)
Monthly payment (0.011)
Savings account (0.008)
Loan amount, log scale (0.008)
So the state of a person's checking account matters much more than anything else in this dataset.
---
Run it yourself
You need Python 3.10 or newer. These commands are for Windows PowerShell.
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1
# 2. install the libraries
pip install -r requirements.txt
# 3. train (downloads the data the first time)
python train_model.py
# 4. open the web app
python -m streamlit run app.py
Training saves models/credit_model.joblib, metrics.json, classification_report.txt and feature_options.json.
The saved model only works with the scikit-learn version it was trained with. If you change versions, just train again.
To run the tests:
python -m unittest discover -s tests -t .
---
Project structure
CodeAlpha_CreditScoringModel/
├── app.py            # the Streamlit web app
├── train_model.py    # compares the models, tests the best, saves results
├── predict.py        # loads the model and scores one applicant
├── src/
│   ├── data.py       # loading, cleaning, feature engineering, splitting
│   ├── models.py     # the three model pipelines
│   └── evaluate.py   # metrics and feature importance
├── ui/
│   ├── components.py # HTML pieces (header, gauge, bars)
│   ├── labels.py     # readable field names and form layout
│   └── style.css     # colours, layout, animations
├── tests/            # unit tests and an end-to-end check
├── models/           # the trained model and results
└── requirements.txt
---
Limitations
It is a small dataset (1,000 loans), so the numbers can shift a bit with a different split. The ± values show that.
The data is old and from one country, so this is not a model to use for real lending decisions.
At threshold 0.5 it misses about 3 in 10 bad loans. A bank that cares more about avoiding losses could lower the threshold, at the cost of rejecting more good applicants.
This is a learning project. The result is a risk estimate, not a final decision.
---
Built with
Python, scikit-learn, pandas, NumPy and Streamlit.
---
Made by Usaini Uba as part of the CodeAlpha Machine Learning Internship.