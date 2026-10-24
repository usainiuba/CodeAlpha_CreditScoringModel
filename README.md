CodeAlpha Credit Scoring Model
A machine learning model that estimates credit risk from loan and applicant details, with a Streamlit web app for trying it out.
Built for the CodeAlpha Machine Learning Internship, Task 1: Credit Scoring Model.
Educational project. The score is a model output learned from a small, old public dataset.
It is not a lending decision and must not be used to approve or reject real loans.
Features
Compares three algorithms: Logistic Regression, Decision Tree and Random Forest
Feature engineering: monthly payment, long-term loan flag, log of credit amount
Missing-value handling, scaling and encoding inside the model pipeline (no data leakage)
Model chosen with 5-fold cross-validation on training data; test set used once at the end
Metrics: ROC-AUC, precision, recall, F1 (for the "bad risk" class), plus confusion matrix
Overall feature influence using permutation importance
Web app with an animated risk gauge, clear inputs, and visible disclaimers
Dataset
German Credit (Statlog) data: 1,000 loans with 20 attributes, from the UCI Machine Learning Repository.
It is downloaded automatically through OpenML (credit-g) the first time you train, and cached in data/ (not stored in this repository).
See the UCI/OpenML pages for terms of use.
Fairness note: personal status (sex/marital status) and foreign-worker status are removed before training,
because they should not drive credit decisions.
Technology
Python, scikit-learn, pandas, NumPy, Streamlit.
Project structure
CodeAlpha_CreditScoringModel/
├── app.py            # Streamlit web app
├── train_model.py    # compares models, tests the best one, saves results
├── predict.py        # loads the model and scores one applicant
├── src/
│   ├── data.py       # loading, cleaning, feature engineering, splitting
│   ├── models.py     # the three pipelines
│   └── evaluate.py   # metrics and feature importance
├── ui/
│   ├── components.py # HTML pieces (hero, gauge, bars)
│   ├── labels.py     # readable field names and form layout
│   └── style.css     # colours, layout, animations
├── tests/            # unit tests (run offline on fake data)
├── models/           # trained model and results
├── .streamlit/config.toml
└── requirements.txt
Install (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Train
python train_model.py
Needs internet the first time. Saves models/credit_model.joblib, metrics.json, classification_report.txt and feature_options.json.
The saved model only works with the scikit-learn version it was trained with. If you change versions, train again.
Run the app
python -m streamlit run app.py
Score from the command line
python predict.py
Run the tests
python -m unittest discover -s tests -t .
The tests use fake data so they run without internet. Their numbers mean nothing.
Model evaluation results
These numbers come from one training run (random seed 42) and are saved in models/metrics.json.
The dataset has 1,000 loans, 30% of them bad risks. 800 were used for training and model choice, and 200 were kept aside for one final test.
Comparing the models
Five-fold cross-validation on the 800 training loans. "Bad risk" is the positive class. The ± value is the standard deviation across the five folds.
Model ROC-AUC Precision Recall F1
Logistic Regression 0.761 ± 0.056 0.480 ± 0.038 0.654 ± 0.082 0.553 ± 0.054
Decision Tree 0.687 ± 0.059 0.448 ± 0.034 0.654 ± 0.148 0.527 ± 0.072
Random Forest 0.798 ± 0.041 0.551 ± 0.037 0.646 ± 0.071 0.594 ± 0.048
Selected model: Random Forest, because it had the highest cross-validation ROC-AUC.
Final test
The selected model was tested once on the 200 held-out loans (decision threshold 0.5).
Metric Value
ROC-AUC 0.802
Precision (bad risk) 0.583
Recall (bad risk) 0.700
F1 (bad risk) 0.636
Accuracy 0.760
Confusion matrix on the test set:
Predicted good Predicted bad
Actually good 110 30
Actually bad 18 42
In plain words: of the 60 bad loans, the model caught 42 and missed 18. Of the 140 good loans, it wrongly flagged 30 as risky.
Accuracy alone can mislead here, because always answering "good" would already be right 70% of the time.
Full per-class results are in models/classification_report.txt.
What matters most to the model
Permutation importance: how much the test ROC-AUC drops when one input is shuffled.
Input Drop in ROC-AUC
Checking account status 0.127
Credit history 0.015
Monthly payment (engineered) 0.011
Savings account 0.008
Credit amount, log scale (engineered) 0.008
Other debtors or guarantors 0.007
Loan purpose 0.006
Other payment plans 0.005
Checking account status stands far above the rest. This shows overall patterns, not why one particular applicant got their score.
Screenshots
Screenshots of the app will be added to a screenshots/ folder.
Known limitations
Only 1,000 loans from one German bank, collected decades ago. Results will not carry over to other places or times.
With so little data, a 200-loan test set gives noisy numbers. Treat small differences between models as noise, and expect results to shift with a different split.
Scores come from a model trained with balanced class weights, so they are not calibrated probabilities of default.
The low / moderate / higher bands in the app are illustrative cut-offs, not policy.
Age is still used as an input. Removing sex and nationality does not make a model free of bias.
Real credit scoring needs legal compliance, fairness audits and human review.
Live demo
Not deployed yet.