# Machine Learning on Existing Loan Metrics to Flag High-Risk Loans for Manual Review
The intent of the code within this repository is to use a supervised machine learning algorithm using various available loan dimensions on historical loans to train the machine learning algorithm on how to identify high-risk loans. This algorithm could be run against future loans to identify which loans are most likely to be healthy and which are most likely to be at high-risk so that underwriters or other financial analysts could focus their attention on the loans that are most likely to be at risk based upon dimensions that may not be obvious indicators of an issue. 

The overall process followed will be as follows:
- An existing set of loans that are already known to be either healthy or "at risk" will be loaded into a dataframe for analysis
- The `loan_status` will be split off from the dimensions for analysis; note that a loan_status of '0' indicates a healthy loan, where a loan_status of '1' indicates an at-risk loan
- The dataframe with the `loan_status` dropped will be designated as our X dataset of features
- The `loan_status` values will be saved as the label array
- Train-Test-Split will be used to segregate the data into 75% of the data used for training and 25% used for testing
- A random_state of `1` will be used to ensure repeatability/validation of the model
- `Logistic Regression` is used to both regularize the X features and to separate them into two classes; this method is works well for linearly separable datasets.
- The model is instantiated, fit, and then run against the test data.
- A confusion matrix and classification report are generated to allow analysis of the usefulness of the model

At the end of the notebook, you will find the answer to the question of whether this machine learning model would be useful. If it can accurately predict, with a reasonable degree of accuracy, which loans are at risk then analysts and underwriters could use it to determine which loans should be flagged for manual review.

The `Resources` subdirectory contains the datafile for training and analysis. The `Credit_Risk` subdirectory contains a Jupyter notebook called `credit_risk_classification.ipynb` with all the required python code. The `credit_risk_analysis.md` report contains the analysis of results.

All code is my own, but Xpert Learning Assistance was used to lookup the correct syntax for the logistic regression model.