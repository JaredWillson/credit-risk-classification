# Credit Risk Analysis

## Overview

The intent of the code within this repository is to use a supervised machine learning algorithm using various available loan dimensions on historical loans to train the machine learning algorithm on how to identify high-risk loans. This algorithm could be run against future loans to identify which loans are most likely to be healthy and which are most likely to be at high-risk so that underwriters or other financial analysts could focus their attention on the loans that are most likely to be at risk based upon dimensions that may not be obvious indicators of an issue. 

The overall process followed will be as follows:
- An existing set of loans that are already known to be either healthy or "at risk" will be loaded into a dataframe for analysis
- The `loan_status` will be split off from the dimensions for analysis; note that a loan_status of '0' indicates a healthy loan, where a loan_status of '1' indicates an at-risk loan
- The dataframe with the `loan_status` dropped will be designated as our X dataset of features
- The `loan_status` values will be saved as the label array
- Train-Test-Split will be used to segregate the data into 75% of the data used for training and 25% used for testing
- A random_state of `1` will be used to ensure repeatability/validation of the model
- `Logistic Regression` is used to both regularize the X features and to separate them into two classes; this method is works well for linearly separable datasets and requires minimal resources to run.
- The model is instantiated, fit, and then run against the test data.
- A confusion matrix and classification report are generated to allow analysis of the usefulness of the model

The dimensions used for the analysis are:
- The size of the loan
- The interest rate for the loan
- The income of the borrower
- The debt to income ratio
- The number of accounts for the borrower
- The number of derogatory marks
- The total debt outstanding for the borrower

Based on these dimensions, the model will hopefully predict with reasonable accuracy which loans are at high risk. Specifically, we are trying to make a binary prediction of the `loan_status` value where a value of 0 means the loan is healthy, and a value of 1 means the loan is at high risk. 

## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

### Logistic Regression Model:
    - Accuracy: 99.2%
    - Precision for Healthy Loans: 100%
    - Precision for "At Risk" loans: 85%
    - Recall for Healthy Loans: 99%
    - Recall for "At Risk" loans: 91%


## Summary

Since most of the training data were healthy loans (loan_status = 0), we are much more interested in the ability of the model to predict "at risk" loans than we are protecting against false positives. An at risk loan has the potential to cost the underwriters a large amount of money, so flagging these loans for additional review is critical. As a result, we are most interested in the "recall" value for the "at risk" loans (loan_status = 1). I believe that any value above 90% is likely to be useful to underwriters as a quick screening for the health of a prospective loan. The analysis performed using logistic regression (in order to regularize the X features and to separate them into classes) worked very well, resulting in a recall of 91%. That exceeds my own self-imposed threshold of 90%. In addition, it had very few false positives, with a recall of 99% for "not at risk" loans. 


