# Credit Card Fraud Detection

Detecting credit card fraud with logistic regression and SMOTE.

## Data

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset on Kaggle (`creditcard.csv`).

- 284,807 transactions, 31 columns
- `Class` is the target: 1 = fraud, 0 = normal
- `V1`–`V28` are anonymised features; `Time` and `Amount` are the only readable ones
- Fraud is extremely rare (well under 1% of rows), so the data is highly imbalanced

The dataset is not included in this repository. The notebook downloads it automatically with `kagglehub`.

## Approach

1. **Set up and load the data**: download the dataset from Kaggle and load it with pandas.
2. **Explore the data**: check the class balance and the distribution of `Amount` and `Time`.
3. **Prepare and split the data**: split into train (80%) and test (20%) sets, stratified on `Class`, then scale the features with a scaler fitted on the training set only.
4. **Balance with SMOTE and train the model**: apply SMOTE to the training set only, then train a logistic regression.
5. **Evaluate the model**: accuracy, precision, recall, F1, ROC AUC, ROC curve and confusion matrix on the untouched test set.

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9741 |
| Precision | 0.0578 |
| Recall    | 0.9184 |
| F1        | 0.1088 |
| ROC AUC   | 0.9708 |

![ROC curve](images/roc_curve.png)

![Confusion matrix](images/confusion_matrix.png)

**Summary:** The model catches 90 of the 98 frauds in the test set (recall of about 92%) and misses only 8. The trade-off is a high number of false alarms: 1,467 normal transactions were flagged as fraud, so only about 6% of the alerts are real frauds. Accuracy is 97.4%, but it says little here because almost all transactions are normal. With a ROC AUC of 0.97 the model separates the two classes very well, which makes it a good first filter, as long as flagged transactions get a second check.

## How to run

```bash
pip install -r requirements.txt
```

Then open `credit_card_fraud_detection.ipynb` in Jupyter, Google Colab or Databricks and run all cells from top to bottom.
