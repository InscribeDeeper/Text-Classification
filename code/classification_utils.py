from sklearn import metrics
from sklearn.model_selection import cross_validate
import pandas as pd


def evaluation_report(y_test, pred, labels=None):
    """Generate report for demand metrics

    Args:
        y_test ([pd.series]): [ground truth label]
        pred ([[pd.series]]): [model prediction]
        labels ([dict], optional): [ground truth label name mapping to index]. Defaults to None.

    Returns:
        [pd.dataframe]: [the detail report table]
    """
    classification_report = pd.DataFrame(metrics.classification_report(y_true=y_test, y_pred=pred, target_names=labels, output_dict=True)).T
    classification_report.loc['micro avg', :] = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=pred, average='micro')
    classification_report.loc['micro avg', 'support'] = classification_report.loc['macro avg', 'support']
    classification_report.loc['accuracy', 'support'] = classification_report.loc['macro avg', 'support']

    print("classification_report:\nf1: ", classification_report.loc[['micro avg', 'macro avg'], 'f1-score'].to_dict(), '\n')
    print(classification_report.round(3).to_string())
    return classification_report


def roc_auc(y_test, y_prob):
    roc_auc = metrics.roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    print("roc_auc: ", roc_auc)


def clf_cv(clf, X_train, y_train, kfold=4, cv_metrics=["precision_macro", "accuracy", "f1_macro", "f1_micro"]):
    """cross validate the model performance on the training set

    Args:
        clf ([class]): [model classifier from sklearn]
        X_train ([np.array]): [train tfidf or tf matrix]
        y_train ([np.array]): [train label]
        kfold (int, optional): [how many fold]. Defaults to 4.
        cv_metrics (list, optional): [matrix to record]. Defaults to ["precision_macro", "accuracy", "f1_macro", "f1_micro"].

    Returns:
        [pd.dataframe]: [the record on every fold]
    """
    cv = cross_validate(clf, X_train, y_train, scoring=cv_metrics, cv=kfold, return_train_score=True)
    cv = pd.DataFrame(cv)
    f1_macro = cv['test_f1_macro'].mean()
    f1_micro = cv['test_f1_micro'].mean()
    print("cv average f1 macro: ", f1_macro)
    print("cv average f1 micro: ", f1_micro)
    return cv
