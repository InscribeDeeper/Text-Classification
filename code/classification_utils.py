from sklearn import metrics
from sklearn.model_selection import cross_validate
import pandas as pd


def evaluation_report(y_test, pred, labels=None):
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
    cv = cross_validate(clf, X_train, y_train, scoring=cv_metrics, cv=kfold, return_train_score=True)
    cv = pd.DataFrame(cv)
    f1_macro = cv['test_f1_macro'].mean()
    f1_micro = cv['test_f1_micro'].mean()
    print("cv average f1 macro: ", f1_macro)
    print("cv average f1 micro: ", f1_micro)
    return cv