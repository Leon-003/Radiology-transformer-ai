from sklearn.metrics import (
    classification_report,
    f1_score
)

def evaluate_model(
    y_true,
    y_pred
):

    print(
        classification_report(
            y_true,
            y_pred
        )
    )

    print(
        "F1 Score:",
        f1_score(
            y_true,
            y_pred,
            average='micro'
        )
    )