import pandas as pd
from analysis.evaluator import ModelEvaluator


def test_metrics():
    df = pd.DataFrame({"actual": [1, 2, 3], "pred": [1, 2, 4]})
    eval = ModelEvaluator(df, actual_col="actual", predicted_col="pred")
    assert eval.mae() == 0.3333333333333333
    assert round(eval.mse(), 2) == 0.33
