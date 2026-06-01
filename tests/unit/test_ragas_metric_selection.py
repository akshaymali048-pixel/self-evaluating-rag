from self_evaluating_rag.application.services.evaluation_mode import metric_names_for_mode


def test_metric_names_user_mode():
    assert metric_names_for_mode(run_reference_metrics=False) == {
        "faithfulness",
        "answer_relevancy",
    }


def test_metric_names_evaluation_mode():
    assert metric_names_for_mode(run_reference_metrics=True) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }
