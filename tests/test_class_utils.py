import numpy as np

from inference_core import resolve_class_name, summarize_class_counts


def test_resolve_class_name_uses_model_names_when_present():
    assert resolve_class_name(0, ["sawit", "kebun_lain"]) == "sawit"
    assert resolve_class_name(1, ["sawit", "kebun_lain"]) == "kebun_lain"


def test_summarize_class_counts_handles_multiple_classes():
    classes = np.array([0, 1, 0, 2, 1], dtype=np.int32)
    counts = summarize_class_counts(classes, ["sawit", "kebun_lain", "semak"])
    assert counts == [("sawit", 2), ("kebun_lain", 2), ("semak", 1)]
