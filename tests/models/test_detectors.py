from vlmbench.models.detectors.base_detector import target_class


def test_target_class_parses_the_prompt_template():
    assert target_class("Is there a person in this image? Answer yes or no.") == "person"
    assert target_class("Is there a traffic cone in this image? Answer yes or no.") == "traffic cone"


def test_fixed_vocab_answers_no_for_unknown_class():
    from vlmbench.models.detectors.ultralytics_detectors import FixedVocabYolo
    det = FixedVocabYolo.__new__(FixedVocabYolo)  # bypass model load
    det._names = {0: "person", 2: "car"}
    det._threshold = 0.25
    det._model = None  # not called for an unknown class
    assert det.infer(object(), "Is there a dog in this image? Answer yes or no.") == "no"
