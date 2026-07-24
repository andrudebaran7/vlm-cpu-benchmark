from vlmbench.models.base import ModelMeta
from vlmbench.models.fake import FakeVLModel
from vlmbench.tasks.base import Example, TaskSpec, run_task, subsample
from vlmbench.tasks.docvqa import _rows_to_examples
from vlmbench.tasks.metrics import anls, exact_match
from vlmbench.tasks.registry import known_tasks


def test_exact_match_and_anls():
    assert exact_match("Paris", ["paris"]) == 1.0
    assert exact_match("London", ["paris"]) == 0.0
    assert anls("paris", ["paris"]) == 1.0
    assert 0.0 <= anls("pariss", ["paris"]) < 1.0


def test_subsample_is_deterministic():
    examples = [Example(image=None, prompt=str(i), answers=[str(i)]) for i in range(20)]
    a = subsample(examples, n=5, seed=7)
    b = subsample(examples, n=5, seed=7)
    assert [e.prompt for e in a] == [e.prompt for e in b]
    assert len(a) == 5


def test_docvqa_is_registered():
    assert "docvqa" in known_tasks()


def test_docvqa_row_mapping():
    class _Img:
        def __init__(self, tag):
            self.tag = tag
            self.converted = False

        def convert(self, mode):
            self.converted = True
            return self

    rows = [
        # multilingual dict query -> English is selected
        {"image": _Img("a"), "query": {"en": "What is the total?", "es": "..."},
         "answers": ["$5.00", "5.00"]},
        # plain-string query still works; falls back to singular 'answer'
        {"image": _Img("b"), "query": "Who signed it?",
         "answers": None, "answer": "Alice"},
    ]
    examples = _rows_to_examples(rows)
    assert [e.prompt for e in examples] == ["What is the total?", "Who signed it?"]
    assert examples[0].answers == ["$5.00", "5.00"]
    assert examples[1].answers == ["Alice"]  # falls back to singular 'answer'
    assert examples[0].image.converted is True  # RGB-converted


def test_run_task_scores_predictions():
    meta = ModelMeta(name="fake", params_b=0.1, license="Apache-2.0",
                     source="fake", supported_backends=("fp32",))
    # Model echoes the prompt as the answer.
    model = FakeVLModel(meta=meta, responder=lambda image, prompt: prompt)
    examples = [Example(image=None, prompt="cat", answers=["cat"]),
                Example(image=None, prompt="dog", answers=["fish"])]
    spec = TaskSpec(name="toy", metric=exact_match)
    score, preds = run_task(model, examples, spec)
    assert preds == ["cat", "dog"]
    assert score == 0.5
