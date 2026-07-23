from vlmbench.models.base import ModelMeta
from vlmbench.models.fake import FakeVLModel
from vlmbench.tasks.base import Example, TaskSpec, run_task, subsample
from vlmbench.tasks.metrics import anls, exact_match


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
