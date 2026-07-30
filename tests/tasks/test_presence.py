from vlmbench.tasks.presence import PROMPT_TEMPLATE, build_presence
from vlmbench.tasks.registry import known_tasks


class _Img:
    def convert(self, mode):
        return self


def test_presence_coco_registered():
    assert "presence-coco" in known_tasks()


def test_build_presence_labels_yes_no_from_present_set():
    rows = [(_Img(), {"person"}), (_Img(), {"car", "dog"})]
    ex = build_presence(rows, classes=("person", "car"), seed=1)
    labels = {(e.prompt, e.answers[0]) for e in ex}
    assert (PROMPT_TEMPLATE.format(cls="person"), "yes") in labels
    assert (PROMPT_TEMPLATE.format(cls="car"), "no") in labels
    assert (PROMPT_TEMPLATE.format(cls="car"), "yes") in labels


def test_build_presence_is_balanced():
    rows = [(_Img(), {"person"}), (_Img(), {"person"}),
            (_Img(), set()), (_Img(), set())]
    ex = build_presence(rows, classes=("person",), seed=1)
    yes = sum(e.answers[0] == "yes" for e in ex)
    no = sum(e.answers[0] == "no" for e in ex)
    assert yes == no == 2
