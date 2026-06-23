from hybridrag.retrieve.router import route


def test_text_leaning_for_code():
    d = route("how to fix this python stack trace exception in my function")
    assert d.text_weight > d.vision_weight


def test_vision_leaning_for_charts():
    d = route("which chart shows the revenue table and its axis legend")
    assert d.vision_weight > d.text_weight


def test_balanced_default():
    d = route("tell me about the history of rome")
    assert d.text_weight == 1.0
    assert d.vision_weight == 1.0


def test_codeish_query_is_text():
    d = route("def foo(): return {a: 1}")
    assert d.text_weight > d.vision_weight


def test_both_cues_boost_both():
    d = route("show the json config and the diagram of the api")
    assert d.text_weight > 1.0
    assert d.vision_weight > 1.0
