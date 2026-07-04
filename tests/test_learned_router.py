
import numpy as np
import pytest

from hybridrag.config import HybridConfig
from hybridrag.engine import HybridRAG
from hybridrag.retrieve.router import (
    FEATURE_NAMES,
    DEFAULT_TRAINING_EXAMPLES,
    HeuristicRouter,
    LearnedRouter,
    build_router,
    extract_features,
    route,
)


# ------------------------------------------------------------------ features
def test_feature_vector_shape_and_bias():
    f = extract_features("which chart shows the revenue table")
    assert f.shape == (len(FEATURE_NAMES),)
    assert f[0] == 1.0  # bias term
    assert np.all(f >= 0.0)


def test_features_separate_text_and_vision_cues():
    text_f = extract_features("fix the python stack trace exception in def foo()")
    vis_f = extract_features("which chart and table and figure and diagram")
    # index 1 = text_cue_density, index 2 = vision_cue_density, 3 = codeish
    assert text_f[1] > 0 and text_f[3] == 1.0
    assert vis_f[2] > text_f[2]


# ------------------------------------------------------------------ training
def test_default_router_is_deterministic():
    a = LearnedRouter.default()
    b = LearnedRouter.default()
    assert np.allclose(a.weights, b.weights)


def test_learned_router_routes_text_query_to_text():
    r = LearnedRouter.default()
    d = r.route("how do I fix this json parsing exception in my function")
    assert d.vision_probability is not None
    assert d.vision_probability < 0.5
    assert d.text_weight > d.vision_weight


def test_learned_router_routes_visual_query_to_vision():
    r = LearnedRouter.default()
    d = r.route("which chart shows the revenue table and its axis legend")
    assert d.vision_probability > 0.5
    assert d.vision_weight > d.text_weight


def test_learned_router_separates_the_seed_set():
    # After training, the classifier should order the seed examples correctly:
    # every clearly-text example scores below every clearly-vision example.
    r = LearnedRouter.default()
    text_scores = [r.vision_probability(q) for q, y in DEFAULT_TRAINING_EXAMPLES if y <= 0.2]
    vis_scores = [r.vision_probability(q) for q, y in DEFAULT_TRAINING_EXAMPLES if y >= 0.9]
    assert max(text_scores) < min(vis_scores)


def test_weights_never_zero_a_modality():
    r = LearnedRouter.default()
    for q, _ in DEFAULT_TRAINING_EXAMPLES:
        d = r.route(q)
        assert d.text_weight > 0.0
        assert d.vision_weight > 0.0


def test_lean_zero_keeps_weights_balanced():
    r = LearnedRouter.default()
    r.lean = 0.0
    d = r.route("which chart shows the revenue table")
    assert d.text_weight == pytest.approx(1.0)
    assert d.vision_weight == pytest.approx(1.0)


def test_fit_changes_behaviour():
    # Train a router that always prefers vision, and confirm it overrides cues.
    r = LearnedRouter().fit([("anything at all", 1.0), ("more text here", 1.0)])
    d = r.route("stack trace exception in json")
    assert d.vision_probability > 0.5


def test_fit_requires_examples():
    with pytest.raises(ValueError):
        LearnedRouter().fit([])


# --------------------------------------------------------------- persistence
def test_save_load_roundtrip(tmp_path):
    r = LearnedRouter.default()
    path = tmp_path / "router.json"
    r.save(str(path))
    loaded = LearnedRouter.load(str(path))
    assert np.allclose(loaded.weights, r.weights)
    q = "which diagram shows the api layout"
    assert loaded.route(q).vision_probability == pytest.approx(r.route(q).vision_probability)


def test_from_dict_rejects_wrong_width():
    with pytest.raises(ValueError):
        LearnedRouter.from_dict({"weights": [0.1, 0.2]})


def test_to_dict_shape():
    d = LearnedRouter.default().to_dict()
    assert d["model"] == "learned"
    assert d["feature_names"] == list(FEATURE_NAMES)
    assert len(d["weights"]) == len(FEATURE_NAMES)


# --------------------------------------------------------------- build_router
def test_build_router_default_is_heuristic():
    router = build_router(HybridConfig())
    assert isinstance(router, HeuristicRouter)
    assert router.model == "heuristic"


def test_build_router_learned():
    router = build_router(HybridConfig(router_model="learned"))
    assert isinstance(router, LearnedRouter)


def test_build_router_learned_from_weights_path(tmp_path):
    path = tmp_path / "w.json"
    LearnedRouter.default().save(str(path))
    router = build_router(HybridConfig(router_model="learned", router_weights_path=str(path)))
    assert isinstance(router, LearnedRouter)


# --------------------------------------------------------------- engine wiring
def test_engine_uses_configured_router():
    heuristic = HybridRAG(HybridConfig())
    learned = HybridRAG(HybridConfig(router_model="learned"))
    assert heuristic.router.model == "heuristic"
    assert learned.router.model == "learned"


def test_engine_search_with_learned_router():
    rag = HybridRAG(HybridConfig(router_model="learned"))
    rag.add_text("d1", "Revenue grew across every region this quarter." * 5, title="Report")
    rag.add_text("d2", "def parse(x): return json.loads(x)  # error handling", title="Code")
    results = rag.search("which chart shows the revenue", top_k=2)
    assert results  # routing must not break search


# --------------------------------------------------------------- back-compat
def test_heuristic_route_still_works():
    d = route("show the json config and the diagram of the api")
    assert d.text_weight > 1.0 and d.vision_weight > 1.0
    assert d.vision_probability is None  # heuristic leaves probability unset
