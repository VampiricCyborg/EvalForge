from evalforge.scorers import SCORERS, ExactMatchScorer, SemanticSimilarityScorer


def test_registry_maps_names_to_classes():
    assert SCORERS["exact_match"] is ExactMatchScorer
    assert SCORERS["semantic"] is SemanticSimilarityScorer


def test_registry_keys_match_class_name_attribute():
    for key, cls in SCORERS.items():
        assert cls.name == key
