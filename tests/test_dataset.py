import glob
import json
import os

from docqa import config

norm = lambda s: " ".join(s.lower().split())


def _load():
    data = json.load(open(config.EVAL_DIR / "eval_dataset.json", encoding="utf-8"))
    docs = {os.path.splitext(os.path.basename(p))[0]: norm(open(p, encoding="utf-8").read())
            for p in glob.glob(str(config.RAW_DIR / "*.txt"))}
    return data, docs


def test_dataset_size_in_target_range():
    data, _ = _load()
    assert 20 <= len(data) <= 50


def test_every_ground_truth_span_appears_exactly_once_in_its_document():
    data, docs = _load()
    for item in data:
        text = docs[item["source_doc"]]
        assert text.count(norm(item["source_snippet"])) == 1, item["id"]


def test_spans_are_unique_across_the_whole_corpus():
    data, docs = _load()
    corpus = " ".join(docs.values())
    for item in data:
        assert corpus.count(norm(item["source_snippet"])) == 1, item["id"]


def test_required_fields_present():
    data, _ = _load()
    for item in data:
        assert {"id", "question", "answer", "source_snippet", "source_doc"} <= item.keys()
