import importlib.util
import tempfile
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "Evaluation" / "Code" / "new_rag.py"
spec = importlib.util.spec_from_file_location("new_rag", MODULE_PATH)
new_rag = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(new_rag)


class _Doc:
    def __init__(self, page_content):
        self.page_content = page_content


class _Retriever:
    def get_relevant_documents(self, query):
        return [_Doc(f"ctx for {query}\nline2")]


class _QA:
    retriever = _Retriever()

    def run(self, query):
        return f"Some preface\nHelpful Answer: answer for {query}\n\nOther text"


def test_dataset_dict_parses_questions_and_answers(monkeypatch):
    monkeypatch.setattr(new_rag, "qa", _QA())

    content = """
# What is AI?
Artificial intelligence.

# What is ML?
Machine learning.
""".strip()

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "qa.txt"
        f.write_text(content, encoding="utf-8")
        data = new_rag.dataset_dict(str(f))

    assert data["question"] == ["What is AI?", "What is ML?"]
    assert data["ground_truths"] == ["Artificial intelligence.", "Machine learning."]
    assert data["answer"] == ["answer for What is AI?", "answer for What is ML?"]
    assert data["contexts"] == [[["ctx for What is AI?", "line2"]], [["ctx for What is ML?", "line2"]]]


def test_extract_helpful_answer_raises_on_missing_marker():
    try:
        new_rag._extract_helpful_answer("No marker here")
        assert False, "Expected ValueError"
    except ValueError:
        assert True
