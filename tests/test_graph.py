from langchain_core.documents import Document

from support_agent.graph import HANDOFF_TEXT, build_support_graph


class FakeVectorStore:
    def __init__(self, matches):
        self.matches = matches

    def similarity_search_with_relevance_scores(self, query, k=4):
        return self.matches[:k]


class FakeReply:
    def __init__(self, content):
        self.content = content


class FakeChatModel:
    def __init__(self, content="Returns are allowed within 30 days [1]."):
        self.content = content
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return FakeReply(self.content)


def test_no_relevant_documents_hands_off_without_calling_model():
    model = FakeChatModel()
    graph = build_support_graph(FakeVectorStore([]), model)

    result = graph.invoke({"question": "Can I return an opened item?"})

    assert result["handoff_required"] is True
    assert result["answer"] == HANDOFF_TEXT
    assert model.calls == 0


def test_relevant_document_is_cited_and_answered():
    document = Document(
        page_content="Sealed, unused items may be returned within 30 days.",
        metadata={"source": "returns.md"},
    )
    model = FakeChatModel()
    graph = build_support_graph(FakeVectorStore([(document, 0.82)]), model)

    result = graph.invoke({"question": "What is the return window?"})

    assert result["handoff_required"] is False
    assert "30 days" in result["answer"]
    assert result["sources"] == [{"id": 1, "source": "returns.md", "page": None}]
    assert model.calls == 1


def test_low_relevance_result_hands_off():
    document = Document(page_content="Shipping takes 3 to 5 days.", metadata={"source": "shipping.md"})
    model = FakeChatModel()
    graph = build_support_graph(FakeVectorStore([(document, 0.1)]), model, score_threshold=0.25)

    result = graph.invoke({"question": "Can I return a product?"})

    assert result["handoff_required"] is True
    assert model.calls == 0


def test_model_can_request_a_human_handoff():
    document = Document(page_content="Shipping takes 3 to 5 days.", metadata={"source": "shipping.md"})
    model = FakeChatModel("[[HUMAN_HANDOFF]] I cannot confirm the exception policy.")
    graph = build_support_graph(FakeVectorStore([(document, 0.8)]), model)

    result = graph.invoke({"question": "Can you guarantee delivery tomorrow?"})

    assert result["handoff_required"] is True
    assert "I cannot confirm" in result["answer"]

