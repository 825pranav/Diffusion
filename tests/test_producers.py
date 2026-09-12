"""
Pure-function tests for the ingestion producers.

No network: these cover the flattening and filtering that decide what ever
reaches Kafka, which is where propagation signals were being lost.
"""

from ingestion.mastodon_producer import _serialize, _should_publish


def _status(**over) -> dict:
    base = {
        "id": "100",
        "content": "<p>hello</p>",
        "account": {"acct": "someone@example.org"},
        "tags": [],
        "language": "en",
        "reblogs_count": 0,
        "favourites_count": 0,
        "url": "https://example.org/@someone/100",
        "created_at": "2026-01-01T00:00:00.000Z",
        "in_reply_to_id": None,
        "reblog": None,
    }
    base.update(over)
    return base


def test_reply_keeps_its_parent():
    rec = _serialize(_status(id="200", in_reply_to_id="100"))
    assert rec["parent_id"] == "100"
    assert rec["type"] == "status"


def test_boost_takes_the_parent_and_language_from_the_status_it_carries():
    # A boost's own content is empty; everything identifying it sits in `reblog`.
    rec = _serialize(_status(
        id="300", content="", language=None,
        reblog={"id": "100", "language": "de"},
    ))
    assert rec["type"] == "boost"
    assert rec["parent_id"] == "100"
    assert rec["langs"] == ["de"]
    assert rec["text"] == ""


def test_original_status_has_no_parent():
    rec = _serialize(_status())
    assert rec["parent_id"] is None
    assert rec["type"] == "status"


def test_a_boost_is_published_despite_having_no_text():
    """
    The guard used to be `if record["text"]`, which dropped every boost — and
    with it the propagation edge the boost existed to carry.
    """
    boost = _serialize(_status(id="300", content="", reblog={"id": "100"}))
    assert boost["text"] == ""
    assert _should_publish(boost) is True


def test_a_textless_status_with_no_parent_is_not_published():
    # Nothing to say and nowhere it came from: it contributes neither topic
    # mentions nor cascade structure.
    assert _should_publish(_serialize(_status(content=""))) is False


def test_an_ordinary_post_is_published():
    assert _should_publish(_serialize(_status())) is True
