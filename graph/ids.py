"""
Node-ID prefix constants for the propagation graph.

Every node ID is built as "<prefix><raw_id>".  Defining the prefixes here
gives entity_extractor.py (write path) and api/routes.py (read path) a
single source of truth — if a prefix ever changes, one edit here covers both.
"""

# ── Content nodes ─────────────────────────────────────────────────────────────
BLUESKY_CONTENT_PREFIX = "bluesky:"
MASTODON_CONTENT_PREFIX = "mastodon:"
HN_CONTENT_PREFIX = "hn:"

# ── Actor / author nodes ──────────────────────────────────────────────────────
BLUESKY_USER_PREFIX = "bluesky:user:"
MASTODON_USER_PREFIX = "mastodon:user:"
MASTODON_TAG_PREFIX = "mastodon:tag:"
HN_USER_PREFIX = "hn:user:"
GITHUB_USER_PREFIX = "github:user:"

# ── Repo nodes ────────────────────────────────────────────────────────────────
GITHUB_REPO_PREFIX = "github:repo:"

# ── Named-entity nodes ────────────────────────────────────────────────────────
# Full ID: entity:<ner_label_lower>:<normalised_text>
ENTITY_PREFIX = "entity:"
