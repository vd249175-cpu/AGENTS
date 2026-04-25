"""Runtime state helpers for graph query buckets."""

from typing import Any, Mapping

from langchain.agents.middleware import AgentState


GraphItemPayload = dict[str, Any]
GraphItemMap = dict[str, GraphItemPayload]


class GraphQueryToolStateTydict(AgentState, total=False):
    useful_items: GraphItemMap
    blocked_items: GraphItemMap
    path_blocked_items: GraphItemMap


def normalize_item_map(value: Any) -> GraphItemMap:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(node_id): dict(payload)
        for node_id, payload in value.items()
        if isinstance(payload, Mapping)
    }


def get_useful_items(state: Mapping[str, Any] | None) -> GraphItemMap:
    return normalize_item_map((state or {}).get("useful_items"))


def get_blocked_items(state: Mapping[str, Any] | None) -> GraphItemMap:
    raw_state = state or {}
    if "blocked_items" in raw_state:
        return normalize_item_map(raw_state.get("blocked_items"))
    return normalize_item_map(raw_state.get("path_blocked_items"))


def blocked_ids(state: Mapping[str, Any] | None) -> set[str]:
    return set(get_blocked_items(state))


def body_chars(items: Mapping[str, GraphItemPayload]) -> int:
    return sum(len(str(payload.get("body") or "")) for payload in items.values())


def mark_useful(
    state: Mapping[str, Any] | None,
    items: list[GraphItemPayload],
    *,
    rationale: str,
) -> dict[str, GraphItemMap]:
    useful = get_useful_items(state)
    blocked = get_blocked_items(state)
    for item in items:
        node_id = str(item.get("node_id") or "").strip()
        if not node_id:
            continue
        payload = dict(item)
        payload["rationale"] = rationale
        useful[node_id] = payload
        blocked.pop(node_id, None)
    return {"useful_items": useful, "blocked_items": blocked}


def mark_blocked(
    state: Mapping[str, Any] | None,
    items: list[GraphItemPayload],
    *,
    rationale: str,
) -> dict[str, GraphItemMap]:
    useful = get_useful_items(state)
    blocked = get_blocked_items(state)
    for item in items:
        node_id = str(item.get("node_id") or "").strip()
        if not node_id:
            continue
        payload = dict(item)
        payload["rationale"] = rationale
        blocked[node_id] = payload
        useful.pop(node_id, None)
    return {"useful_items": useful, "blocked_items": blocked}


def clear_blocked(
    state: Mapping[str, Any] | None,
    *,
    node_ids: list[str] | None = None,
) -> dict[str, Any]:
    useful = get_useful_items(state)
    blocked = get_blocked_items(state)
    if node_ids is None:
        cleared = list(blocked)
        blocked.clear()
    else:
        cleared = []
        for node_id in node_ids:
            cleaned = str(node_id).strip()
            if cleaned in blocked:
                blocked.pop(cleaned, None)
                cleared.append(cleaned)
    return {
        "useful_items": useful,
        "blocked_items": blocked,
        "cleared_ids": cleared,
    }
