"""
Knowledge Graph loading and path sampling.

See spec/schemas.md for kg.json format.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional


def load_kg(path: Path) -> Dict[str, Any]:
    """Load knowledge graph from JSON file."""
    with open(path) as f:
        return json.load(f)


def build_adjacency(kg: Dict[str, Any]) -> Dict[str, List[Tuple[str, str, int]]]:
    """Build adjacency list: entity -> [(rel, dst, edge_idx), ...]"""
    adj = {}
    for i, edge in enumerate(kg["edges"]):
        src = edge["src"]
        if src not in adj:
            adj[src] = []
        adj[src].append((edge["rel"], edge["dst"], i))
    return adj


def get_entity_vocab(kg: Dict[str, Any]) -> Set[str]:
    """Return set of all entity IDs for matching."""
    return {e["id"] for e in kg["entities"]}


def get_entity_by_type(kg: Dict[str, Any], entity_type: str) -> List[str]:
    """Return list of entity IDs matching a specific type."""
    return [e["id"] for e in kg["entities"] if e.get("type") == entity_type]


def get_entity_types(kg: Dict[str, Any]) -> Dict[str, str]:
    """Return mapping from entity ID to its type."""
    return {e["id"]: e.get("type", "unknown") for e in kg["entities"]}


def sample_path(
    kg: Dict[str, Any],
    length: int,
    seed: Optional[int] = None,
    start_types: Optional[List[str]] = None,
    preferred_relations: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Sample a random path of given length from the KG.

    Args:
        kg: Knowledge graph dictionary
        length: Number of edges in the path (path will have length+1 entities)
        seed: Random seed for reproducibility
        start_types: If provided, start from entities of these types (e.g., ["symptom"])
        preferred_relations: If provided, prefer edges with these relations

    Returns:
        {
            "entities": ["A", "B", "C"],
            "edges": [{"src": "A", "rel": "r1", "dst": "B"}, ...]
        }
    """
    if seed is not None:
        random.seed(seed)

    adj = build_adjacency(kg)
    entity_types = get_entity_types(kg)

    # Get valid starting entities (must have outgoing edges for path of requested length)
    valid_starts = []
    for entity_id in adj.keys():
        if start_types is None or entity_types.get(entity_id) in start_types:
            valid_starts.append(entity_id)

    if not valid_starts:
        raise ValueError(f"No valid starting entities found for types: {start_types}")

    # Try multiple times to find a valid path
    max_attempts = 100
    for _ in range(max_attempts):
        # Pick a random starting entity
        current = random.choice(valid_starts)
        visited = {current}
        path_entities = [current]
        path_edges = []

        # Random walk
        success = True
        for step in range(length):
            # Get available next hops (not visited)
            if current not in adj:
                success = False
                break

            candidates = [(rel, dst, idx) for rel, dst, idx in adj[current] if dst not in visited]

            if not candidates:
                success = False
                break

            # Prefer certain relation types if specified
            if preferred_relations:
                preferred = [c for c in candidates if c[0] in preferred_relations]
                if preferred:
                    candidates = preferred

            # Pick a random next hop
            rel, dst, edge_idx = random.choice(candidates)
            edge = kg["edges"][edge_idx]

            path_edges.append({
                "src": edge["src"],
                "rel": edge["rel"],
                "dst": edge["dst"]
            })
            path_entities.append(dst)
            visited.add(dst)
            current = dst

        if success and len(path_edges) == length:
            return {
                "entities": path_entities,
                "edges": path_edges
            }

    raise ValueError(f"Could not sample path of length {length} after {max_attempts} attempts")


def sample_diverse_paths(
    kg: Dict[str, Any],
    n_paths: int,
    length: int,
    seed: Optional[int] = None,
    start_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Sample multiple diverse paths from the KG.

    Tries to maximize diversity by avoiding reusing the same starting entities.
    """
    if seed is not None:
        random.seed(seed)

    paths = []
    used_starts = set()

    for i in range(n_paths):
        # Try to get a path with a new starting entity
        path_seed = seed + i if seed is not None else None

        try:
            path = sample_path(
                kg=kg,
                length=length,
                seed=path_seed,
                start_types=start_types
            )

            # Check if we've used this start before
            start = path["entities"][0]
            if start in used_starts and len(used_starts) < 50:
                # Try a few more times to get a different start
                for retry in range(5):
                    retry_seed = path_seed + 1000 + retry if path_seed else None
                    retry_path = sample_path(
                        kg=kg,
                        length=length,
                        seed=retry_seed,
                        start_types=start_types
                    )
                    if retry_path["entities"][0] not in used_starts:
                        path = retry_path
                        break

            used_starts.add(path["entities"][0])
            paths.append(path)

        except ValueError:
            # If we can't get more paths, break
            break

    return paths


def get_neighbors(kg: Dict[str, Any], entity_id: str, max_hops: int = 1) -> Set[str]:
    """Get all entities within max_hops of the given entity."""
    adj = build_adjacency(kg)

    # Also build reverse adjacency for incoming edges
    rev_adj = {}
    for edge in kg["edges"]:
        dst = edge["dst"]
        if dst not in rev_adj:
            rev_adj[dst] = []
        rev_adj[dst].append(edge["src"])

    neighbors = set()
    frontier = {entity_id}

    for _ in range(max_hops):
        new_frontier = set()
        for node in frontier:
            # Outgoing edges
            if node in adj:
                for _, dst, _ in adj[node]:
                    if dst not in neighbors and dst != entity_id:
                        neighbors.add(dst)
                        new_frontier.add(dst)
            # Incoming edges
            if node in rev_adj:
                for src in rev_adj[node]:
                    if src not in neighbors and src != entity_id:
                        neighbors.add(src)
                        new_frontier.add(src)
        frontier = new_frontier

    return neighbors
