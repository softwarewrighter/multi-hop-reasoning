"""
Reward computation for model completions.

See spec/reward.md for full specification.
"""

import re
from typing import Dict, Any, Set, List, Tuple
from collections import Counter


# Default reward parameters
DEFAULT_CORRECT_REWARD = 1.0
DEFAULT_INCORRECT_REWARD = -2.0
DEFAULT_MIN_HITS = 2
DEFAULT_MAX_PATH_REWARD = 1.0
DEFAULT_REPEAT_MAX = 2
DEFAULT_REPEAT_PENALTY = 0.5
DEFAULT_W_CORR = 1.0
DEFAULT_W_PATH = 0.5


def parse_completion(completion: str) -> Tuple[bool, str, str]:
    """
    Parse model completion into trace and answer.

    Returns:
        (valid_format, trace_text, answer)
    """
    trace_match = re.search(r'TRACE:\s*(.+?)(?=ANSWER:|$)', completion, re.DOTALL)
    answer_match = re.search(r'ANSWER:\s*([A-D])', completion)

    if not trace_match or not answer_match:
        return False, "", ""

    trace_text = trace_match.group(1).strip()
    answer = answer_match.group(1).strip()

    return True, trace_text, answer


def extract_entities(text: str, entity_vocab: Set[str]) -> List[str]:
    """Extract entities from text using vocabulary matching."""
    found = []
    for entity in entity_vocab:
        # Simple substring matching - could be improved with word boundaries
        if entity in text:
            found.append(entity)
    return found


def count_entity_mentions(text: str, entities: List[str]) -> Dict[str, int]:
    """Count how many times each entity appears in text."""
    counts = {}
    for entity in entities:
        counts[entity] = text.count(entity)
    return counts


def compute_reward(
    completion: str,
    answer_star: str,
    path_entities: List[str],
    entity_vocab: Set[str],
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Compute reward breakdown for a completion.

    Args:
        completion: Raw model output
        answer_star: Correct answer (A/B/C/D)
        path_entities: Ground-truth path entities
        entity_vocab: All entity IDs from KG
        config: Reward configuration (weights, thresholds)

    Returns:
        Reward breakdown dict matching episodes.jsonl schema
    """
    config = config or {}

    # Parse completion
    valid_format, trace_text, answer = parse_completion(completion)

    if not valid_format:
        return {
            "correctness": DEFAULT_INCORRECT_REWARD,
            "path_coverage": 0.0,
            "path_reward": 0.0,
            "spam_penalty": 0.0,
            "total": DEFAULT_INCORRECT_REWARD,
            "parsed": {
                "answer": "",
                "trace_text": "",
                "trace_entities": [],
                "path_entities": path_entities,
                "valid_format": False
            },
            "debug": {
                "hits": 0,
                "entity_repeat_counts": {}
            }
        }

    # Extract entities from trace
    trace_entities = extract_entities(trace_text, entity_vocab)
    entity_counts = count_entity_mentions(trace_text, trace_entities)

    # Correctness reward
    correct_reward = config.get("correct", DEFAULT_CORRECT_REWARD)
    incorrect_reward = config.get("incorrect", DEFAULT_INCORRECT_REWARD)
    correctness = correct_reward if answer == answer_star else incorrect_reward

    # Path coverage
    path_set = set(path_entities)
    trace_set = set(trace_entities)
    hits = len(path_set & trace_set)
    coverage = hits / max(1, len(path_set))

    # Path reward with min_hits constraint
    min_hits = config.get("min_hits", DEFAULT_MIN_HITS)
    max_path = config.get("max_path_reward", DEFAULT_MAX_PATH_REWARD)

    if hits < min_hits:
        path_reward = 0.0
    else:
        path_reward = min(coverage, max_path)

    # Spam penalty
    repeat_max = config.get("repeat_entity_max", DEFAULT_REPEAT_MAX)
    repeat_penalty = config.get("repeat_penalty", DEFAULT_REPEAT_PENALTY)

    spam_penalty = 0.0
    for count in entity_counts.values():
        if count > repeat_max:
            spam_penalty = repeat_penalty
            break

    # Total reward
    w_corr = config.get("w_corr", DEFAULT_W_CORR)
    w_path = config.get("w_path", DEFAULT_W_PATH)
    total = (w_corr * correctness) + (w_path * path_reward) - spam_penalty

    return {
        "correctness": correctness,
        "path_coverage": coverage,
        "path_reward": path_reward,
        "spam_penalty": spam_penalty,
        "total": total,
        "parsed": {
            "answer": answer,
            "trace_text": trace_text,
            "trace_entities": trace_entities,
            "path_entities": path_entities,
            "valid_format": True
        },
        "debug": {
            "hits": hits,
            "entity_repeat_counts": entity_counts
        }
    }
