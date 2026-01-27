"""
Tests for the core multi-hop reasoning library.
"""

import json
import pytest
from pathlib import Path
import tempfile

from core.kg import load_kg, sample_path, get_entity_vocab, build_adjacency, get_neighbors
from core.dataset import generate_mcq, generate_distractors, generate_dataset, build_prompt
from core.reward import compute_reward, parse_completion, extract_entities


# Test fixtures
@pytest.fixture
def sample_kg():
    """Create a minimal test KG."""
    return {
        "version": "1.0",
        "domain": "test",
        "entities": [
            {"id": "SymptomA", "label": "Symptom A", "type": "symptom"},
            {"id": "CauseB", "label": "Cause B", "type": "cause"},
            {"id": "FixC", "label": "Fix C", "type": "fix"},
            {"id": "DiagD", "label": "Diagnostic D", "type": "diagnostic"},
            {"id": "ToolE", "label": "Tool E", "type": "tool"},
            {"id": "CauseF", "label": "Cause F", "type": "cause"},
        ],
        "relations": [
            {"id": "caused_by", "label": "caused by"},
            {"id": "fixed_by", "label": "fixed by"},
            {"id": "diagnosed_by", "label": "diagnosed by"},
            {"id": "uses_tool", "label": "uses tool"},
        ],
        "edges": [
            {"src": "SymptomA", "rel": "caused_by", "dst": "CauseB"},
            {"src": "CauseB", "rel": "fixed_by", "dst": "FixC"},
            {"src": "SymptomA", "rel": "diagnosed_by", "dst": "DiagD"},
            {"src": "DiagD", "rel": "uses_tool", "dst": "ToolE"},
            {"src": "CauseB", "rel": "caused_by", "dst": "CauseF"},
        ]
    }


@pytest.fixture
def real_kg():
    """Load the actual KG."""
    kg_path = Path(__file__).parent.parent / "data" / "kg.json"
    if not kg_path.exists():
        pytest.skip("Real KG not available")
    return load_kg(kg_path)


class TestKG:
    """Tests for kg.py module."""

    def test_build_adjacency(self, sample_kg):
        """Test adjacency list construction."""
        adj = build_adjacency(sample_kg)

        assert "SymptomA" in adj
        assert len(adj["SymptomA"]) == 2  # caused_by and diagnosed_by
        assert "CauseB" in adj
        assert len(adj["CauseB"]) == 2  # fixed_by and caused_by

    def test_get_entity_vocab(self, sample_kg):
        """Test entity vocabulary extraction."""
        vocab = get_entity_vocab(sample_kg)

        assert len(vocab) == 6
        assert "SymptomA" in vocab
        assert "ToolE" in vocab

    def test_sample_path_length_1(self, sample_kg):
        """Test 1-hop path sampling."""
        path = sample_path(sample_kg, 1, seed=42)

        assert len(path["entities"]) == 2
        assert len(path["edges"]) == 1

    def test_sample_path_length_2(self, sample_kg):
        """Test 2-hop path sampling."""
        path = sample_path(sample_kg, 2, seed=42)

        assert len(path["entities"]) == 3
        assert len(path["edges"]) == 2

    def test_sample_path_no_cycles(self, sample_kg):
        """Test that sampled paths don't have cycles."""
        for seed in range(10):
            try:
                path = sample_path(sample_kg, 2, seed=seed)
                # All entities should be unique
                assert len(path["entities"]) == len(set(path["entities"]))
            except ValueError:
                pass  # Some seeds might not find valid paths

    def test_sample_path_with_start_types(self, sample_kg):
        """Test path sampling with specific start types."""
        path = sample_path(sample_kg, 1, seed=42, start_types=["symptom"])

        assert path["entities"][0] == "SymptomA"

    def test_get_neighbors(self, sample_kg):
        """Test neighbor discovery."""
        neighbors = get_neighbors(sample_kg, "SymptomA", max_hops=1)

        assert "CauseB" in neighbors
        assert "DiagD" in neighbors


class TestReward:
    """Tests for reward.py module."""

    def test_parse_completion_valid(self):
        """Test parsing valid completion."""
        completion = "TRACE: This is the reasoning.\nANSWER: B"
        valid, trace, answer = parse_completion(completion)

        assert valid is True
        assert "reasoning" in trace
        assert answer == "B"

    def test_parse_completion_invalid_no_trace(self):
        """Test parsing completion without TRACE."""
        completion = "ANSWER: B"
        valid, trace, answer = parse_completion(completion)

        assert valid is False

    def test_parse_completion_invalid_no_answer(self):
        """Test parsing completion without ANSWER."""
        completion = "TRACE: Some reasoning."
        valid, trace, answer = parse_completion(completion)

        assert valid is False

    def test_parse_completion_invalid_answer_value(self):
        """Test parsing completion with invalid answer value."""
        completion = "TRACE: Some reasoning.\nANSWER: X"
        valid, trace, answer = parse_completion(completion)

        assert valid is False

    def test_extract_entities(self):
        """Test entity extraction from text."""
        vocab = {"EntityA", "EntityB", "Other"}
        text = "EntityA is related to EntityB and also EntityA again."

        found = extract_entities(text, vocab)

        assert "EntityA" in found
        assert "EntityB" in found
        assert "Other" not in found

    def test_compute_reward_correct_with_path(self):
        """Test reward for correct answer with path mentions."""
        completion = "TRACE: SymptomA is caused by CauseB.\nANSWER: A"
        vocab = {"SymptomA", "CauseB", "FixC"}
        path_entities = ["SymptomA", "CauseB", "FixC"]

        reward = compute_reward(completion, "A", path_entities, vocab)

        assert reward["correctness"] == 1.0
        assert reward["path_coverage"] > 0
        assert reward["total"] > 0

    def test_compute_reward_wrong_answer(self):
        """Test reward for wrong answer."""
        completion = "TRACE: SymptomA is caused by CauseB.\nANSWER: B"
        vocab = {"SymptomA", "CauseB"}
        path_entities = ["SymptomA", "CauseB"]

        reward = compute_reward(completion, "A", path_entities, vocab)

        assert reward["correctness"] == -2.0
        assert reward["total"] < 0

    def test_compute_reward_invalid_format(self):
        """Test reward for invalid format."""
        completion = "This is not valid"
        vocab = {"SymptomA"}
        path_entities = ["SymptomA"]

        reward = compute_reward(completion, "A", path_entities, vocab)

        assert reward["total"] == -2.0
        assert reward["parsed"]["valid_format"] is False

    def test_compute_reward_spam_penalty(self):
        """Test spam penalty for repeated entities."""
        # Repeat entity more than 2 times
        completion = "TRACE: SymptomA SymptomA SymptomA SymptomA.\nANSWER: A"
        vocab = {"SymptomA", "CauseB"}
        path_entities = ["SymptomA", "CauseB"]

        reward = compute_reward(completion, "A", path_entities, vocab)

        assert reward["spam_penalty"] == 0.5

    def test_compute_reward_min_hits(self):
        """Test that path reward requires minimum hits."""
        # Only mention 1 entity (min_hits = 2 by default)
        completion = "TRACE: SymptomA is the issue.\nANSWER: A"
        vocab = {"SymptomA", "CauseB", "FixC"}
        path_entities = ["SymptomA", "CauseB", "FixC"]

        reward = compute_reward(completion, "A", path_entities, vocab)

        assert reward["path_reward"] == 0.0  # Not enough hits


class TestDataset:
    """Tests for dataset.py module."""

    def test_generate_distractors(self, sample_kg):
        """Test distractor generation."""
        distractors = generate_distractors(
            correct_answer="CauseB",
            kg=sample_kg,
            path_entities=["SymptomA", "CauseB"],
            n=3,
            seed=42
        )

        assert len(distractors) == 3
        assert "CauseB" not in distractors  # Correct answer excluded
        assert "SymptomA" not in distractors  # Path entities excluded

    def test_generate_mcq(self, sample_kg):
        """Test MCQ generation from path."""
        path = {
            "entities": ["SymptomA", "CauseB", "FixC"],
            "edges": [
                {"src": "SymptomA", "rel": "caused_by", "dst": "CauseB"},
                {"src": "CauseB", "rel": "fixed_by", "dst": "FixC"},
            ]
        }

        mcq = generate_mcq(path, sample_kg, "test_001", seed=42)

        assert "id" in mcq
        assert "question" in mcq
        assert "options" in mcq
        assert len(mcq["options"]) == 4
        assert mcq["answer_star"] in ["A", "B", "C", "D"]
        assert "prompt" in mcq
        assert "ref" in mcq

    def test_build_prompt(self):
        """Test prompt building."""
        question = "What causes SymptomA?"
        options = {"A": "CauseA", "B": "CauseB", "C": "CauseC", "D": "CauseD"}

        prompt = build_prompt(question, options)

        assert "OUTPUT FORMAT:" in prompt
        assert "TRACE:" in prompt
        assert "ANSWER:" in prompt
        assert "What causes SymptomA?" in prompt
        assert "A) CauseA" in prompt

    def test_generate_dataset(self, sample_kg):
        """Test full dataset generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg_path = Path(tmpdir) / "kg.json"
            output_path = Path(tmpdir) / "test.jsonl"

            # Write sample KG
            with open(kg_path, "w") as f:
                json.dump(sample_kg, f)

            # Generate dataset
            generate_dataset(
                kg_path=kg_path,
                output_path=output_path,
                n_examples=5,
                hop_lengths=[1, 2],
                split="test",
                seed=42
            )

            # Verify output
            with open(output_path) as f:
                examples = [json.loads(line) for line in f]

            assert len(examples) > 0
            for ex in examples:
                assert ex["split"] == "test"
                assert ex["hop_len"] in [1, 2]


class TestIntegration:
    """Integration tests using the real KG."""

    def test_real_kg_stats(self, real_kg):
        """Test real KG has expected structure."""
        assert len(real_kg["entities"]) >= 200
        assert len(real_kg["edges"]) >= 600

    def test_real_kg_path_sampling(self, real_kg):
        """Test path sampling on real KG."""
        for length in [1, 2, 3, 4, 5]:
            path = sample_path(real_kg, length, seed=42 + length)
            assert len(path["entities"]) == length + 1
            assert len(path["edges"]) == length

    def test_full_pipeline(self, real_kg):
        """Test full MCQ generation pipeline."""
        # Sample path
        path = sample_path(real_kg, 3, seed=42, start_types=["symptom"])

        # Generate MCQ
        mcq = generate_mcq(path, real_kg, "test_full", seed=42)

        # Simulate correct response
        ref_trace = mcq["ref"]["trace"]

        # Compute reward
        vocab = get_entity_vocab(real_kg)
        reward = compute_reward(
            completion=ref_trace,
            answer_star=mcq["answer_star"],
            path_entities=mcq["path_star"]["entities"],
            entity_vocab=vocab
        )

        # Reference trace should get good reward
        assert reward["correctness"] == 1.0
        assert reward["path_coverage"] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
