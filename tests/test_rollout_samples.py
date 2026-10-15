"""rollout.sample_records keeps a bounded, readable record of evaluation
completions: one dict per kept row with the step, the prompt's question, the
completion text, its reward and length, in row order and at most `limit`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks import rollout


def main():
    inputs = [{"prompt": [{"role": "user", "content": f"question {i}"}], "answer": str(i), "example_id": i // 2}
              for i in range(4)]
    texts = [f"completion {i}" for i in range(4)]
    rewards = [1.0, 0.0, 0.0, 1.0]
    lengths = [10, 20, 30, 40]
    records = rollout.sample_records(7, inputs, texts, rewards, lengths, limit=3)
    assert len(records) == 3
    assert records[0] == {"step": 7, "example_id": 0, "question": "question 0", "answer": "0",
                          "completion": "completion 0", "reward": 1.0, "length": 10}, records[0]
    assert [r["example_id"] for r in records] == [0, 0, 1]
    assert rollout.sample_records(7, inputs, texts, rewards, lengths, limit=0) == []
    # A plain-string prompt is kept as the question.
    plain = rollout.sample_records(1, [{"prompt": "2 + 2", "answer": "4"}], ["4"], [1.0], [1], limit=5)
    assert plain[0]["question"] == "2 + 2" and plain[0]["example_id"] is None
    print("ok: rollout.sample_records")


if __name__ == "__main__":
    main()
