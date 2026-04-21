from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        TODO: Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        TODO: Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Chạy eval cho toàn bộ bộ dữ liệu.
        Dataset cần có trường 'expected_retrieval_ids' và Agent trả về 'retrieved_ids'.
        """
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}

        hit_rates = []
        mrr_scores = []

        for item in dataset:
            expected_ids = (
                item.get("expected_ids")
                or item.get("ground_truth_ids")
                or item.get("expected_retrieval_ids")
                or []
            )
            retrieved_ids = item.get("retrieved_ids", [])

            hit_rates.append(self.calculate_hit_rate(expected_ids, retrieved_ids))
            mrr_scores.append(self.calculate_mrr(expected_ids, retrieved_ids))

        return {
            "avg_hit_rate": sum(hit_rates) / len(hit_rates),
            "avg_mrr": sum(mrr_scores) / len(mrr_scores),
        }


if __name__ == "__main__":
    import asyncio

    evaluator = RetrievalEvaluator()

    # --- Test calculate_hit_rate ---
    assert evaluator.calculate_hit_rate(["doc1"], ["doc1", "doc2", "doc3"]) == 1.0, "FAIL: hit case"
    assert evaluator.calculate_hit_rate(["doc1"], ["doc4", "doc5", "doc6"]) == 0.0, "FAIL: miss case"
    assert evaluator.calculate_hit_rate(["doc1"], ["doc2", "doc1"], top_k=1) == 0.0, "FAIL: ngoài top_k"
    assert evaluator.calculate_hit_rate(["doc1", "doc2"], ["doc3", "doc2"]) == 1.0, "FAIL: multi expected"
    print("calculate_hit_rate: OK")

    # --- Test calculate_mrr ---
    assert evaluator.calculate_mrr(["doc1"], ["doc1", "doc2"]) == 1.0,        "FAIL: rank 1"
    assert evaluator.calculate_mrr(["doc1"], ["doc2", "doc1"]) == 0.5,        "FAIL: rank 2"
    assert abs(evaluator.calculate_mrr(["doc1"], ["doc2", "doc3", "doc1"]) - 1/3) < 1e-9, "FAIL: rank 3"
    assert evaluator.calculate_mrr(["doc1"], ["doc3", "doc4"]) == 0.0,        "FAIL: không có"
    print("calculate_mrr: OK")

    # --- Test evaluate_batch ---
    dataset = [
        {"expected_ids": ["doc1"], "retrieved_ids": ["doc1", "doc2", "doc3"]},  # hit, mrr=1.0
        {"expected_ids": ["doc5"], "retrieved_ids": ["doc2", "doc5", "doc3"]},  # hit, mrr=0.5
        {"expected_ids": ["doc9"], "retrieved_ids": ["doc1", "doc2", "doc3"]},  # miss, mrr=0.0
    ]
    result = asyncio.run(evaluator.evaluate_batch(dataset))
    print(f"evaluate_batch result: {result}")
    assert abs(result["avg_hit_rate"] - 2/3) < 1e-9, "FAIL: avg_hit_rate"
    assert abs(result["avg_mrr"] - 0.5) < 1e-9,      "FAIL: avg_mrr"
    print("evaluate_batch: OK")

    # --- Test dataset rỗng ---
    empty_result = asyncio.run(evaluator.evaluate_batch([]))
    assert empty_result == {"avg_hit_rate": 0.0, "avg_mrr": 0.0}, "FAIL: empty dataset"
    print("evaluate_batch (empty): OK")

    print("\nAll tests passed!")
