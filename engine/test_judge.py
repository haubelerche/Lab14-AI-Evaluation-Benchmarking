import asyncio
import os
from engine.llm_judge import LLMJudge

async def main():
    judge = LLMJudge()
    
    question = "Làm thế nào để đo lường hiệu quả của một hệ thống RAG?"
    ground_truth = "Sử dụng các chỉ số như Hit Rate, MRR, và các khung đánh giá như RAGAS (Faithfulness, Relevancy)."
    
    print("--- TEST 1: Câu trả lời chính xác ---")
    correct_answer = "Chúng ta có thể đo lường RAG qua Hit Rate, MRR và framework RAGAS với các tiêu chí như độ trung thực và tính liên quan."
    result = await judge.evaluate_multi_judge(question, correct_answer, ground_truth)
    print(f"Final Score: {result['final_score']}")
    print(f"Agreement: {result['agreement_rate']}")
    print(f"GPT Reasoning: {result['individual_judgments']['gpt']['reasoning']}")
    print(f"Gemini Reasoning: {result['individual_judgments']['gemini']['reasoning']}")

    print("\n--- TEST 2: Câu trả lời sai (Hallucination) ---")
    hallucinated_answer = "Chúng ta đo lường RAG bằng cách đếm số lượng file PDF và kiểm tra màu sắc của giao diện người dùng."
    result = await judge.evaluate_multi_judge(question, hallucinated_answer, ground_truth)
    print(f"Final Score: {result['final_score']}")
    print(f"Agreement: {result['agreement_rate']}")
    print(f"GPT Reasoning: {result['individual_judgments']['gpt']['reasoning']}")
    print(f"Gemini Reasoning: {result['individual_judgments']['gemini']['reasoning']}")

    print("\n--- TEST 3: Swap Test (Position Bias) ---")
    bias_result = await judge.check_position_bias(question, correct_answer, hallucinated_answer, ground_truth)
    print(f"Bias detected: {bias_result['bias_detected']}")

if __name__ == "__main__":
    asyncio.run(main())
