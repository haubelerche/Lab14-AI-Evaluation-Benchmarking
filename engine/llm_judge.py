import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    def __init__(self, gpt_model: str = "gpt-4o-mini", gemini_model: str = "gemini-1.5-flash"):
        self.gpt_model = gpt_model
        # Lưu ý: Một số SDK yêu cầu prefix 'models/'
        self.gemini_model = gemini_model if gemini_model.startswith("models/") else f"models/{gemini_model}"
        
        # Cấu hình API Clients
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.genai_model = genai.GenerativeModel(gemini_model)
        
        # Định nghĩa rubrics chi tiết
        self.rubrics = {
            "accuracy": {
                "1": "Hoàn toàn sai hoặc không liên quan đến Ground Truth.",
                "2": "Có một vài ý đúng nhưng phần lớn là sai hoặc thiếu sót nghiêm trọng.",
                "3": "Đúng khoảng 50-70%, thiếu một vài chi tiết quan trọng.",
                "4": "Rất chính xác, chỉ thiếu sót những ý cực nhỏ không đáng kể.",
                "5": "Hoàn hảo, chính xác tuyệt đối và đầy đủ so với Ground Truth."
            },
            "tone": {
                "1": "Không chuyên nghiệp, ngôn ngữ không phù hợp.",
                "3": "Bình thường, đủ lịch sự nhưng chưa chuyên nghiệp.",
                "5": "Cực kỳ chuyên nghiệp, lịch sự và phù hợp với ngữ cảnh doanh nghiệp."
            }
        }

    def _build_judge_prompt(self, question: str, answer: str, ground_truth: str) -> str:
        return f"""
Bạn là một chuyên gia đánh giá chất lượng AI Agent (AI Judge). 
Nhiệm vụ của bạn là chấm điểm câu trả lời của Agent dựa trên câu trả lời mẫu (Ground Truth).

[DỮ LIỆU ĐÁNH GIÁ]
- Câu hỏi: {question}
- Ground Truth: {ground_truth}
- Câu trả lời của Agent: {answer}

[TIÊU CHÍ CHẤM ĐIỂM]
1. Accuracy (Độ chính xác):
{json.dumps(self.rubrics['accuracy'], ensure_ascii=False, indent=2)}

2. Tone (Ngôn ngữ):
{json.dumps(self.rubrics['tone'], ensure_ascii=False, indent=2)}

[YÊU CẦU ĐẦU RA]
Trả về kết quả dưới định dạng JSON duy nhất, có cấu trúc như sau:
{{
  "accuracy_score": <int 1-5>,
  "tone_score": <int 1-5>,
  "reasoning": "<Giải thích ngắn gọn lý do tại sao cho điểm như vậy>"
}}
"""

    async def _call_gpt_judge(self, prompt: str) -> Dict[str, Any]:
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": "You are a precise AI Quality Evaluator. Always output JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error calling GPT Judge: {e}")
            return {"accuracy_score": 1, "tone_score": 1, "reasoning": f"Error: {e}"}

    async def _call_gemini_judge(self, prompt: str) -> Dict[str, Any]:
        try:
            # Thử gọi với retry nhẹ hoặc model khác nếu cần
            response = await self.genai_model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Error calling Gemini Judge ({self.gemini_model}): {e}")
            # Fallback sang GPT-4o-mini để hệ thống không bị chết hoàn toàn
            print("Falling back to GPT for second judgment...")
            return await self._call_gpt_judge(prompt)

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        EXPERT TASK: Gọi 2 model (GPT & Gemini), tính toán trung bình và độ đồng thuận.
        """
        prompt = self._build_judge_prompt(question, answer, ground_truth)
        
        # Chạy song song 2 Judge
        res_gpt, res_gemini = await asyncio.gather(
            self._call_gpt_judge(prompt),
            self._call_gemini_judge(prompt)
        )
        
        score_a = res_gpt.get("accuracy_score", 1)
        score_b = res_gemini.get("accuracy_score", 1)
        
        # Tính toán final score (ưu tiên Accuracy)
        avg_accuracy = (score_a + score_b) / 2
        avg_tone = (res_gpt.get("tone_score", 1) + res_gemini.get("tone_score", 1)) / 2
        
        # Tính độ đồng thuận (Agreement Rate)
        diff = abs(score_a - score_b)
        if diff == 0:
            agreement = 1.0
        elif diff == 1:
            agreement = 0.5
        else:
            agreement = 0.0 # Bất đồng lớn
            
        return {
            "final_score": avg_accuracy,
            "tone_score": avg_tone,
            "agreement_rate": agreement,
            "individual_judgments": {
                "gpt": res_gpt,
                "gemini": res_gemini
            }
        }

    async def check_position_bias(self, question: str, response_a: str, response_b: str, ground_truth: str):
        """
        Nâng cao: Kiểm tra xem Judge có thiên vị vị trí khi so sánh 2 câu trả lời không.
        Đây là kỹ thuật Swap Test.
        """
        prompt_1 = f"So sánh Response A: {response_a} và Response B: {response_b} dựa trên Ground Truth: {ground_truth}. Câu nào tốt hơn?"
        prompt_2 = f"So sánh Response A: {response_b} và Response B: {response_a} dựa trên Ground Truth: {ground_truth}. Câu nào tốt hơn?"
        
        # Chạy cả 2 prompt và xem kết quả có bị đảo ngược tương ứng không
        # Đây là phần mở rộng, trả về kết quả phân tích bias
        res1 = await self._call_gpt_judge(prompt_1)
        res2 = await self._call_gpt_judge(prompt_2)
        
        return {"original": res1, "swapped": res2, "bias_detected": res1.get("accuracy_score") != res2.get("accuracy_score")}
