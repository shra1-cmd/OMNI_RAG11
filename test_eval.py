import pandas as pd
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# 1. Load your Golden Dataset
df = pd.read_csv("eval_dataset.csv")

results = []

print("🚀 Starting RAG Evaluation...")

for index, row in df.iterrows():
    # --- SIMULATED RAG STEPS ---
    # In a real test, you would call: 
    # response, context = agent.run(row['question'])
    
    # For now, let's assume these are your system's outputs:
    actual_output = "The system uses BGE embeddings." 
    retrieved_contexts = ["BGE-base-en is the primary embedding model."]

    # 2. Setup the Metric
    metric = FaithfulnessMetric(threshold=0.7)
    test_case = LLMTestCase(
        input=row['question'],
        actual_output=actual_output,
        retrieval_context=retrieved_contexts
    )

    # 3. Measure
    metric.measure(test_case)
    
    results.append({
        "question": row['question'],
        "score": metric.score,
        "reason": metric.reason
    })
    print(f"Checked Q{index+1}: Score {metric.score}")

# 4. Save Results
pd.DataFrame(results).to_csv("evaluation_results.csv", index=False)
print("✅ Evaluation complete! Results saved to evaluation_results.csv")