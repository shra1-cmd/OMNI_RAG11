from agents.runner import run_agent

print("\n=== OmniRAG Terminal Test ===")
print("Type 'exit' to quit\n")

while True:
    q = input("You: ").strip()
    if q.lower() == "exit":
        print("\nExiting OmniRAG.")
        break

    answer = run_agent(q)

    print("\n--- FINAL ANSWER ---")
    print(answer)
    print("--------------------\n")
