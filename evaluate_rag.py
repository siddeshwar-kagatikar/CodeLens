import requests
import pandas as pd
from datasets import Dataset
from ragas import evaluate
import time  # <-- NEW: Adding time for our pause

# FIXED: Updated the import paths to fix the DeprecationWarnings
# Replace the old lowercase imports with these:
from ragas.metrics import ContextPrecision, Faithfulness, AnswerRelevancy
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)

test_suite = [
    {
        "question": "How is the video call functionality implemented in the interview module? look into the code and explain the libraries and technologies used.",
        "ground_truth": "The video call functionality is implemented using the ZegoUIKitPrebuilt library. It generates a kitToken using an APP_ID and SECRET for authentication, and dynamically supports both One-on-One and Group calls based on URL parameters."
    }
]

def run_evaluation():
    print("Starting automated RAG evaluation...")
    results_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, test in enumerate(test_suite):
        print(f"\nAsking Question {i+1}/3: {test['question']}")
        
        response = requests.post(
            "http://127.0.0.1:8000/query",
            json={"question": test["question"]}
        )
        
        if response.status_code == 200:
            data = response.json()
            results_data["question"].append(test["question"])
            results_data["answer"].append(data["answer"])
            results_data["contexts"].append(data.get("sources", ["No sources retrieved"])) 
            results_data["ground_truth"].append(test["ground_truth"])
            print("Successfully retrieved answer!")
            print(f"Answer: {data['answer']}")
        else:
            print(f"Error querying API: {response.text}")
            
        # FIXED: Wait 15 seconds before the next question to reset the rate limit!
        if i < len(test_suite) - 1:
            print("Waiting 15 seconds to respect Gemini API limits...")
            time.sleep(5)

    # Make sure we actually have data before running the eval
    if not results_data["answer"]:
        print("\nEvaluation aborted: No answers were successfully retrieved from the API.")
        return

    dataset = Dataset.from_dict(results_data)

    print("\nGrading the answers using Ragas Triad... (This takes a moment)")
    
    evaluation_result = evaluate(
        dataset,
        metrics=[ContextPrecision(), Faithfulness()], 
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=1, max_retries=3) 
    )

    print("\n=== FINAL RAG METRICS ===")
    df = evaluation_result.to_pandas()
    
    # Just print all columns so we don't get missing key errors!
    print(df) 
    
    print("\n--- RESUME BULLET POINT ---")
    # Wrap in a try-except just in case Ragas capitalized the column names
    try:
        avg_precision = df['context_precision'].mean() * 100
        avg_faithfulness = df['faithfulness'].mean() * 100
    except KeyError:
        avg_precision = df['ContextPrecision'].mean() * 100
        avg_faithfulness = df['Faithfulness'].mean() * 100
        
    print(f"\"Architected a codebase RAG assistant for a CodeHive MERN-stack platform, utilizing AST-aware chunking to achieve {avg_precision:.1f}% context retrieval precision and {avg_faithfulness:.1f}% response faithfulness, validated via automated Ragas pipelines.\"")

if __name__ == "__main__":
    run_evaluation()