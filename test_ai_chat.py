"""
Test script to verify AI Chat provides direct answers
"""
import pandas as pd
from dotenv import load_dotenv
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.ai_chat import AIChat

# Load environment
load_dotenv()

# Create sample dataset (similar to user's car dataset)
df = pd.DataFrame({
    'Brand': ['BMW', 'Toyota', 'BMW', 'Honda', 'BMW', 'Toyota', 'Ford'],
    'Price': [50000, 25000, 55000, 30000, 48000, 27000, 35000],
    'Year': [2020, 2019, 2021, 2020, 2019, 2020, 2021]
})

print("=" * 60)
print("TESTING AI CHAT - Direct Answer Feature")
print("=" * 60)
print(f"\nSample Dataset:\n{df}\n")

# Initialize AI Chat
ai_chat = AIChat()

if ai_chat.model:
    print("✓ Gemini AI is configured and ready\n")
    
    # Test questions
    test_questions = [
        "How many BMW cars are in the dataset?",
        "What is the average price?",
        "Which brand appears most frequently?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        print("-" * 60)
        
        response = ai_chat.query_data(df, question)
        
        print(f"Answer: {response.get('answer', 'No answer')}")
        if response.get('code'):
            print(f"Code executed: {response['code']}")
        if response.get('data'):
            print(f"Result: {response['data']}")
        print()
else:
    print("❌ Gemini AI is NOT configured")
    print("Check your .env file and API key")

print("=" * 60)
print("Test Complete!")
print("=" * 60)
