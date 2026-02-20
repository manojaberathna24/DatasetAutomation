import re
import os

file_agent = 'pages/7_🎯_AI_Agent.py'
text_agent = ""

if os.path.exists(file_agent):
    with open(file_agent, 'r', encoding='utf-8') as f:
        text_agent = f.read()

# Fix the method name mistake
if 'ai_chat.ask_question(df, prompt)' in text_agent:
    text_agent = text_agent.replace('ai_chat.ask_question(df, prompt)', 'ai_chat.query_data(df, prompt)')
    
    with open(file_agent, 'w', encoding='utf-8') as f:
        f.write(text_agent)
    print("Fixed AI Agent query method name to query_data()")
else:
    print("Method name is already correct or not found.")
