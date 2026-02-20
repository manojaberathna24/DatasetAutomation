"""
AI chat integration using Google Gemini
"""
import pandas as pd
import streamlit as st
from config import GEMINI_API_KEY, OPENROUTER_API_KEY, OPENROUTER_MODEL
import json
import re

# Optional Gemini AI import
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None

# Optional OpenRouter AI import
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

class AIChat:
    """AI-powered chat for data analysis"""
    
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.openrouter_key = OPENROUTER_API_KEY
        self.model = None
        self.chat = None
        self.openrouter_client = None
        
        if GENAI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-flash-latest')
                self.chat = self.model.start_chat(history=[])
            except Exception as e:
                st.warning(f"⚠️ Gemini AI not configured: {str(e)}")
                
        if OPENAI_AVAILABLE and self.openrouter_key:
            try:
                self.openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.openrouter_key,
                )
            except Exception as e:
                st.warning(f"⚠️ OpenRouter AI not configured: {str(e)}")
    
    def _generate_content(self, prompt):
        """Generate content using Gemini or fallback to OpenRouter"""
        if self.model:
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                # If Gemini fails and OpenRouter is available, fallback
                if self.openrouter_client:
                    response = self.openrouter_client.chat.completions.create(
                        model=OPENROUTER_MODEL,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.choices[0].message.content.strip()
                raise e
        elif self.openrouter_client:
            response = self.openrouter_client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        else:
            raise Exception("No AI provider configured")
    
    def query_data(self, df, user_question):
        """
        Process natural language query about the data
        
        Args:
            df: pandas DataFrame
            user_question: User's question in natural language
            
        Returns:
            dict with answer, code, and chart_type
        """
        if not self.model and not self.openrouter_client:
            return self._fallback_query(df, user_question)
        
        try:
            # Create context about the dataset
            context = self._create_dataset_context(df)
            
            # Prepare prompt - emphasize providing DIRECT answers
            prompt = f"""
You are a helpful data analysis assistant. Given the following dataset information:

{context}

User question: {user_question}

IMPORTANT: Provide a DIRECT, COMPLETE answer to the question. Do NOT ask the user to execute code.
If you need to calculate something, provide the Python pandas code, and I will execute it for you.

Respond with a JSON object containing:
1. "answer": A direct, conversational answer (e.g., "BMW appears 45 times in the Brand column")
2. "code": Python pandas code to calculate the answer (store result in 'result' variable)
3. "chart_type": Suggested chart type (bar/pie/line/scatter/none)

For the code:
- Store the final result in a variable named 'result'
- Keep it simple and clean
- Use df as the dataframe variable

Return only valid JSON, no extra text.
"""
            
            response_text = self._generate_content(prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Execute code if provided and update answer with actual result
                if result.get('code'):
                    try:
                        exec_result = self._safe_execute_code(df, result['code'])
                        if exec_result is not None:
                            result['data'] = exec_result
                            
                            # Ask AI to reformulate answer with the actual result
                            followup_prompt = f"""
The user asked: {user_question}

I executed this code:
{result['code']}

The result is: {exec_result}

Provide a clear, conversational answer incorporating this result. Be specific and direct.
Just return the answer text, no JSON.
"""
                            followup_response_text = self._generate_content(followup_prompt)
                            # Update answer with the executed result
                            result['answer'] = followup_response_text
                    except Exception as e:
                        result['error'] = str(e)
                        result['answer'] = f"I tried to calculate this but encountered an error: {str(e)}"
                
                return result
            else:
                return {"answer": response_text, "code": None, "chart_type": "none"}
                
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            return self._fallback_query(df, user_question)
    
    def get_business_insights(self, df):
        """
        Generate comprehensive business insights from data
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict with insights, recommendations, and risks
        """
        if not self.model and not self.openrouter_client:
            return self._fallback_insights(df)
        
        try:
            context = self._create_dataset_context(df)
            stats = self._get_quick_stats(df)
            
            prompt = f"""
Analyze this business dataset and provide strategic insights:

Dataset Info:
{context}

Key Statistics:
{stats}

Provide a comprehensive analysis in JSON format with:
1. "key_insights": List of 5 most important findings
2. "recommendations": List of 5 actionable business recommendations
3. "risks": List of 3 potential risks or warnings
4. "trends": List of notable trends in the data
5. "next_steps": List of recommended next actions

Return only valid JSON.
"""
            
            response_text = self._generate_content(prompt)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_insights(df)
                
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
            return self._fallback_insights(df)
    
    def _create_dataset_context(self, df):
        """Create a text description of the dataset"""
        context = f"""
Rows: {len(df)}
Columns: {len(df.columns)}
Column Names: {', '.join(df.columns.tolist())}

Column Details:
"""
        for col in df.columns[:10]:  # Limit to first 10 columns
            dtype = df[col].dtype
            unique = df[col].nunique()
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            
            context += f"- {col}: {dtype}, {unique} unique values, {null_pct:.1f}% missing\n"
            
            # Add sample values for categorical columns
            if unique < 10 and df[col].dtype == 'object':
                context += f"  Values: {', '.join(df[col].dropna().unique().astype(str)[:5].tolist())}\n"
        
        return context
    
    def _get_quick_stats(self, df):
        """Get quick statistics about the data"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            return "No numeric columns in dataset"
        
        stats = "Numeric Summary:\n"
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            stats += f"- {col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, std={df[col].std():.2f}\n"
        
        return stats
    
    def _safe_execute_code(self, df, code):
        """Safely execute pandas code"""
        try:
            # Create a safe execution environment
            local_vars = {'df': df, 'pd': pd}
            exec(code, {}, local_vars)
            
            # Return result if available
            if 'result' in local_vars:
                return local_vars['result']
            
            return None
        except Exception as e:
            raise Exception(f"Code execution error: {str(e)}")
    
    def _fallback_query(self, df, question):
        """Fallback responses when AI is not available"""
        question_lower = question.lower()
        
        # Simple pattern matching
        if 'how many' in question_lower or 'count' in question_lower:
            return {
                "answer": f"The dataset contains {len(df)} rows and {len(df.columns)} columns.",
                "code": "len(df)",
                "chart_type": "none"
            }
        elif 'columns' in question_lower or 'features' in question_lower:
            return {
                "answer": f"Columns: {', '.join(df.columns.tolist())}",
                "code": "df.columns.tolist()",
                "chart_type": "none"
            }
        elif 'mean' in question_lower or 'average' in question_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                return {
                    "answer": f"Average {col}: {df[col].mean():.2f}",
                    "code": f"df['{col}'].mean()",
                    "chart_type": "bar"
                }
        
        return {
            "answer": "AI chat is not configured. Please set GEMINI_API_KEY or OPENROUTER_API_KEY in your .env file.",
            "code": None,
            "chart_type": "none"
        }
    
    def _fallback_insights(self, df):
        """Fallback insights when AI is not available"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        insights = {
            "key_insights": [
                f"Dataset contains {len(df)} records and {len(df.columns)} features",
                f"There are {len(numeric_cols)} numeric columns",
                f"Missing values found in {df.isnull().sum().sum()} cells",
                "AI insights require Gemini or OpenRouter API key configuration",
                "Configure GEMINI_API_KEY or OPENROUTER_API_KEY in .env for detailed analysis"
            ],
            "recommendations": [
                "Set up API keys for AI-powered insights",
                "Review missing values and data quality",
                "Explore visualizations to understand patterns",
                "Consider data cleaning before analysis",
                "Use AutoML feature to build predictive models"
            ],
            "risks": [
                "AI analysis not available without API keys",
                "Manual review recommended for data quality",
                "Consider data privacy when using cloud AI"
            ],
            "trends": [
                "Basic statistical analysis available",
                "Visualization features fully functional",
                "ML capabilities operational"
            ],
            "next_steps": [
                "Configure API keys",
                "Clean and prepare data",
                "Build ML models",
                "Generate comprehensive report"
            ]
        }
        
        return insights
