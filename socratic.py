import os
import ssl
import urllib3
import requests
from dotenv import load_dotenv

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create custom SSL context
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

load_dotenv()

class SocraticCoder:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
    def generate_code(self, problem: str) -> str:
        """Generate code using Socratic dialogue"""
        
        prompt = f"""Three programmers debate then write code:

Problem: {problem}

ARCHITECT: "Let's use a simple solution..."
TESTER: "But what about edge cases like..."
OPTIMIZER: "Maybe we should consider..."
ARCHITECT: "OK, revised approach..."

Based on this discussion, write Python code:

```python
"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        try:
            # Use verify=False to bypass SSL certificate check
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=data, 
                verify=False,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                
                # Extract just the Python code
                if "```python" in result:
                    code = result.split("```python")[1].split("```")[0].strip()
                elif "```" in result:
                    code = result.split("```")[1].split("```")[0].strip()
                else:
                    code = result
                
                return code
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Connection Error: {str(e)}"

# Test it
if __name__ == "__main__":
    coder = SocraticCoder()
    test_problem = "Write a function to reverse a string"
    print("Testing Socratic coder with Groq...")
    print(f"Problem: {test_problem}")
    print("\nGenerated Code:")
    result = coder.generate_code(test_problem)
    print(result)