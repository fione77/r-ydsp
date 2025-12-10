import os
import time
import requests
from dotenv import load_dotenv

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
            response = requests.post(self.api_url, headers=headers, json=data, verify=False)
            response.raise_for_status()
            
            result = response.json()["choices"][0]["message"]["content"]
            
            # Extract just the Python code
            if "```python" in result:
                code = result.split("```python")[1].split("```")[0].strip()
            elif "```" in result:
                code = result.split("```")[1].split("```")[0].strip()
            else:
                code = result
            
            return code
            
        except Exception as e:
            return f"Error: {str(e)}"

class DirectCoder:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def generate_code(self, problem: str) -> str:
        """Direct code generation without debate"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": f"Write Python code: {problem}"}],
            "max_tokens": 300,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, verify=False)
            response.raise_for_status()
            
            result = response.json()["choices"][0]["message"]["content"]
            
            # Extract code if in block
            if "```python" in result:
                code = result.split("```python")[1].split("```")[0].strip()
            elif "```" in result:
                code = result.split("```")[1].split("```")[0].strip()
            else:
                code = result
            
            return code
            
        except Exception as e:
            return f"Error: {str(e)}"

# Compare
problem = "Write a function to safely divide two numbers"

print("🔍 COMPARISON: Socratic vs Direct Generation")
print("=" * 60)

# Socratic
print("\n🧠 SOCRATIC APPROACH:")
socratic_coder = SocraticCoder()
start = time.time()
socratic_code = socratic_coder.generate_code(problem)
socratic_time = time.time() - start
print(socratic_code)
print(f"\n⏱️ Time: {socratic_time:.2f}s")

# Check for quality indicators
socratic_has_error = 'try' in socratic_code.lower() or 'except' in socratic_code.lower() or 'if' in socratic_code
socratic_has_comments = '#' in socratic_code
socratic_lines = len(socratic_code.split('\n'))

# Direct
print("\n" + "=" * 60)
print("🤖 DIRECT APPROACH:")
direct_coder = DirectCoder()
start = time.time()
direct_code = direct_coder.generate_code(problem)
direct_time = time.time() - start
print(direct_code)
print(f"\n⏱️ Time: {direct_time:.2f}s")

# Check for quality indicators
direct_has_error = 'try' in direct_code.lower() or 'except' in direct_code.lower() or 'if' in direct_code
direct_has_comments = '#' in direct_code
direct_lines = len(direct_code.split('\n'))

print("\n" + "=" * 60)
print("📊 COMPARISON SUMMARY:")
print("-" * 60)
print(f"{'Metric':<20} {'Socratic':<15} {'Direct':<15} {'Winner':<10}")
print("-" * 60)
print(f"{'Time (seconds)':<20} {socratic_time:.2f}s{'':<5} {direct_time:.2f}s{'':<5} {'Faster' if direct_time < socratic_time else 'Socratic'}")
print(f"{'Error Handling':<20} {str(socratic_has_error):<15} {str(direct_has_error):<15} {'✓' if socratic_has_error and not direct_has_error else '✗'}")
print(f"{'Has Comments':<20} {str(socratic_has_comments):<15} {str(direct_has_comments):<15} {'✓' if socratic_has_comments and not direct_has_comments else '✗'}")
print(f"{'Code Lines':<20} {socratic_lines:<15} {direct_lines:<15} {'Shorter' if direct_lines < socratic_lines else 'Longer'}")
print("-" * 60)

# Calculate Socratic advantage
advantage_score = 0
if socratic_has_error and not direct_has_error:
    advantage_score += 2
if socratic_has_comments and not direct_has_comments:
    advantage_score += 1
if 'edge' in socratic_code.lower() or 'case' in socratic_code.lower():
    advantage_score += 1

print(f"\n🎯 SOCRATIC ADVANTAGE SCORE: {advantage_score}/4")
if advantage_score >= 2:
    print("✅ Socratic produces higher quality code!")
elif advantage_score == 1:
    print("⚠️ Socratic has slight advantages")
else:
    print("❌ Direct generation is comparable or better")

print("\n💡 Insights:")
if socratic_has_error and not direct_has_error:
    print("  • Socratic considered error handling, Direct didn't")
if socratic_has_comments and not direct_has_comments:
    print("  • Socratic added documentation, Direct didn't")
if 'edge' in socratic_code.lower() or 'case' in socratic_code.lower():
    print("  • Socratic discussed edge cases explicitly")