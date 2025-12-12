"""
Direct Code Generator
Generates code without any debate or Socratic method
"""
import os
import time
import requests
import json
import re
import warnings
from typing import Tuple
from dotenv import load_dotenv

warnings.filterwarnings('ignore', message='Unverified HTTPS request')
load_dotenv()

class DirectCodeGenerator:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        print(f"🤖 Direct Generator initialized")
        
    def _clean_generated_code(self, code: str) -> str:
        """Clean common issues in LLM-generated code"""
        # Remove markdown code blocks
        code = re.sub(r'^```python\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        code = code.replace("```python", "").replace("```", "")
        
        # Remove common preambles
        lines = code.split('\n')
        cleaned_lines = []
        
        skip_phrases = [
            "here's", "here is", "i'll", "let me", "this code",
            "the following", "below is", "this implementation"
        ]
        
        for line in lines:
            line_lower = line.lower().strip()
            if any(phrase in line_lower for phrase in skip_phrases) and not line.strip().startswith('#'):
                continue
            if not cleaned_lines and not line.strip():
                continue
            cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines).strip()
        
        # Remove trailing explanations
        last_def_index = -1
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                last_def_index = i
        
        if last_def_index > 0:
            for i in range(last_def_index + 1, len(lines)):
                if lines[i] and not lines[i][0].isspace() and not lines[i].startswith('#'):
                    if not lines[i].strip().startswith('def ') and not lines[i].strip().startswith('class '):
                        code = '\n'.join(lines[:i]).strip()
                        break
        
        return code
    
    def _make_api_call(self, prompt: str, max_tokens: int, temperature: float = 0.7) -> str:
        """Make API call with retry logic"""
        if not self.api_key:
            return "Error: GROQ_API_KEY not found"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        max_retries = 5
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=data, 
                    verify=False, 
                    timeout=90
                )
                
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        try:
                            error_data = response.json()
                            if 'message' in error_data:
                                match = re.search(r'try again in (\d+)ms', error_data['message'])
                                if match:
                                    wait_ms = int(match.group(1))
                                    wait_time = (wait_ms / 1000.0) + 1.0
                                else:
                                    wait_time = base_delay * (2 ** attempt)
                            else:
                                wait_time = base_delay * (2 ** attempt)
                        except:
                            wait_time = base_delay * (2 ** attempt)
                        
                        print(f"  ⏳ Rate limit hit (attempt {attempt+1}/{max_retries}), waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("  ❌ Rate limit exceeded after all retries")
                        return "Error: Rate limit exceeded after retries"
                
                if response.status_code != 200:
                    error_msg = f"API Error {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg += f": {error_data['error']}"
                    except:
                        pass
                    print(f"  ⚠️  {error_msg}")
                    
                    if response.status_code >= 500 and attempt < max_retries - 1:
                        wait_time = base_delay * (attempt + 1)
                        print(f"  ⏳ Server error, retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    
                    return f"Error: {error_msg}"
                
                response_data = response.json()
                if "choices" not in response_data:
                    return "Error: Unexpected API response"
                
                return response_data["choices"][0]["message"]["content"].strip()
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (attempt + 1)
                    print(f"  ⏳ Timeout, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                return "Error: API timeout"
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (attempt + 1)
                    print(f"  ⏳ Error: {str(e)}, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                return f"Error: {str(e)}"
        
        return "Error: Max retries exceeded"
    
    def generate(self, problem: str) -> Tuple[str, float]:
        """Direct code generation"""
        print("\n🎯 Generating direct solution...")
        start = time.time()
        
        prompt = f"""You are an expert Python programmer. Write COMPLETE, EXECUTABLE Python code for this problem.

PROBLEM:
{problem}

CRITICAL REQUIREMENTS:
1. Write ONLY valid Python code - NO markdown, NO explanations, NO comments outside code
2. Include ALL necessary imports at the top
3. Every function MUST have a complete implementation - NO 'pass', NO '# TODO', NO placeholders
4. All classes must have __init__ and all required methods fully implemented
5. Code must be syntactically correct and immediately executable
6. Handle all edge cases with actual code, not comments
7. DO NOT include usage examples or test code - ONLY the class/function definitions

Begin your response with the first import or class definition. End with the last line of code.
NO text before or after the code.

CODE:"""
        
        code = self._make_api_call(prompt, max_tokens=2000, temperature=0.3)
        code = self._clean_generated_code(code)
        
        elapsed = time.time() - start
        print(f"  ✓ Direct generation completed ({elapsed:.1f}s)")
        
        return code, elapsed
    
    def save_result(self, code: str, filename: str = "direct_code.py"):
        """Save generated code to file"""
        with open(filename, "w", encoding='utf-8') as f:
            f.write("# DIRECT GENERATION CODE\n")
            f.write("# Generated without debate\n\n")
            f.write(code)
        print(f"  ✓ Saved to {filename}")


if __name__ == "__main__":
    generator = DirectCodeGenerator()
    
    problem = """
Implement a thread-safe LRU (Least Recently Used) cache with TTL (Time To Live).

Requirements:
- Support get(key) and put(key, value) operations
- Maximum capacity that evicts least recently used items when full
- Each entry has a TTL; expired entries should not be returned
- Must be thread-safe for concurrent access
- O(1) time complexity for both operations
"""
    
    code, timing = generator.generate(problem)
    
    print("\n" + "="*70)
    print("GENERATED CODE:")
    print("="*70)
    print(code[:500] + "..." if len(code) > 500 else code)
    print(f"\nTime taken: {timing:.2f}s")
    
    generator.save_result(code)