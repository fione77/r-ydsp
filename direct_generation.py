"""
Direct Code Generator - FOR LEETCODE HARD PROBLEMS - OPENROUTER VERSION
Generates code without any debate or Socratic method
"""
import os
import time
import requests
import re
import warnings
import ast
from typing import Tuple
from dotenv import load_dotenv

warnings.filterwarnings('ignore', message='Unverified HTTPS request')
load_dotenv()

class DirectCodeGenerator:
    def __init__(self):
        # CHANGE 1: Switch from GROQ_API_KEY to OPENROUTER_API_KEY
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        # CHANGE 2: Switch to OpenRouter API endpoint
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        print(f"🤖 Direct Generator initialized (OpenRouter)")
        
        # Test API connection
        if not self.api_key:
            print("  ⚠️  WARNING: OPENROUTER_API_KEY not found in .env file")
            print("  💡 Add to .env: OPENROUTER_API_KEY=your-key-here")
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean LLM-generated code - IMPROVED VERSION"""
        # Remove markdown code blocks
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # Split into lines
        lines = code.split('\n')
        cleaned_lines = []
        
        # Track if we're in actual code
        in_code_block = False
        found_first_code_line = False
        
        # Phrases that indicate non-code text (to be removed)
        explanatory_phrases = [
            "this code", "here's", "here is", "i'll", "let me", 
            "the following", "below is", "this implementation",
            "the code above", "in summary", "to summarize",
            "the solution", "this function", "the algorithm",
            "in conclusion", "in this code", "as shown above",
            "explanation:", "note:", "important:", "example:",
            "the output will be", "we can test", "here is how",
            "first, we", "then, we", "finally, we", "let's",
            "we'll", "we will", "we are going to", "approach:",
            "algorithm:", "implementation:", "solution:",
            "the function", "returns", "takes", "parameters:",
            "example usage:", "test case:", "for example,",
            "note that", "it is important", "make sure",
            "ensure that", "remember to", "keep in mind"
        ]
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines at the beginning
            if not found_first_code_line and not stripped:
                continue
                
            # Check if line looks like code
            is_code_like = (
                stripped.startswith(('#', 'import ', 'from ', 'def ', 'class ', '@')) or
                stripped.endswith(':') or
                ' = ' in line or
                '):' in line or
                re.match(r'^\s*(if |for |while |def |class |return |yield |raise |try:|except |finally:)', stripped)
            )
            
            # Check if line is explanatory text (not code)
            is_explanatory = any(phrase in stripped.lower() for phrase in explanatory_phrases)
            
            # Once we find real code, start collecting
            if is_code_like and not is_explanatory:
                found_first_code_line = True
                in_code_block = True
                cleaned_lines.append(line)
            elif in_code_block:
                # We're in a code block, keep adding lines
                cleaned_lines.append(line)
            elif not found_first_code_line and stripped:
                # Haven't found code yet, skip explanatory lines
                if not is_explanatory and len(stripped) < 100:
                    cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines).strip()
        
        # Remove trailing non-code text after the last function/class
        lines = code.split('\n')
        last_code_line_index = -1
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('def ', 'class ', '@')):
                last_code_line_index = i
        
        if last_code_line_index >= 0:
            # Find where actual code ends
            for i in range(last_code_line_index + 1, len(lines)):
                stripped = lines[i].strip()
                # Check if this looks like non-code after the main code
                if (stripped and 
                    not stripped.startswith('#') and 
                    not stripped.startswith('@') and
                    len(stripped) > 80 and
                    ' = ' not in stripped and
                    '):' not in stripped):
                    # Likely explanatory text, cut off here
                    code = '\n'.join(lines[:i])
                    break
        
        # Final cleanup: remove any lines that are pure English sentences
        final_lines = []
        for line in code.split('\n'):
            stripped = line.strip()
            if not stripped:
                final_lines.append(line)
                continue
                
            # Skip lines that look like paragraphs of text
            words = stripped.split()
            if (len(words) > 6 and 
                not stripped.startswith('#') and
                not '=' in stripped and
                not ':' in stripped and
                not stripped.endswith(':')):
                # Looks like a sentence, not code
                # But allow short comments
                if not stripped.startswith('#') or len(stripped) > 50:
                    continue
                    
            final_lines.append(line)
        
        result = '\n'.join(final_lines).strip()
        
        # If result is empty or too short, return original cleaned version
        if len(result) < 50:
            return code
        
        return result
    
    def _validate_code_syntax(self, code: str) -> bool:
        """Quick syntax validation"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _retry_with_stricter_prompt(self, problem: str, attempt: int) -> str:
        """Retry code generation with stricter prompt"""
        stricter_prompt = f"""Write Python code to solve this LeetCode problem:

{problem}

IMPORTANT: Provide ONLY the Python code. NO explanations, NO comments outside code blocks, 
NO text before or after the code. The code must be syntactically correct and complete.

Start with imports or function/class definition. End with the last line of code.
DO NOT add any explanatory text.

Code:"""
        
        return self._make_api_call(stricter_prompt, max_tokens=2500, temperature=0.2)
    
    def _make_api_call(self, prompt: str, max_tokens: int, temperature: float = 0.7) -> str:
        """Make API call with retry logic"""
        # CHANGE 3: Check for OpenRouter API key
        if not self.api_key:
            print("    ❌ ERROR: OPENROUTER_API_KEY not found")
            return "# Error: OpenRouter API key not configured\n# Please add OPENROUTER_API_KEY to .env file\n\ndef solution():\n    return 0"
        
        # CHANGE 4: OpenRouter requires additional headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",  # Required by OpenRouter
            "X-Title": "Direct Code Generator"         # Required by OpenRouter
        }
        
        # CHANGE 5: Use Llama 3.3 70B on OpenRouter (free tier)
        data = {
            "model": "meta-llama/llama-3.3-70b-instruct:free",  # Llama 3.3 70B free tier
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95
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
                        wait_time = base_delay * (2 ** attempt)
                        print(f"  ⏳ Rate limit hit (attempt {attempt+1}/{max_retries}), waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("  ❌ Rate limit exceeded after all retries")
                        return "# Error: Rate limit exceeded after retries"
                
                if response.status_code != 200:
                    print(f"  ⚠️  API Error {response.status_code}")
                    # Try to get error message from OpenRouter
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                        print(f"  💡 OpenRouter says: {error_msg}")
                    except:
                        pass
                    return f"# Error: API Error {response.status_code}"
                
                response_data = response.json()
                if "choices" not in response_data:
                    return "# Error: Unexpected API response"
                
                return response_data["choices"][0]["message"]["content"].strip()
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (attempt + 1)
                    print(f"  ⏳ Error: {str(e)}, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                return f"# Error: {str(e)}"
        
        return "# Error: Max retries exceeded"
    
    def generate_for_problem(self, problem_data: dict) -> Tuple[str, float]:
        """Direct code generation for a specific LeetCode problem"""
        problem_id = problem_data.get("id", "Unknown")
        title = problem_data.get("title", "")
        description = problem_data.get("description", "")
        function_sig = problem_data.get("function_signature", "")
        
        print(f"\n🎯 Generating direct solution for Problem {problem_id}: {title}")
        start = time.time()
        
        # Build problem description
        full_problem = f"{title}\n\n{description}"
        
        # Use function signature if available
        if function_sig:
            full_problem += f"\n\nFunction signature: {function_sig}"
        
        # First attempt with strict prompt
        prompt = f"""You are an expert Python programmer. Write COMPLETE, EXECUTABLE Python code for this LeetCode problem.

PROBLEM:
{full_problem}

CRITICAL REQUIREMENTS:
1. Write ONLY valid Python code - NO explanations, NO text before or after
2. Include ALL necessary imports at the top
3. Every function MUST have a complete implementation
4. Code must be syntactically correct and immediately executable
5. Handle all edge cases mentioned in the problem

Begin your response with the first import or class definition. End with the last line of code.
DO NOT include any explanatory text.

CODE:"""
        
        code = self._make_api_call(prompt, max_tokens=2500, temperature=0.3)
        code = self._clean_generated_code(code)
        
        # Validate and retry if needed
        max_retries = 2
        for attempt in range(max_retries):
            if self._validate_code_syntax(code):
                break
            print(f"  ⚠️  Syntax error detected, retry {attempt + 1}/{max_retries}...")
            code = self._retry_with_stricter_prompt(full_problem, attempt)
            code = self._clean_generated_code(code)
        
        elapsed = time.time() - start
        print(f"  ✓ Direct generation completed ({elapsed:.1f}s)")
        print(f"  📝 Code length: {len(code)} characters")
        print(f"  ✅ Syntax valid: {self._validate_code_syntax(code)}")
        
        return code, elapsed
    
    def save_result(self, code: str, problem_id: int, filename: str = None):
        """Save generated code to file"""
        if not filename:
            filename = f"direct_{problem_id}.py"
        
        # Validate syntax before saving
        if not self._validate_code_syntax(code):
            print(f"  ⚠️  WARNING: Code has syntax errors, but saving anyway")
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"# DIRECT GENERATION - Problem {problem_id}\n")
            f.write("# Generated without debate (OpenRouter)\n")
            f.write(f"# Syntax valid: {self._validate_code_syntax(code)}\n\n")
            f.write(code)
        print(f"  ✓ Saved to {filename}")


if __name__ == "__main__":
    # Test with a single problem
    generator = DirectCodeGenerator()
    
    # Example problem (you'll load from leetcode.py)
    test_problem = {
        "id": 42,
        "title": "Trapping Rain Water",
        "description": """Given n non-negative integers representing an elevation map where the width of each bar is 1,
compute how much water it can trap after raining.

Example 1:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

Example 2:
Input: height = [4,2,0,3,2,5]
Output: 9

Constraints:
- n == height.length
- 1 <= n <= 2 * 10^4
- 0 <= height[i] <= 10^5

Requirements:
- Time complexity: O(n)
- Space complexity: O(1) ideally, O(n) acceptable""",
        "function_signature": "def trap(height: List[int]) -> int:"
    }
    
    code, timing = generator.generate_for_problem(test_problem)
    
    print("\n" + "="*70)
    print("GENERATED CODE (First 500 chars):")
    print("="*70)
    print(code[:500] + "..." if len(code) > 500 else code)
    print(f"\nTime taken: {timing:.2f}s")
    
    generator.save_result(code, test_problem["id"])