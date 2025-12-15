"""
Socratic Code Generator - FOR LEETCODE HARD PROBLEMS - FLEXIBLE VERSION
Generates code through flexible debate and blueprint validation
"""
import os
import time
import requests
import re
import warnings
from typing import Tuple
from dotenv import load_dotenv

warnings.filterwarnings('ignore', message='Unverified HTTPS request')
load_dotenv()


class FlexibleDebateValidator:
    """Flexible debate with blueprint validation"""
    
    def __init__(self, api_caller):
        self._make_api_call = api_caller
        
        # ========== MORE FLEXIBLE PERSONAS ==========
        self.advisor_system = """You are a CODING ADVISOR - Help design optimal solutions

YOUR ROLE:
1. Analyze the problem type and constraints
2. Suggest 2-3 viable approaches with trade-offs
3. Recommend the best approach for given constraints
4. Outline key algorithmic insights

FORMAT:
PROBLEM ANALYSIS:
- Type: [type]
- Key constraints: [list]

APPROACHES:
1. [Approach 1]: [pros/cons] - [time/space]
2. [Approach 2]: [pros/cons] - [time/space]
3. [Approach 3]: [pros/cons] - [time/space]

RECOMMENDATION:
- Best: [approach] because [reason]
- Key insight: [main algorithmic idea]
- Potential pitfalls: [what to watch for]"""

        self.blueprint_system = """You are a BLUEPRINT ENGINEER - Create concrete implementation plans

YOUR ROLE:
1. Based on the recommendation, create detailed implementation blueprint
2. Include function signatures, data structures, and key algorithms
3. Add pseudocode for complex parts
4. Note edge cases to handle

FORMAT:
BLUEPRINT FOR: [approach name]

IMPLEMENTATION STRUCTURE:
- Main function: [signature]
- Key helper functions: [list]
- Data structures: [list with purposes]

PSEUDOCODE:
[Step-by-step pseudocode showing algorithm flow]

EDGE CASES TO HANDLE:
1. [case 1]: [how to handle]
2. [case 2]: [how to handle]
3. [case 3]: [how to handle]

COMPLEXITY VERIFICATION:
- Time: O(?) - [justification]
- Space: O(?) - [justification]"""

        self.validator_system = """You are a REALITY VALIDATOR - Check blueprints against reality

YOUR ROLE:
1. Verify the blueprint matches problem requirements
2. Test the pseudocode with given examples
3. Identify logical gaps or errors
4. Suggest concrete fixes

FORMAT:
VALIDATION REPORT:

CORRECTNESS CHECK:
- Matches problem: ✓/✗ [explain]
- Handles examples: ✓/✗ [show test]
- Satisfies constraints: ✓/✗ [time/space]

LOGICAL GAPS:
1. [gap 1]: [why it's a problem]
2. [gap 2]: [why it's a problem]

ERRORS FOUND:
1. [error 1]: [specific fix]
2. [error 2]: [specific fix]

CONCRETE IMPROVEMENTS:
[Specific changes needed in blueprint]"""
    
    def conduct_flexible_debate(self, problem: str) -> str:
        """Run flexible debate with blueprint validation"""
        print("  🔄 Starting flexible debate with blueprint...")
        
        debate_log = []
        debate_log.append(f"PROBLEM:\n{problem}\n")
        debate_log.append("\n" + "="*70)
        debate_log.append("FLEXIBLE DEBATE WITH BLUEPRINT VALIDATION")
        debate_log.append("="*70 + "\n")
        
        # ========== PHASE 1: INITIAL ANALYSIS ==========
        debate_log.append("\n--- PHASE 1: PROBLEM ANALYSIS ---\n")
        
        print("  🧠 ADVISOR analyzing problem...")
        advisor = self._make_api_call(
            f"{self.advisor_system}\n\nPROBLEM:\n{problem}\n\nAnalyze this problem and recommend approaches.",
            max_tokens=800, temperature=0.7
        )
        debate_log.append(f"🧠 ADVISOR:\n{advisor}\n")
        time.sleep(2)
        
        # ========== PHASE 2: BLUEPRINT CREATION ==========
        debate_log.append("\n--- PHASE 2: BLUEPRINT CREATION ---\n")
        
        print("  📐 ENGINEER creating blueprint...")
        blueprint = self._make_api_call(
            f"{self.blueprint_system}\n\nADVISOR'S ANALYSIS:\n{advisor}\n\nPROBLEM:\n{problem}\n\nCreate a detailed blueprint for the recommended approach.",
            max_tokens=1200, temperature=0.6
        )
        debate_log.append(f"📐 ENGINEER (BLUEPRINT):\n{blueprint}\n")
        time.sleep(2)
        
        # ========== PHASE 3: BLUEPRINT VALIDATION ==========
        debate_log.append("\n--- PHASE 3: BLUEPRINT VALIDATION ---\n")
        
        print("  🔍 VALIDATOR checking blueprint...")
        validator = self._make_api_call(
            f"{self.validator_system}\n\nPROBLEM:\n{problem}\n\nADVISOR'S ANALYSIS:\n{advisor}\n\nBLUEPRINT:\n{blueprint}\n\nValidate this blueprint and suggest fixes.",
            max_tokens=1000, temperature=0.6
        )
        debate_log.append(f"🔍 VALIDATOR:\n{validator}\n")
        time.sleep(2)
        
        # ========== PHASE 4: FINAL SYNTHESIS ==========
        debate_log.append("\n--- PHASE 4: FINAL SYNTHESIS ---\n")
        
        print("  📐 ENGINEER finalizing blueprint...")
        final_blueprint = self._make_api_call(
            f"{self.blueprint_system}\n\nPROBLEM:\n{problem}\n\nORIGINAL BLUEPRINT:\n{blueprint}\n\nVALIDATOR'S FEEDBACK:\n{validator}\n\nCreate a FINAL, CORRECTED blueprint incorporating all fixes.",
            max_tokens=1200, temperature=0.5
        )
        debate_log.append(f"📐 ENGINEER (FINAL BLUEPRINT):\n{final_blueprint}\n")
        
        debate_log.append("\n" + "="*70)
        debate_log.append("DEBATE-BLUEPRINT-DEBATE COMPLETED")
        debate_log.append("="*70)
        
        return "\n".join(debate_log), final_blueprint


class SocraticCodeGenerator:
    def __init__(self):
        # CHANGE 1: Switch to OpenRouter API key
        self.api_key = os.getenv("OPENROUTER_API_KEY")  # Changed from GROQ_API_KEY
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"  # Changed URL
        print(f"🤖 Socratic Generator initialized (OpenRouter - Flexible Debate)")
        
        # Test API
        test_response = self._make_api_call("Say 'OpenRouter test successful'", max_tokens=20, temperature=0.1)
        print(f"🔧 API Test: {'✅ Working' if 'test' in test_response.lower() else f'❌ Failed: {test_response}'}")
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean LLM-generated code"""
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        return code.strip()
    
    def _make_api_call(self, prompt: str, max_tokens: int, temperature: float = 0.7, model: str = None) -> str:
        """Make API call to OpenRouter with retry logic"""
        if not self.api_key:
            return "Error: OPENROUTER_API_KEY not found in environment variables"
        
        # CHANGE 2: OpenRouter headers and data structure
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",  # Required by OpenRouter
            "X-Title": "Socratic Code Generator"       # Your app name
        }
        
        # CHANGE 3: Choose OpenRouter model (Llama 3.3 70B)
        if model is None:
            # Using the free tier model first
            model = "meta-llama/llama-3.3-70b-instruct:free"
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        
        for attempt in range(3):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=data, 
                    timeout=120  # Longer timeout for larger models
                )
                
                # CHANGE 4: Handle OpenRouter-specific rate limits
                if response.status_code == 429:
                    error_msg = response.json().get("error", {}).get("message", "")
                    if "free tier" in error_msg.lower() and attempt == 0:
                        print("    ⚠️ Free tier limit, switching to paid model...")
                        # Switch to paid model
                        data["model"] = model.replace(":free", "")
                        continue
                    
                    wait_time = 5 * (attempt + 1)
                    print(f"    ⚠️ Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code != 200:
                    error_msg = f"OpenRouter Error {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg += f": {error_data['error'].get('message', 'Unknown')}"
                    except:
                        pass
                    print(f"    ❌ {error_msg}")
                    
                    # Fallback to Groq if OpenRouter fails
                    if attempt == 2:
                        return self._fallback_to_groq(prompt, max_tokens, temperature)
                    continue
                
                response_data = response.json()
                if "choices" not in response_data or not response_data["choices"]:
                    return "Error: No choices in response"
                
                content = response_data["choices"][0]["message"]["content"].strip()
                return content if content else "Error: Empty response content"
                
            except Exception as e:
                print(f"    ⚠️ Exception: {str(e)[:100]}")
                if attempt < 2:
                    time.sleep(2)
                    continue
                return f"Error: {str(e)[:100]}"
        
        return "Error: Max retries exceeded"
    
    def _fallback_to_groq(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Fallback to Groq if OpenRouter fails"""
        print("    🔄 Falling back to Groq...")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return "Error: No fallback API available"
        
        groq_headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
        groq_data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=groq_headers,
                json=groq_data,
                timeout=90
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            return f"Groq fallback failed: {str(e)[:100]}"
        
        return "Error: All API providers failed"
    
    def _generate_code_from_blueprint(self, problem: str, blueprint: str) -> str:
        """Generate code from validated blueprint"""
        prompt = f"""Convert this blueprint into complete Python code:

PROBLEM:
{problem}

VALIDATED BLUEPRINT:
{blueprint}

Write complete, executable Python code that implements the blueprint exactly.
Include all necessary imports and handle all edge cases mentioned.

IMPORTANT: 
1. Follow the blueprint structure exactly
2. Implement all helper functions mentioned
3. Add docstrings explaining the algorithm
4. Ensure time and space complexity match the blueprint

Return ONLY the Python code, no explanations.

CODE:"""
        
        # Use the free model for code generation
        code = self._make_api_call(prompt, max_tokens=2000, temperature=0.3)
        return self._clean_generated_code(code)
    
    def generate_for_problem(self, problem_data: dict) -> Tuple[dict, float]:
        """Full Socratic generation with flexible debate"""
        problem_id = problem_data.get("id", "Unknown")
        title = problem_data.get("title", "")
        description = problem_data.get("description", "")
        
        print(f"\n{'='*70}")
        print(f"SOCRATIC GENERATION - Problem {problem_id}: {title}")
        print(f"{'='*70}")
        
        total_start = time.time()
        results = {}
        
        full_problem = f"{title}\n\n{description}"
        
        # Step 1: Flexible debate with blueprint
        print("\n💭 Conducting flexible debate with blueprint...")
        debate_manager = FlexibleDebateValidator(self._make_api_call)
        debate_log, final_blueprint = debate_manager.conduct_flexible_debate(full_problem)
        results['debate'] = debate_log
        results['blueprint'] = final_blueprint
        
        print(f"  ✓ Debate completed, blueprint ready ({len(final_blueprint)} chars)")
        time.sleep(2)
        
        # Step 2: Generate code from validated blueprint
        print("\n⚙️  Generating code from validated blueprint...")
        code = self._generate_code_from_blueprint(full_problem, final_blueprint)
        
        # Syntax validation and retry
        max_retries = 2
        for attempt in range(max_retries):
            try:
                import ast
                ast.parse(code)
                break  # Syntax is valid
            except SyntaxError:
                print(f"  ⚠️  Syntax error, retry {attempt + 1}/{max_retries}...")
                # Retry with stricter prompt
                retry_prompt = f"""Fix syntax errors in this code for problem:

{full_problem}

Code with errors:
{code}

Provide ONLY the corrected Python code, no explanations:"""
                
                code = self._make_api_call(retry_prompt, max_tokens=2000, temperature=0.2)
                code = self._clean_generated_code(code)
        
        results['code'] = code
        results['total_time'] = time.time() - total_start
        
        print(f"\n📊 Generation completed in {results['total_time']:.1f}s")
        print(f"  Code length: {len(code)} characters")
        
        return results, results['total_time']
    
    def save_results(self, results: dict, problem_id: int):
        """Save generated code and debate"""
        # Save code
        code_file = f"socratic_{problem_id}.py"
        with open(code_file, "w", encoding='utf-8') as f:
            f.write(f"# SOCRATIC GENERATION - Problem {problem_id}\n")
            f.write("# Generated from validated blueprint (OpenRouter)\n\n")
            f.write(results['code'])
        print(f"  ✓ Saved code to {code_file}")
        
        # Save blueprint
        blueprint_file = f"blueprint_{problem_id}.txt"
        with open(blueprint_file, "w", encoding='utf-8') as f:
            f.write(f"VALIDATED BLUEPRINT - Problem {problem_id}\n")
            f.write("="*70 + "\n\n")
            f.write(results.get('blueprint', 'No blueprint generated'))
        print(f"  ✓ Saved blueprint to {blueprint_file}")


# ADD THIS: Test function for OpenRouter connection
def test_openrouter_connection():
    """Test OpenRouter connection before running main"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in .env file")
        print("Please add: OPENROUTER_API_KEY=your-key-here")
        return False
    
    print("✅ OPENROUTER_API_KEY found")
    
    # Quick test
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "messages": [{"role": "user", "content": "Say 'test successful'"}],
                "max_tokens": 10
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ OpenRouter connection successful!")
            return True
        else:
            print(f"❌ OpenRouter error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test connection first
    if not test_openrouter_connection():
        print("\n⚠️  OpenRouter connection failed. Please check:")
        print("1. Your .env file has OPENROUTER_API_KEY")
        print("2. You have credits on OpenRouter")
        print("3. Your internet connection is working")
        exit(1)
    
    generator = SocraticCodeGenerator()
    
    # Example problem
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
- Space complexity: O(1) ideally, O(n) acceptable"""
    }
    
    results, timing = generator.generate_for_problem(test_problem)
    
    print("\n" + "="*70)
    print("CODE PREVIEW (First 500 chars):")
    print("="*70)
    print(results['code'][:500] + "..." if len(results['code']) > 500 else results['code'])
    
    print(f"\nTotal time: {timing:.2f}s")
    
    generator.save_results(results, test_problem["id"])