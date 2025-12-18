"""
Socratic Code Generator - FOR LEETCODE HARD PROBLEMS - FLEXIBLE VERSION
Generates code through flexible debate and blueprint validation with hallucination check
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
    """Flexible debate with blueprint validation and hallucination check"""
    
    def __init__(self, api_caller):
        self._make_api_call = api_caller
        
        # ========== RENAMED PERSONAS ==========
        self.architect_system = """You are a CODING ARCHITECT - Design optimal solution blueprints

YOUR ROLE:
1. Analyze the problem type, constraints, and requirements
2. Design 2-3 architectural approaches with clear trade-offs
3. Recommend the most suitable architecture for given constraints
4. Outline key algorithmic patterns and data structures needed

FORMAT:
PROBLEM ARCHITECTURE:
- Problem Type: [type]
- Key Requirements: [functional requirements]
- Constraints: [time/space/memory limits]

ARCHITECTURAL APPROACHES:
1. [Approach 1 - Architecture Name]: 
   - Design: [high-level design]
   - Pros: [advantages]
   - Cons: [limitations]
   - Complexity: [time/space analysis]

2. [Approach 2 - Architecture Name]:
   - Design: [high-level design]
   - Pros: [advantages]
   - Cons: [limitations]
   - Complexity: [time/space analysis]

RECOMMENDED ARCHITECTURE:
- Selected: [approach name] 
- Reason: [why this fits requirements]
- Key Insight: [main algorithmic idea]
- Risk Factors: [what could go wrong]"""

        self.optimizer_system = """You are a CODE OPTIMIZER - Create detailed, optimized implementation plans

YOUR ROLE:
1. Based on the architectural blueprint, create concrete implementation plans
2. Design function signatures, data structures, and control flow
3. Include optimization techniques and performance considerations
4. Note edge cases and error handling strategies

FORMAT:
OPTIMIZED IMPLEMENTATION PLAN FOR: [architecture name]

IMPLEMENTATION DESIGN:
- Main function: [signature with parameters and return type]
- Key helper functions: [list with purposes]
- Data structures: [structures with memory considerations]
- Algorithm flow: [step-by-step process]

PERFORMANCE OPTIMIZATIONS:
- Time complexity: O(?) - [justification with worst-case]
- Space complexity: O(?) - [justification with memory usage]
- Optimization techniques: [e.g., memoization, pruning, etc.]

EDGE CASE HANDLING:
1. [edge case 1]: [handling strategy]
2. [edge case 2]: [handling strategy]
3. [edge case 3]: [handling strategy]

IMPLEMENTATION PSEUDOCODE:
[Detailed pseudocode showing exact algorithm steps]"""

        self.tester_system = """You are a REALITY TESTER - Validate implementation against reality

YOUR ROLE:
1. Test the implementation plan against problem requirements
2. Verify the pseudocode works with given examples
3. Identify logical gaps, errors, or misunderstandings
4. Check for hallucination (inventing requirements not in problem)

FORMAT:
VALIDATION REPORT:

REQUIREMENTS MATCH:
- Matches problem statement: ✓/✗ [explain]
- Satisfies all constraints: ✓/✗ [time/space/limits]
- Handles all examples: ✓/✗ [show test results]

HALLUCINATION CHECK:
- Invented requirements: [list any requirements not in original problem]
- Missing requirements: [list any requirements from problem not addressed]
- Assumptions made: [list any unwarranted assumptions]

LOGICAL ERRORS:
1. [error 1]: [specific issue and fix]
2. [error 2]: [specific issue and fix]

CONCRETE IMPROVEMENTS NEEDED:
[Specific changes to make implementation correct]"""

        self.hallucination_checker_system = """You are a HALLUCINATION DETECTOR - Ensure the blueprint follows exact problem requirements

YOUR ROLE:
1. Compare the final blueprint against the original problem statement
2. Identify ANY requirements added that weren't in the original problem
3. Flag ANY requirements from the problem that were missed
4. Ensure the solution doesn't invent constraints or assumptions

FORMAT:
HALLUCINATION AUDIT:

ORIGINAL PROBLEM REQUIREMENTS:
[Extract exact requirements from problem statement]

BLUEPRINT REQUIREMENTS:
[Extract requirements from the blueprint]

COMPARISON:
ADDED (Hallucinated):
1. [requirement 1 added that wasn't in original]
2. [requirement 2 added that wasn't in original]

MISSED:
1. [requirement 1 from problem not addressed]
2. [requirement 2 from problem not addressed]

ASSUMPTIONS MADE:
1. [assumption 1 not justified by problem]
2. [assumption 2 not justified by problem]

FINAL VERDICT:
- Pass: ✓ (if no hallucinations and all requirements addressed)
- Fail: ✗ (if hallucinations found or requirements missed)
- Score: [X/Y requirements correctly addressed]

REQUIRED CORRECTIONS:
[Specific corrections needed to eliminate hallucinations]"""
    
    def conduct_flexible_debate(self, problem: str) -> str:
        """Run flexible debate with blueprint validation and hallucination check"""
        print("  🔄 Starting flexible debate with hallucination check...")
        
        debate_log = []
        debate_log.append(f"PROBLEM:\n{problem}\n")
        debate_log.append("\n" + "="*80)
        debate_log.append("FLEXIBLE DEBATE WITH HALLUCINATION VALIDATION")
        debate_log.append("="*80 + "\n")
        
        # ========== PHASE 1: ARCHITECTURAL DESIGN ==========
        debate_log.append("\n--- PHASE 1: ARCHITECTURAL DESIGN ---\n")
        
        print("  🏛️  ARCHITECT designing solution architecture...")
        architect = self._make_api_call(
            f"{self.architect_system}\n\nPROBLEM:\n{problem}\n\nDesign the architectural blueprint for this problem.",
            max_tokens=1000, temperature=0.7
        )
        debate_log.append(f"🏛️  ARCHITECT (ARCHITECTURE):\n{architect}\n")
        time.sleep(2)
        
        # ========== PHASE 2: OPTIMIZED IMPLEMENTATION ==========
        debate_log.append("\n--- PHASE 2: OPTIMIZED IMPLEMENTATION ---\n")
        
        print("  ⚙️  OPTIMIZER creating implementation plan...")
        optimizer = self._make_api_call(
            f"{self.optimizer_system}\n\nARCHITECT'S DESIGN:\n{architect}\n\nPROBLEM:\n{problem}\n\nCreate an optimized implementation plan based on this architecture.",
            max_tokens=1500, temperature=0.6
        )
        debate_log.append(f"⚙️  OPTIMIZER (IMPLEMENTATION PLAN):\n{optimizer}\n")
        time.sleep(2)
        
        # ========== PHASE 3: REALITY TESTING ==========
        debate_log.append("\n--- PHASE 3: REALITY TESTING ---\n")
        
        print("  🧪 TESTER validating implementation...")
        tester = self._make_api_call(
            f"{self.tester_system}\n\nPROBLEM:\n{problem}\n\nARCHITECT'S DESIGN:\n{architect}\n\nOPTIMIZER'S PLAN:\n{optimizer}\n\nValidate this implementation plan and find issues.",
            max_tokens=1200, temperature=0.6
        )
        debate_log.append(f"🧪 TESTER (VALIDATION):\n{tester}\n")
        time.sleep(2)
        
        # ========== PHASE 4: HALLUCINATION CHECK ==========
        debate_log.append("\n--- PHASE 4: HALLUCINATION AUDIT ---\n")
        
        print("  🔍 HALLUCINATION DETECTOR checking for invented requirements...")
        hallucination_check = self._make_api_call(
            f"{self.hallucination_checker_system}\n\nORIGINAL PROBLEM:\n{problem}\n\nCURRENT BLUEPRINT:\n{optimizer}\n\nTESTER'S FEEDBACK:\n{tester}\n\nAudit this blueprint for hallucinations.",
            max_tokens=1000, temperature=0.5
        )
        debate_log.append(f"🔍 HALLUCINATION DETECTOR:\n{hallucination_check}\n")
        time.sleep(2)
        
        # ========== PHASE 5: FINAL SYNTHESIS WITH CORRECTIONS ==========
        debate_log.append("\n--- PHASE 5: FINAL CORRECTED BLUEPRINT ---\n")
        
        print("  ⚙️  OPTIMIZER finalizing with all corrections...")
        final_optimizer = self._make_api_call(
            f"{self.optimizer_system}\n\nPROBLEM:\n{problem}\n\nORIGINAL IMPLEMENTATION PLAN:\n{optimizer}\n\nTESTER'S VALIDATION:\n{tester}\n\nHALLUCINATION AUDIT:\n{hallucination_check}\n\nCreate a FINAL, CORRECTED implementation plan that addresses all issues and eliminates hallucinations.",
            max_tokens=1500, temperature=0.4
        )
        debate_log.append(f"⚙️  OPTIMIZER (FINAL CORRECTED PLAN):\n{final_optimizer}\n")
        
        # ========== PHASE 6: FINAL VALIDATION ==========
        debate_log.append("\n--- PHASE 6: FINAL VALIDATION ---\n")
        
        print("  🧪 TESTER performing final validation...")
        final_tester = self._make_api_call(
            f"{self.tester_system}\n\nPROBLEM:\n{problem}\n\nFINAL IMPLEMENTATION PLAN:\n{final_optimizer}\n\nPerform a final comprehensive validation of the corrected plan.",
            max_tokens=800, temperature=0.5
        )
        debate_log.append(f"🧪 TESTER (FINAL VALIDATION):\n{final_tester}\n")
        
        debate_log.append("\n" + "="*80)
        debate_log.append("DEBATE-VALIDATION-HALLUCINATION-CHECK COMPLETED")
        debate_log.append("="*80)
        
        return "\n".join(debate_log), final_optimizer


class SocraticCodeGenerator:
    def __init__(self):
        # CHANGE 1: Switch to OpenRouter API key
        self.api_key = os.getenv("OPENROUTER_API_KEY")  # Changed from GROQ_API_KEY
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"  # Changed URL
        print(f"🤖 Socratic Generator initialized (OpenRouter - Flexible Debate with Hallucination Check)")
        
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
            model = "meta-llama/llama-3.3-70b-instruct"
        
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
    
    def _generate_code_from_optimizer(self, problem: str, optimizer_plan: str) -> str:
        """Generate code from validated optimizer plan"""
        prompt = f"""Convert this OPTIMIZED IMPLEMENTATION PLAN into complete, executable Python code:

PROBLEM:
{problem}

VALIDATED OPTIMIZER PLAN:
{optimizer_plan}

Write complete, production-ready Python code that implements the optimizer plan exactly.
Follow these guidelines:

1. IMPLEMENTATION FIDELITY:
   - Follow the optimizer plan structure EXACTLY
   - Implement ALL helper functions mentioned
   - Use EXACT data structures specified
   - Maintain EXACT time/space complexity

2. CODE QUALITY:
   - Add comprehensive docstrings explaining algorithm
   - Include type hints for all functions
   - Add comments for complex logic
   - Handle all edge cases mentioned in plan

3. ERROR HANDLING:
   - Add input validation
   - Raise appropriate exceptions for invalid inputs
   - Include meaningful error messages

4. TEST READINESS:
   - Make code easy to test
   - Ensure functions are pure where possible
   - Avoid global state

Return ONLY the Python code, no explanations.

CODE:"""
        
        # Use a model good at code generation
        code = self._make_api_call(prompt, max_tokens=2500, temperature=0.3, 
                                  model="meta-llama/llama-3.3-70b-instruct")
        return self._clean_generated_code(code)
    
    def generate_for_problem(self, problem_data: dict) -> Tuple[dict, float]:
        """Full Socratic generation with flexible debate and hallucination check"""
        problem_id = problem_data.get("id", "Unknown")
        title = problem_data.get("title", "")
        description = problem_data.get("description", "")
        
        print(f"\n{'='*80}")
        print(f"SOCRATIC GENERATION WITH HALLUCINATION CHECK - Problem {problem_id}: {title}")
        print(f"{'='*80}")
        
        total_start = time.time()
        results = {}
        
        full_problem = f"{title}\n\n{description}"
        
        # Step 1: Flexible debate with hallucination check
        print("\n💭 Conducting flexible debate with hallucination validation...")
        debate_manager = FlexibleDebateValidator(self._make_api_call)
        debate_log, final_optimizer_plan = debate_manager.conduct_flexible_debate(full_problem)
        results['debate_log'] = debate_log
        results['optimizer_plan'] = final_optimizer_plan
        
        print(f"  ✓ Debate completed, optimized plan ready ({len(final_optimizer_plan)} chars)")
        time.sleep(2)
        
        # Step 2: Generate code from validated optimizer plan
        print("\n⚙️  Generating code from validated optimizer plan...")
        code = self._generate_code_from_optimizer(full_problem, final_optimizer_plan)
        
        # Enhanced syntax validation with multiple retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                import ast
                ast.parse(code)
                print(f"  ✅ Syntax valid on attempt {attempt + 1}")
                break  # Syntax is valid
            except SyntaxError as e:
                print(f"  ⚠️  Syntax error, retry {attempt + 1}/{max_retries}...")
                print(f"  Error: {str(e)[:100]}")
                
                if attempt < max_retries - 1:
                    # Retry with more specific prompt
                    retry_prompt = f"""Fix syntax errors in this Python code:

PROBLEM:
{full_problem}

OPTIMIZER PLAN:
{final_optimizer_plan}

CODE WITH ERRORS:
{code}

SYNTAX ERROR: {str(e)}

Provide ONLY the corrected Python code with proper syntax, no explanations:"""
                    
                    code = self._make_api_call(retry_prompt, max_tokens=2500, temperature=0.2)
                    code = self._clean_generated_code(code)
                else:
                    print("  ❌ Max retries reached for syntax errors")
        
        results['code'] = code
        results['total_time'] = time.time() - total_start
        
        print(f"\n📊 Generation completed in {results['total_time']:.1f}s")
        print(f"  Code length: {len(code)} characters")
        print(f"  Debate log length: {len(debate_log)} characters")
        print(f"  Optimizer plan length: {len(final_optimizer_plan)} characters")
        
        return results, results['total_time']
    
    def save_results(self, results: dict, problem_id: int):
        """Save generated code and debate"""
        # Save code
        code_file = f"socratic_{problem_id}.py"
        with open(code_file, "w", encoding='utf-8') as f:
            f.write(f"# SOCRATIC GENERATION WITH HALLUCINATION CHECK - Problem {problem_id}\n")
            f.write("# Generated from validated optimizer plan (OpenRouter)\n\n")
            f.write(results['code'])
        print(f"  ✓ Saved code to {code_file}")
        
        # Save optimizer plan
        optimizer_file = f"optimizer_plan_{problem_id}.txt"
        with open(optimizer_file, "w", encoding='utf-8') as f:
            f.write(f"VALIDATED OPTIMIZER PLAN - Problem {problem_id}\n")
            f.write("="*80 + "\n\n")
            f.write(results.get('optimizer_plan', 'No optimizer plan generated'))
        print(f"  ✓ Saved optimizer plan to {optimizer_file}")
        
        # Save debate log
        debate_file = f"debate_log_{problem_id}.txt"
        with open(debate_file, "w", encoding='utf-8') as f:
            f.write(f"DEBATE LOG WITH HALLUCINATION CHECK - Problem {problem_id}\n")
            f.write("="*80 + "\n\n")
            f.write(results.get('debate_log', 'No debate log generated'))
        print(f"  ✓ Saved debate log to {debate_file}")


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

def generate_for_humaneval(problem_prompt: str, problem_id: int) -> str:
    """Simple wrapper for EvalPlus compatibility"""
    generator = SocraticCodeGenerator()
    return generator.generate_for_humaneval(problem_prompt)


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
    
    print("\n" + "="*80)
    print("CODE PREVIEW (First 500 chars):")
    print("="*80)
    print(results['code'][:500] + "..." if len(results['code']) > 500 else results['code'])
    
    print(f"\nTotal time: {timing:.2f}s")
    
    generator.save_results(results, test_problem["id"])