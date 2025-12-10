import os
import time
import requests
import json
import re
import warnings
import ast
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

warnings.filterwarnings('ignore', message='Unverified HTTPS request')
load_dotenv()

@dataclass
class EdgeCaseResult:
    test_name: str
    passed: bool
    error: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None

@dataclass
class CodeEvaluation:
    is_complete: bool
    has_syntax_errors: bool
    syntax_error_msg: Optional[str]
    edge_cases_passed: List[EdgeCaseResult]
    edge_cases_failed: List[EdgeCaseResult]
    edge_case_score: float  # percentage
    execution_errors: List[str]

@dataclass
class SynthesisResult:
    problem: str
    direct_code: str
    socratic_plan: str
    socratic_debate: str
    socratic_code: str
    direct_eval: CodeEvaluation
    socratic_eval: CodeEvaluation
    llm_comparison: str
    timing: Dict[str, float]

class CodeValidator:
    """Actually execute code and test edge cases"""
    
    @staticmethod
    def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """Check if code has syntax errors"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def check_completeness(code: str) -> bool:
        """Check if code has obvious incompleteness markers"""
        incomplete_markers = [
            'pass  # TODO',
            '# TODO',
            '# implement',
            '# placeholder',
            'raise NotImplementedError',
            '...'
        ]
        
        code_lower = code.lower()
        
        # Check for markers
        for marker in incomplete_markers:
            if marker.lower() in code_lower:
                return False
        
        # Check for functions with only 'pass'
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if 'def ' in line and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line == 'pass' or next_line == '':
                    return False
        
        return True
    
    @staticmethod
    def execute_edge_case_tests(code: str, test_cases: List[Dict]) -> List[EdgeCaseResult]:
        """Execute code with specific test cases"""
        results = []
        
        # Create isolated namespace
        namespace = {}
        
        try:
            # Execute the code
            exec(code, namespace)
        except Exception as e:
            # Code doesn't even run
            return [EdgeCaseResult(
                test_name="Code Execution",
                passed=False,
                error=f"Failed to execute: {str(e)}"
            )]
        
        # Run each test case
        for test in test_cases:
            try:
                test_name = test['name']
                setup = test.get('setup', '')
                test_code = test['test']
                expected = test.get('expected')
                
                # Execute setup if provided
                if setup:
                    exec(setup, namespace)
                
                # Execute test
                exec(test_code, namespace)
                result_var = test.get('result_var', 'result')
                actual = namespace.get(result_var)
                
                # Check result
                if expected is not None:
                    passed = actual == expected
                    results.append(EdgeCaseResult(
                        test_name=test_name,
                        passed=passed,
                        expected=expected,
                        actual=actual,
                        error=None if passed else f"Expected {expected}, got {actual}"
                    ))
                else:
                    # Just check it didn't crash
                    results.append(EdgeCaseResult(
                        test_name=test_name,
                        passed=True,
                        error=None
                    ))
                    
            except Exception as e:
                results.append(EdgeCaseResult(
                    test_name=test['name'],
                    passed=False,
                    error=f"{type(e).__name__}: {str(e)}"
                ))
        
        return results
    
    @staticmethod
    def evaluate_code(code: str, edge_case_tests: List[Dict]) -> CodeEvaluation:
        """Comprehensive code evaluation"""
        
        # Check syntax
        has_valid_syntax, syntax_error = CodeValidator.check_syntax(code)
        
        # Check completeness
        is_complete = CodeValidator.check_completeness(code)
        
        # Run edge case tests
        edge_case_results = []
        execution_errors = []
        
        if has_valid_syntax:
            edge_case_results = CodeValidator.execute_edge_case_tests(code, edge_case_tests)
        else:
            execution_errors.append(f"Syntax error: {syntax_error}")
        
        # Calculate scores
        passed = [r for r in edge_case_results if r.passed]
        failed = [r for r in edge_case_results if not r.passed]
        
        edge_case_score = (len(passed) / len(edge_case_results) * 100) if edge_case_results else 0.0
        
        return CodeEvaluation(
            is_complete=is_complete,
            has_syntax_errors=not has_valid_syntax,
            syntax_error_msg=syntax_error,
            edge_cases_passed=passed,
            edge_cases_failed=failed,
            edge_case_score=edge_case_score,
            execution_errors=execution_errors
        )

class ImprovedSocraticCoder:
    def __init__(self, model: str = "llama-3.1-70b-versatile"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = model
        print(f"🤖 Using model: {model}")
        
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
            
            # Skip obvious non-code lines
            if any(phrase in line_lower for phrase in skip_phrases) and not line.strip().startswith('#'):
                continue
            
            # Skip empty explanation lines at the start
            if not cleaned_lines and not line.strip():
                continue
                
            cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines).strip()
        
        # Remove trailing explanations after code
        last_def_index = -1
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                last_def_index = i
        
        # If we found definitions, cut off obvious trailing text
        if last_def_index > 0:
            for i in range(last_def_index + 1, len(lines)):
                if lines[i] and not lines[i][0].isspace() and not lines[i].startswith('#'):
                    if not lines[i].strip().startswith('def ') and not lines[i].strip().startswith('class '):
                        code = '\n'.join(lines[:i]).strip()
                        break
        
        return code
    
    def _make_api_call(self, prompt: str, max_tokens: int, temperature: float = 0.7) -> str:
        """Make API call with error handling and aggressive retry logic"""
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
        
        max_retries = 5  # Increased from 3
        base_delay = 2.0  # Increased from 1.0
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=data, 
                    verify=False, 
                    timeout=90  # Increased timeout
                )
                
                if response.status_code == 429:
                    # Rate limit hit - parse the retry-after if available
                    if attempt < max_retries - 1:
                        try:
                            error_data = response.json()
                            # Try to extract wait time from error message
                            if 'message' in error_data:
                                import re
                                match = re.search(r'try again in (\d+)ms', error_data['message'])
                                if match:
                                    wait_ms = int(match.group(1))
                                    wait_time = (wait_ms / 1000.0) + 1.0  # Add 1s buffer
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
                        print("  💡 Tip: Wait 60 seconds and try again, or upgrade your Groq tier")
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
                    
                    # Retry on server errors (5xx)
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
                    print(f"  ⏳ Timeout (attempt {attempt+1}/{max_retries}), retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                print("  ⚠️  API Timeout after all retries")
                return "Error: API timeout"
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (attempt + 1)
                    print(f"  ⏳ Error: {str(e)} (attempt {attempt+1}/{max_retries}), retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                print(f"  ⚠️  {str(e)}")
                return f"Error: {str(e)}"
        
        return "Error: Max retries exceeded"
    
    def generate_edge_case_tests(self, problem: str) -> List[Dict]:
        """Generate executable test cases for the problem"""
        print("\n🧪 Generating edge case tests...")
        start = time.time()
        
        prompt = f"""Generate 8-10 executable Python test cases for this problem:

{problem}

Create tests for:
1. Empty/null inputs
2. Single element
3. Boundary values (min/max capacity)
4. Concurrent access (if applicable)
5. Invalid inputs
6. Normal operation
7. Expired items (if TTL)
8. Overflow conditions

Format each test as JSON:
[
  {{
    "name": "Empty input test",
    "setup": "obj = ClassName(capacity=5)",
    "test": "result = obj.get('nonexistent')",
    "result_var": "result",
    "expected": null
  }},
  ...
]

ONLY output the JSON array, nothing else."""
        
        response = self._make_api_call(prompt, max_tokens=4000, temperature=0.7)
        
        # Parse JSON
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                tests = json.loads(json_match.group(0))
                print(f"  ✓ Generated {len(tests)} test cases ({time.time() - start:.1f}s)")
                return tests
            else:
                print(f"  ⚠️  Failed to parse test cases")
                return []
        except Exception as e:
            print(f"  ⚠️  Error parsing tests: {e}")
            return []
    
    def generate_direct(self, problem: str) -> Tuple[str, float]:
        """Direct code generation (baseline) - IMPROVED"""
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
    
    def plan_debate(self, problem: str) -> Tuple[str, float]:
        """Create debate agenda"""
        print("\n📋 Planning debate...")
        start = time.time()
        
        prompt = f"""Analyze this coding problem and create a debate agenda:

{problem}

List:
1. Two viable algorithmic approaches
2. Key trade-offs to debate (performance, memory, simplicity)
3. Five critical edge cases to handle
4. Main design decisions needed

Keep it concise."""
        
        plan = self._make_api_call(prompt, max_tokens=400, temperature=0.8)
        elapsed = time.time() - start
        
        print(f"  ✓ Planning completed ({elapsed:.1f}s)")
        return plan, elapsed
    
    def conduct_debate(self, problem: str, plan: str) -> Tuple[str, float]:
        """Run structured debate between three personas"""
        print("\n💬 Conducting debate...")
        start = time.time()
        
        prompt = f"""Three experts debate the best implementation approach.

PROBLEM: {problem}

DEBATE AGENDA:
{plan}

THREE PERSONAS:
🏗️ ARCHITECT - Proposes elegant, maintainable designs
🔬 TESTER - Finds edge cases and failure modes  
⚡ OPTIMIZER - Pushes for performance and efficiency

DEBATE (3 rounds):

ROUND 1 - Initial Positions:
🏗️ ARCHITECT: [Propose clean solution approach]
🔬 TESTER: [Challenge with 2 specific edge cases that could fail]
⚡ OPTIMIZER: [Suggest performance improvements]

ROUND 2 - Refinement:
🏗️ ARCHITECT: [Adapt design to handle edge cases]
🔬 TESTER: [Verify edge cases fixed, find new concerns]
⚡ OPTIMIZER: [Evaluate complexity trade-offs]

ROUND 3 - Consensus:
🏗️ ARCHITECT: [Final design incorporating all feedback]
🔬 TESTER: [Confirm edge cases covered]
⚡ OPTIMIZER: [Approve or suggest final tweak]

START DEBATE:"""
        
        debate = self._make_api_call(prompt, max_tokens=1200, temperature=0.85)
        elapsed = time.time() - start
        
        print(f"  ✓ Debate completed ({elapsed:.1f}s)")
        return debate, elapsed
    
    def synthesize_code(self, problem: str, debate: str) -> Tuple[str, float]:
        """Generate code from debate - IMPROVED"""
        print("\n⚙️  Synthesizing code from debate...")
        start = time.time()
        
        prompt = f"""You are an expert Python programmer implementing the solution agreed upon in a debate.

ORIGINAL PROBLEM:
{problem}

DEBATE CONSENSUS (last part):
{debate[-1000:]}

YOUR TASK: Write COMPLETE, EXECUTABLE Python code implementing what was agreed upon.

CRITICAL REQUIREMENTS:
1. Write ONLY valid Python code - NO markdown, NO explanations
2. Include ALL necessary imports (threading, time, collections, etc.)
3. FULLY IMPLEMENT every method - NO 'pass', NO '# TODO', NO 'raise NotImplementedError'
4. If debate mentions thread-safety, use actual locks (threading.Lock)
5. If debate mentions TTL, implement actual time-based expiration
6. If debate mentions O(1), use appropriate data structures (dict + OrderedDict)
7. Code must be syntactically correct and run without errors
8. DO NOT include usage examples or test code

Begin directly with imports or class definition. NO text before or after.

CODE:"""
        
        code = self._make_api_call(prompt, max_tokens=2500, temperature=0.3)
        code = self._clean_generated_code(code)
        
        elapsed = time.time() - start
        print(f"  ✓ Code synthesis completed ({elapsed:.1f}s)")
        
        return code, elapsed
    
    def llm_comparison(self, problem: str, direct_code: str, socratic_code: str, 
                      direct_eval: CodeEvaluation, socratic_eval: CodeEvaluation) -> Tuple[str, float]:
        """LLM-based qualitative comparison"""
        print("\n🤖 Running LLM comparison...")
        start = time.time()
        
        prompt = f"""Compare these two solutions qualitatively.

PROBLEM: {problem}

SOLUTION A (Direct):
{direct_code[:800]}
Objective Results: {direct_eval.edge_case_score:.1f}% edge cases passed

SOLUTION B (Socratic):
{socratic_code[:800]}
Objective Results: {socratic_eval.edge_case_score:.1f}% edge cases passed

Analyze:
1. Code quality and readability
2. Design patterns used
3. Error handling approach
4. Completeness of implementation

Keep analysis brief (4-5 sentences)."""
        
        comparison = self._make_api_call(prompt, max_tokens=400, temperature=0.3)
        elapsed = time.time() - start
        
        print(f"  ✓ LLM comparison completed ({elapsed:.1f}s)")
        return comparison, elapsed
    
    def run_full_comparison(self, problem: str, edge_case_tests: List[Dict]) -> SynthesisResult:
        """Run both approaches and compare with REAL testing"""
        print(f"\n{'='*70}")
        print(f"RUNNING SYNTHESIS COMPARISON WITH REAL EDGE CASE TESTING")
        print(f"{'='*70}")
        
        timing = {}
        
        # Method 1: Direct Generation
        direct_code, t1 = self.generate_direct(problem)
        timing['direct'] = t1
        
        # Add delay to avoid rate limits
        print("  ⏳ Cooling down to avoid rate limits (3s)...")
        time.sleep(3)
        
        # Method 2: Socratic Method
        plan, t2 = self.plan_debate(problem)
        timing['plan'] = t2
        
        print("  ⏳ Cooling down (3s)...")
        time.sleep(3)
        
        debate, t3 = self.conduct_debate(problem, plan)
        timing['debate'] = t3
        
        print("  ⏳ Cooling down (3s)...")
        time.sleep(3)
        
        socratic_code, t4 = self.synthesize_code(problem, debate)
        timing['synthesis'] = t4
        timing['socratic_total'] = t2 + t3 + t4
        
        # REAL evaluation with actual code execution
        print("\n" + "="*70)
        print("OBJECTIVE EVALUATION (ACTUAL CODE EXECUTION)")
        print("="*70)
        
        print("\n🧪 Testing Direct solution...")
        direct_eval = CodeValidator.evaluate_code(direct_code, edge_case_tests)
        
        print("\n🧪 Testing Socratic solution...")
        socratic_eval = CodeValidator.evaluate_code(socratic_code, edge_case_tests)
        
        print("  ⏳ Cooling down (3s)...")
        time.sleep(3)
        
        # LLM comparison for qualitative aspects
        llm_comp, t5 = self.llm_comparison(problem, direct_code, socratic_code, 
                                          direct_eval, socratic_eval)
        timing['evaluation'] = t5
        timing['total'] = sum(timing.values())
        
        print(f"\n{'='*70}")
        print(f"COMPARISON COMPLETE")
        print(f"{'='*70}")
        
        return SynthesisResult(
            problem=problem,
            direct_code=direct_code,
            socratic_plan=plan,
            socratic_debate=debate,
            socratic_code=socratic_code,
            direct_eval=direct_eval,
            socratic_eval=socratic_eval,
            llm_comparison=llm_comp,
            timing=timing
        )
    
    def print_results(self, result: SynthesisResult):
        """Print formatted results with objective metrics"""
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        print("\n📝 PROBLEM:")
        print(result.problem)
        
        print("\n" + "-"*70)
        print("METHOD 1: DIRECT GENERATION")
        print("-"*70)
        print(result.direct_code[:400] + "..." if len(result.direct_code) > 400 else result.direct_code)
        
        print("\n📊 Direct Solution Evaluation:")
        self._print_evaluation(result.direct_eval)
        
        print("\n" + "-"*70)
        print("METHOD 2: SOCRATIC DEBATE")
        print("-"*70)
        print("\n💬 Debate (excerpt):")
        print(result.socratic_debate[:300] + "..." if len(result.socratic_debate) > 300 else result.socratic_debate)
        
        print("\n⚙️  Synthesized Code:")
        print(result.socratic_code[:400] + "..." if len(result.socratic_code) > 400 else result.socratic_code)
        
        print("\n📊 Socratic Solution Evaluation:")
        self._print_evaluation(result.socratic_eval)
        
        print("\n" + "="*70)
        print("📊 OBJECTIVE COMPARISON")
        print("="*70)
        
        # Side-by-side metrics
        print(f"\n{'Metric':<30} {'Direct':<15} {'Socratic':<15} {'Winner'}")
        print("-" * 70)
        
        # Completeness
        direct_complete = "✓ Complete" if result.direct_eval.is_complete else "✗ Incomplete"
        socratic_complete = "✓ Complete" if result.socratic_eval.is_complete else "✗ Incomplete"
        winner = "Tie" if result.direct_eval.is_complete == result.socratic_eval.is_complete else \
                 ("Socratic" if result.socratic_eval.is_complete else "Direct")
        print(f"{'Completeness':<30} {direct_complete:<15} {socratic_complete:<15} {winner}")
        
        # Syntax
        direct_syntax = "✓ Valid" if not result.direct_eval.has_syntax_errors else "✗ Errors"
        socratic_syntax = "✓ Valid" if not result.socratic_eval.has_syntax_errors else "✗ Errors"
        winner = "Tie" if result.direct_eval.has_syntax_errors == result.socratic_eval.has_syntax_errors else \
                 ("Socratic" if not result.socratic_eval.has_syntax_errors else "Direct")
        print(f"{'Syntax':<30} {direct_syntax:<15} {socratic_syntax:<15} {winner}")
        
        # Edge cases
        direct_score = f"{result.direct_eval.edge_case_score:.1f}%"
        socratic_score = f"{result.socratic_eval.edge_case_score:.1f}%"
        if abs(result.direct_eval.edge_case_score - result.socratic_eval.edge_case_score) < 5:
            winner = "Tie"
        else:
            winner = "Socratic" if result.socratic_eval.edge_case_score > result.direct_eval.edge_case_score else "Direct"
        print(f"{'Edge Case Coverage':<30} {direct_score:<15} {socratic_score:<15} {winner}")
        
        # Tests passed
        direct_passed = f"{len(result.direct_eval.edge_cases_passed)}/{len(result.direct_eval.edge_cases_passed) + len(result.direct_eval.edge_cases_failed)}"
        socratic_passed = f"{len(result.socratic_eval.edge_cases_passed)}/{len(result.socratic_eval.edge_cases_passed) + len(result.socratic_eval.edge_cases_failed)}"
        print(f"{'Tests Passed':<30} {direct_passed:<15} {socratic_passed:<15}")
        
        print("\n🤖 LLM Qualitative Assessment:")
        print(result.llm_comparison)
        
        print("\n" + "="*70)
        print("⏱️  TIMING")
        print("="*70)
        print(f"Direct Generation:     {result.timing['direct']:.2f}s")
        print(f"Socratic (Total):      {result.timing['socratic_total']:.2f}s")
        print(f"  - Plan:              {result.timing['plan']:.2f}s")
        print(f"  - Debate:            {result.timing['debate']:.2f}s")
        print(f"  - Synthesis:         {result.timing['synthesis']:.2f}s")
        print(f"Evaluation:            {result.timing['evaluation']:.2f}s")
        print(f"---")
        print(f"TOTAL TIME:            {result.timing['total']:.2f}s")
        overhead = result.timing['socratic_total'] - result.timing['direct']
        overhead_pct = (result.timing['socratic_total'] / result.timing['direct'] - 1) * 100
        print(f"Socratic Overhead:     +{overhead:.2f}s ({overhead_pct:.1f}% longer)")
        
        # Final verdict
        print("\n" + "="*70)
        print("🏆 FINAL VERDICT")
        print("="*70)
        
        direct_wins = 0
        socratic_wins = 0
        
        if result.direct_eval.is_complete and not result.socratic_eval.is_complete:
            direct_wins += 1
        elif result.socratic_eval.is_complete and not result.direct_eval.is_complete:
            socratic_wins += 1
            
        if result.direct_eval.edge_case_score > result.socratic_eval.edge_case_score + 5:
            direct_wins += 1
        elif result.socratic_eval.edge_case_score > result.direct_eval.edge_case_score + 5:
            socratic_wins += 1
        
        if socratic_wins > direct_wins:
            print("Winner: SOCRATIC METHOD")
            print("The debate-driven approach produced superior code.")
        elif direct_wins > socratic_wins:
            print("Winner: DIRECT GENERATION")
            print("Direct generation was more effective for this problem.")
        else:
            print("Result: TIE")
            print("Both methods performed similarly.")
    
    def _print_evaluation(self, eval: CodeEvaluation):
        """Helper to print evaluation details"""
        print(f"  Complete: {'✓ Yes' if eval.is_complete else '✗ No (has placeholders/TODOs)'}")
        print(f"  Syntax: {'✓ Valid' if not eval.has_syntax_errors else f'✗ Error: {eval.syntax_error_msg}'}")
        print(f"  Edge Case Score: {eval.edge_case_score:.1f}% ({len(eval.edge_cases_passed)}/{len(eval.edge_cases_passed) + len(eval.edge_cases_failed)} passed)")
        
        if eval.edge_cases_passed:
            print(f"  ✓ Passed tests:")
            for test in eval.edge_cases_passed[:3]:
                print(f"    - {test.test_name}")
            if len(eval.edge_cases_passed) > 3:
                print(f"    ... and {len(eval.edge_cases_passed) - 3} more")
        
        if eval.edge_cases_failed:
            print(f"  ✗ Failed tests:")
            for test in eval.edge_cases_failed[:3]:
                print(f"    - {test.test_name}: {test.error}")
            if len(eval.edge_cases_failed) > 3:
                print(f"    ... and {len(eval.edge_cases_failed) - 3} more")


# USAGE EXAMPLE
if __name__ == "__main__":
    coder = ImprovedSocraticCoder()
    
    # Test problem
    problem = """
Implement a thread-safe LRU (Least Recently Used) cache with TTL (Time To Live).

Requirements:
- Support get(key) and put(key, value) operations
- Maximum capacity that evicts least recently used items when full
- Each entry has a TTL; expired entries should not be returned
- Must be thread-safe for concurrent access
- O(1) time complexity for both operations
"""
    
    # Generate edge case tests
    edge_tests = coder.generate_edge_case_tests(problem)
    
    # If generation failed, use manual tests
    if not edge_tests:
        print("⚠️  Using manual test cases")
        edge_tests = [
            {
                "name": "Get non-existent key",
                "setup": "cache = LRUCacheWithTTL(capacity=5, ttl_seconds=60)",
                "test": "result = cache.get('missing')",
                "result_var": "result",
                "expected": None
            },
            {
                "name": "Basic put and get",
                "setup": "cache = LRUCacheWithTTL(capacity=5, ttl_seconds=60); cache.put('key1', 'value1')",
                "test": "result = cache.get('key1')",
                "result_var": "result",
                "expected": "value1"
            },
        ]
    
    # Run comparison
    result = coder.run_full_comparison(problem, edge_tests)
    coder.print_results(result)
    
    # Save detailed results to multiple files
    print("\n💾 Saving results...")
    
    # 1. Save full debate transcript
    with open("debate_transcript.txt", "w", encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("SOCRATIC DEBATE TRANSCRIPT\n")
        f.write("="*70 + "\n\n")
        f.write("PROBLEM:\n")
        f.write(result.problem + "\n\n")
        f.write("-"*70 + "\n")
        f.write("DEBATE PLAN:\n")
        f.write("-"*70 + "\n")
        f.write(result.socratic_plan + "\n\n")
        f.write("-"*70 + "\n")
        f.write("FULL DEBATE:\n")
        f.write("-"*70 + "\n")
        f.write(result.socratic_debate + "\n")
    
    # 2. Save generated codes
    with open("generated_direct_code.py", "w", encoding='utf-8') as f:
        f.write("# DIRECT GENERATION CODE\n")
        f.write("# Generated without debate\n\n")
        f.write(result.direct_code)
    
    with open("generated_socratic_code.py", "w", encoding='utf-8') as f:
        f.write("# SOCRATIC METHOD CODE\n")
        f.write("# Generated after debate synthesis\n\n")
        f.write(result.socratic_code)
    
    # 3. Save detailed error report
    with open("error_report.txt", "w", encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("DIRECT GENERATION ERRORS:\n")
        f.write("-"*70 + "\n")
        if result.direct_eval.has_syntax_errors:
            f.write(f"❌ SYNTAX ERROR: {result.direct_eval.syntax_error_msg}\n")
        else:
            f.write("✓ No syntax errors\n")
        
        if not result.direct_eval.is_complete:
            f.write("❌ INCOMPLETE: Code contains TODOs or placeholders\n")
        else:
            f.write("✓ Code appears complete\n")
        
        f.write(f"\nEdge Case Results: {len(result.direct_eval.edge_cases_passed)}/{len(result.direct_eval.edge_cases_passed) + len(result.direct_eval.edge_cases_failed)} passed\n\n")
        
        if result.direct_eval.edge_cases_failed:
            f.write("Failed Tests:\n")
            for test in result.direct_eval.edge_cases_failed:
                f.write(f"  • {test.test_name}\n")
                f.write(f"    Error: {test.error}\n")
                if test.expected is not None:
                    f.write(f"    Expected: {test.expected}\n")
                    f.write(f"    Got: {test.actual}\n")
                f.write("\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("SOCRATIC GENERATION ERRORS:\n")
        f.write("-"*70 + "\n")
        if result.socratic_eval.has_syntax_errors:
            f.write(f"❌ SYNTAX ERROR: {result.socratic_eval.syntax_error_msg}\n")
        else:
            f.write("✓ No syntax errors\n")
        
        if not result.socratic_eval.is_complete:
            f.write("❌ INCOMPLETE: Code contains TODOs or placeholders\n")
        else:
            f.write("✓ Code appears complete\n")
        
        f.write(f"\nEdge Case Results: {len(result.socratic_eval.edge_cases_passed)}/{len(result.socratic_eval.edge_cases_passed) + len(result.socratic_eval.edge_cases_failed)} passed\n\n")
        
        if result.socratic_eval.edge_cases_failed:
            f.write("Failed Tests:\n")
            for test in result.socratic_eval.edge_cases_failed:
                f.write(f"  • {test.test_name}\n")
                f.write(f"    Error: {test.error}\n")
                if test.expected is not None:
                    f.write(f"    Expected: {test.expected}\n")
                    f.write(f"    Got: {test.actual}\n")
                f.write("\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("WHY ERRORS OCCUR:\n")
        f.write("-"*70 + "\n")
        f.write("""
Common reasons for errors in generated code:

1. INCOMPLETE IMPLEMENTATION
   - LLM stops generating mid-function
   - Missing method implementations
   - Placeholder code like 'pass' or '# TODO'
   
2. INCORRECT CLASS/FUNCTION NAMES
   - Test expects 'LRUCache' but code defines 'Cache'
   - Mismatch between problem spec and generated code
   
3. MISSING EDGE CASE HANDLING
   - Doesn't check for None/null inputs
   - No validation for invalid parameters
   - Doesn't handle empty containers
   
4. LOGIC ERRORS
   - Wrong algorithm implementation
   - Off-by-one errors
   - Incorrect condition checks
   
5. MISSING IMPORTS
   - Forgot 'import threading' for locks
   - Missing 'from typing import ...'
   
6. CONTEXT LOSS IN SOCRATIC METHOD
   - Debate mentions requirements but synthesis forgets them
   - Important details lost in summarization
        """)
    
    # 4. Save JSON summary
    with open("synthesis_results.json", "w") as f:
        json.dump({
            'problem': result.problem,
            'direct_code': result.direct_code,
            'socratic_code': result.socratic_code,
            'direct_eval': {
                'complete': result.direct_eval.is_complete,
                'has_syntax_errors': result.direct_eval.has_syntax_errors,
                'syntax_error': result.direct_eval.syntax_error_msg,
                'edge_case_score': result.direct_eval.edge_case_score,
                'tests_passed': len(result.direct_eval.edge_cases_passed),
                'tests_failed': len(result.direct_eval.edge_cases_failed),
                'failed_test_names': [t.test_name for t in result.direct_eval.edge_cases_failed]
            },
            'socratic_eval': {
                'complete': result.socratic_eval.is_complete,
                'has_syntax_errors': result.socratic_eval.has_syntax_errors,
                'syntax_error': result.socratic_eval.syntax_error_msg,
                'edge_case_score': result.socratic_eval.edge_case_score,
                'tests_passed': len(result.socratic_eval.edge_cases_passed),
                'tests_failed': len(result.socratic_eval.edge_cases_failed),
                'failed_test_names': [t.test_name for t in result.socratic_eval.edge_cases_failed]
            },
            'timing': result.timing
        }, f, indent=2)
    
    print("  ✓ debate_transcript.txt - Full debate conversation")
    print("  ✓ generated_direct_code.py - Direct generation output")
    print("  ✓ generated_socratic_code.py - Socratic synthesis output")
    print("  ✓ error_report.txt - Detailed error analysis")
    print("  ✓ synthesis_results.json - JSON summary")