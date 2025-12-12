"""
Code Evaluator with Full Test Visibility
Saves test cases and detailed results
"""
import os
import ast
import json
import time
import re
import requests
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

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
    edge_case_score: float
    execution_errors: List[str]

class CodeEvaluator:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        print("🔬 Evaluator initialized")
    
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
        for marker in incomplete_markers:
            if marker.lower() in code_lower:
                return False
        
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
        namespace = {}
        
        try:
            exec(code, namespace)
        except Exception as e:
            return [EdgeCaseResult(
                test_name="Code Execution",
                passed=False,
                error=f"Failed to execute: {str(e)}"
            )]
        
        for test in test_cases:
            try:
                test_name = test['name']
                setup = test.get('setup', '')
                test_code = test['test']
                expected = test.get('expected')
                
                if setup:
                    exec(setup, namespace)
                
                exec(test_code, namespace)
                result_var = test.get('result_var', 'result')
                actual = namespace.get(result_var)
                
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
    
    def evaluate_code(self, code: str, edge_case_tests: List[Dict]) -> CodeEvaluation:
        """Comprehensive code evaluation"""
        has_valid_syntax, syntax_error = self.check_syntax(code)
        is_complete = self.check_completeness(code)
        
        edge_case_results = []
        execution_errors = []
        
        if has_valid_syntax:
            edge_case_results = self.execute_edge_case_tests(code, edge_case_tests)
        else:
            execution_errors.append(f"Syntax error: {syntax_error}")
        
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
    
    def generate_edge_case_tests(self, problem: str) -> List[Dict]:
        """Generate executable test cases"""
        print("\n🧪 Generating edge case tests...")
        
        prompt = f"""Generate 8-10 executable Python test cases for this problem:

{problem}

Format as JSON:
[
  {{
    "name": "Empty input test",
    "setup": "obj = ClassName(capacity=5)",
    "test": "result = obj.get('nonexistent')",
    "result_var": "result",
    "expected": null
  }}
]

ONLY output the JSON array."""
        
        try:
            response = self._make_api_call(prompt, max_tokens=4000, temperature=0.7)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                tests = json.loads(json_match.group(0))
                print(f"  ✓ Generated {len(tests)} test cases")
                
                # SAVE TESTS TO FILE
                self.save_test_cases(tests, "generated_test_cases.json")
                
                return tests
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
        
        return []
    
    def save_test_cases(self, tests: List[Dict], filename: str = "generated_test_cases.json"):
        """Save test cases to a readable file"""
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(tests, f, indent=2)
        
        # Also save as human-readable text
        text_filename = filename.replace('.json', '.txt')
        with open(text_filename, "w", encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("GENERATED TEST CASES\n")
            f.write("="*70 + "\n\n")
            
            for i, test in enumerate(tests, 1):
                f.write(f"TEST {i}: {test['name']}\n")
                f.write("-"*70 + "\n")
                f.write(f"Setup: {test.get('setup', 'None')}\n")
                f.write(f"Test Code: {test['test']}\n")
                f.write(f"Expected Result: {test.get('expected', 'N/A')}\n")
                f.write(f"Result Variable: {test.get('result_var', 'result')}\n")
                f.write("\n")
        
        print(f"  ✓ Saved tests to {filename} and {text_filename}")
    
    def _make_api_call(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Make API call"""
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
        
        response = requests.post(self.api_url, headers=headers, json=data, verify=False, timeout=90)
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"].strip()
    
    def compare_solutions(self, direct_code: str, socratic_code: str, 
                         direct_eval: CodeEvaluation, socratic_eval: CodeEvaluation,
                         problem: str) -> str:
        """LLM-based qualitative comparison"""
        print("\n🤖 Running qualitative comparison...")
        
        prompt = f"""Compare these two solutions qualitatively.

PROBLEM: {problem}

SOLUTION A (Direct):
{direct_code[:800]}
Results: {direct_eval.edge_case_score:.1f}% edge cases passed

SOLUTION B (Socratic):
{socratic_code[:800]}
Results: {socratic_eval.edge_case_score:.1f}% edge cases passed

Analyze:
1. Code quality and readability
2. Design patterns used
3. Error handling approach
4. Completeness of implementation

Keep brief (4-5 sentences)."""
        
        comparison = self._make_api_call(prompt, max_tokens=400, temperature=0.3)
        return comparison
    
    def print_evaluation(self, eval: CodeEvaluation, label: str):
        """Print evaluation details"""
        print(f"\n📊 {label} Evaluation:")
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
    
    def print_comparison(self, direct_eval: CodeEvaluation, socratic_eval: CodeEvaluation):
        """Print side-by-side comparison"""
        print("\n" + "="*70)
        print("📊 SIDE-BY-SIDE COMPARISON")
        print("="*70)
        
        print(f"\n{'Metric':<30} {'Direct':<15} {'Socratic':<15} {'Winner'}")
        print("-" * 70)
        
        # Completeness
        direct_complete = "✓ Complete" if direct_eval.is_complete else "✗ Incomplete"
        socratic_complete = "✓ Complete" if socratic_eval.is_complete else "✗ Incomplete"
        winner = "Tie" if direct_eval.is_complete == socratic_eval.is_complete else \
                 ("Socratic" if socratic_eval.is_complete else "Direct")
        print(f"{'Completeness':<30} {direct_complete:<15} {socratic_complete:<15} {winner}")
        
        # Syntax
        direct_syntax = "✓ Valid" if not direct_eval.has_syntax_errors else "✗ Errors"
        socratic_syntax = "✓ Valid" if not socratic_eval.has_syntax_errors else "✗ Errors"
        winner = "Tie" if direct_eval.has_syntax_errors == socratic_eval.has_syntax_errors else \
                 ("Socratic" if not socratic_eval.has_syntax_errors else "Direct")
        print(f"{'Syntax':<30} {direct_syntax:<15} {socratic_syntax:<15} {winner}")
        
        # Edge cases
        direct_score = f"{direct_eval.edge_case_score:.1f}%"
        socratic_score = f"{socratic_eval.edge_case_score:.1f}%"
        if abs(direct_eval.edge_case_score - socratic_eval.edge_case_score) < 5:
            winner = "Tie"
        else:
            winner = "Socratic" if socratic_eval.edge_case_score > direct_eval.edge_case_score else "Direct"
        print(f"{'Edge Case Coverage':<30} {direct_score:<15} {socratic_score:<15} {winner}")
        
        # Final verdict
        print("\n" + "="*70)
        print("🏆 FINAL VERDICT")
        print("="*70)
        
        direct_wins = 0
        socratic_wins = 0
        
        if direct_eval.is_complete and not socratic_eval.is_complete:
            direct_wins += 1
        elif socratic_eval.is_complete and not direct_eval.is_complete:
            socratic_wins += 1
            
        if direct_eval.edge_case_score > socratic_eval.edge_case_score + 5:
            direct_wins += 1
        elif socratic_eval.edge_case_score > direct_eval.edge_case_score + 5:
            socratic_wins += 1
        
        if socratic_wins > direct_wins:
            print("Winner: SOCRATIC METHOD")
        elif direct_wins > socratic_wins:
            print("Winner: DIRECT GENERATION")
        else:
            print("Result: TIE")
    
    def save_detailed_report(self, problem: str, edge_tests: List[Dict],
                           direct_eval: CodeEvaluation, socratic_eval: CodeEvaluation,
                           filename: str = "detailed_evaluation_report.txt"):
        """Save comprehensive evaluation report"""
        with open(filename, "w", encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("DETAILED EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("PROBLEM:\n")
            f.write("-"*70 + "\n")
            f.write(problem + "\n\n")
            
            f.write("="*70 + "\n")
            f.write("TEST CASES USED\n")
            f.write("="*70 + "\n\n")
            for i, test in enumerate(edge_tests, 1):
                f.write(f"{i}. {test['name']}\n")
                f.write(f"   Setup: {test.get('setup', 'None')}\n")
                f.write(f"   Test: {test['test']}\n")
                f.write(f"   Expected: {test.get('expected', 'N/A')}\n\n")
            
            f.write("="*70 + "\n")
            f.write("DIRECT GENERATION RESULTS\n")
            f.write("="*70 + "\n\n")
            self._write_eval_details(f, direct_eval)
            
            f.write("\n" + "="*70 + "\n")
            f.write("SOCRATIC GENERATION RESULTS\n")
            f.write("="*70 + "\n\n")
            self._write_eval_details(f, socratic_eval)
        
        print(f"  ✓ Saved detailed report to {filename}")
    
    def _write_eval_details(self, f, eval: CodeEvaluation):
        """Write evaluation details to file"""
        f.write(f"Complete: {eval.is_complete}\n")
        f.write(f"Syntax Valid: {not eval.has_syntax_errors}\n")
        if eval.syntax_error_msg:
            f.write(f"Syntax Error: {eval.syntax_error_msg}\n")
        f.write(f"\nEdge Case Score: {eval.edge_case_score:.1f}%\n")
        f.write(f"Tests Passed: {len(eval.edge_cases_passed)}/{len(eval.edge_cases_passed) + len(eval.edge_cases_failed)}\n\n")
        
        if eval.edge_cases_passed:
            f.write("PASSED TESTS:\n")
            for test in eval.edge_cases_passed:
                f.write(f"  ✓ {test.test_name}\n")
                if test.expected is not None:
                    f.write(f"    Expected: {test.expected}, Got: {test.actual}\n")
            f.write("\n")
        
        if eval.edge_cases_failed:
            f.write("FAILED TESTS:\n")
            for test in eval.edge_cases_failed:
                f.write(f"  ✗ {test.test_name}\n")
                f.write(f"    Error: {test.error}\n")
                if test.expected is not None:
                    f.write(f"    Expected: {test.expected}, Got: {test.actual}\n")
            f.write("\n")


if __name__ == "__main__":
    evaluator = CodeEvaluator()
    
    problem = """
Implement a thread-safe LRU (Least Recently Used) cache with TTL (Time To Live).

Requirements:
- Support get(key) and put(key, value) operations
- Maximum capacity that evicts least recently used items when full
- Each entry has a TTL; expired entries should not be returned
- Must be thread-safe for concurrent access
- O(1) time complexity for both operations
"""
    
    # Load generated codes
    try:
        with open("direct_code.py", "r", encoding='utf-8') as f:
            direct_code = f.read()
        with open("socratic_code.py", "r", encoding='utf-8') as f:
            socratic_code = f.read()
    except FileNotFoundError:
        print("❌ Code files not found. Run direct_generator.py and socratic_generator.py first!")
        exit(1)
    
    # Generate or use manual tests
    edge_tests = evaluator.generate_edge_case_tests(problem)
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
        ]
        evaluator.save_test_cases(edge_tests, "manual_test_cases.json")
    
    # Evaluate both
    print("\n" + "="*70)
    print("EVALUATING CODE")
    print("="*70)
    
    direct_eval = evaluator.evaluate_code(direct_code, edge_tests)
    evaluator.print_evaluation(direct_eval, "DIRECT")
    
    socratic_eval = evaluator.evaluate_code(socratic_code, edge_tests)
    evaluator.print_evaluation(socratic_eval, "SOCRATIC")
    
    # Compare
    evaluator.print_comparison(direct_eval, socratic_eval)
    
    # Qualitative comparison
    comparison = evaluator.compare_solutions(direct_code, socratic_code, 
                                            direct_eval, socratic_eval, problem)
    print("\n🤖 Qualitative Assessment:")
    print(comparison)
    
    # Save comprehensive report
    evaluator.save_detailed_report(problem, edge_tests, direct_eval, socratic_eval)