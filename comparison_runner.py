"""
Enhanced LeetCode Comparison Runner
Runs Direct vs Socratic generation on 5 hard LeetCode problems with detailed evaluation
"""
import os
import sys
import time
import json
import ast
import re
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Import from your files
try:
    from leetcode import LEETCODE_HARD_PROBLEMS
    print(f"✅ Loaded {len(LEETCODE_HARD_PROBLEMS)} LeetCode hard problems")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Make sure leetcode.py is in the same directory")
    sys.exit(1)

try:
    from direct_generation import DirectCodeGenerator
    print("✅ Loaded DirectCodeGenerator")
except ImportError as e:
    print(f"❌ Error importing DirectCodeGenerator: {e}")
    print("Make sure direct_generation.py is in the same directory")
    sys.exit(1)

try:
    from socratic_gen import SocraticCodeGenerator
    print("✅ Loaded SocraticCodeGenerator")
except ImportError as e:
    print(f"❌ Error importing SocraticCodeGenerator: {e}")
    print("Make sure socratic_gen.py is in the same directory")
    sys.exit(1)


class EnhancedLeetCodeEvaluator:
    """Enhanced evaluator for LeetCode-style detailed analysis"""
    
    def evaluate_solution(self, code: str, problem_data: Dict) -> Dict[str, Any]:
        """Comprehensive LeetCode-style evaluation"""
        results = {
            'problem_id': problem_data['id'],
            'problem_title': problem_data['title'],
            'syntax_valid': self._check_syntax(code),
            'test_results': self._run_tests(code, problem_data),
            'complexity_analysis': self._analyze_complexity(code, problem_data),
            'edge_case_analysis': self._analyze_edge_cases(code, problem_data),
            'code_quality': self._assess_code_quality(code),
            'scores': self._calculate_scores(code, problem_data),
            'would_pass_leetcode': False
        }
        
        # Determine if would pass LeetCode
        results['would_pass_leetcode'] = self._would_pass_leetcode(results)
        
        return results
    
    def _check_syntax(self, code: str) -> Dict:
        """Check Python syntax"""
        try:
            ast.parse(code)
            return {'valid': True, 'error': None}
        except SyntaxError as e:
            return {'valid': False, 'error': str(e)}
    
    def _run_tests(self, code: str, problem_data: Dict) -> Dict:
        """Run test cases on the code"""
        test_cases = problem_data.get('test_cases', [])
        
        if not test_cases:
            return {'passed': 0, 'failed': 0, 'total': 0, 'all_passed': True, 'details': []}
        
        # Create test code
        test_code = self._create_test_code(code, problem_data, test_cases)
        
        try:
            result = subprocess.run(
                ['python', '-c', test_code],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            passed = 0
            failed = 0
            details = []
            
            # Parse output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'PASSED:' in line:
                    passed += 1
                    details.append({'status': 'PASSED', 'message': line})
                elif 'FAILED:' in line or 'ERROR:' in line:
                    failed += 1
                    details.append({'status': 'FAILED', 'message': line})
            
            return {
                'passed': passed,
                'failed': failed,
                'total': len(test_cases),
                'all_passed': failed == 0,
                'details': details[:10],  # Show first 10
                'error': result.stderr if result.stderr else None
            }
            
        except subprocess.TimeoutExpired:
            return {'passed': 0, 'failed': len(test_cases), 'total': len(test_cases),
                    'all_passed': False, 'details': [], 'error': 'Timeout after 10 seconds'}
        except Exception as e:
            return {'passed': 0, 'failed': len(test_cases), 'total': len(test_cases),
                    'all_passed': False, 'details': [], 'error': str(e)}
    
    def _create_test_code(self, code: str, problem_data: Dict, test_cases: List) -> str:
        """Create executable test code"""
        # Extract function/class name
        func_match = re.search(r'def\s+(\w+)', problem_data.get('function_signature', ''))
        class_match = re.search(r'class\s+(\w+)', problem_data.get('class_signature', ''))
        
        test_code = f"""{code}

# Test runner
import sys
import json
"""
        
        if func_match:
            func_name = func_match.group(1)
            for i, test_case in enumerate(test_cases):
                if 'input' in test_case:
                    inputs = test_case['input']
                    expected = test_case['expected']
                    name = test_case.get('name', f'Test {i+1}')
                    
                    # Build function call
                    if isinstance(inputs, dict):
                        args = ', '.join(f"{k}={repr(v)}" for k, v in inputs.items())
                        test_code += f"""
try:
    result = {func_name}({args})
    if result == {repr(expected)}:
        print(f'PASSED: {{repr({repr(name)})}}')
    else:
        print(f'FAILED: {{repr({repr(name)})}} - Expected {{repr({repr(expected)})}}, got {{repr(result)}}')
except Exception as e:
    print(f'ERROR: {{repr({repr(name)})}} - {{e}}')
"""
        
        return test_code
    
    def _analyze_complexity(self, code: str, problem_data: Dict) -> Dict:
        """Analyze time and space complexity"""
        optimal_time = problem_data.get('optimal_time_complexity', '')
        optimal_space = problem_data.get('optimal_space_complexity', '')
        
        # Detect complexity patterns
        detected_time = self._detect_time_complexity(code)
        detected_space = self._detect_space_complexity(code)
        
        # Check if matches optimal
        time_matches = self._complexity_matches(detected_time, optimal_time)
        space_matches = self._complexity_matches(detected_space, optimal_space)
        
        return {
            'detected_time': detected_time,
            'detected_space': detected_space,
            'optimal_time': optimal_time,
            'optimal_space': optimal_space,
            'time_matches_optimal': time_matches,
            'space_matches_optimal': space_matches,
            'matches_optimal': time_matches and space_matches,
            'explanation': self._generate_complexity_explanation(code)
        }
    
    def _detect_time_complexity(self, code: str) -> str:
        """Detect time complexity from code patterns"""
        code_lower = code.lower()
        
        # Check patterns
        if any(pattern in code_lower for pattern in ['for i in range', 'for j in range', 'while i <', 'while j <']):
            # Check for nested loops
            lines = code.split('\n')
            loop_depths = []
            for line in lines:
                if 'for ' in line or 'while ' in line:
                    indent = len(line) - len(line.lstrip())
                    loop_depths.append(indent)
            
            # Count unique indent levels for loops
            if len(set(loop_depths)) >= 2:
                return 'O(n²)'
            else:
                return 'O(n)'
        
        # Check for binary search patterns
        if any(pattern in code_lower for pattern in ['mid =', 'left < right', 'binary']):
            return 'O(log n)'
        
        # Check for sorting
        if any(pattern in code_lower for pattern in ['sorted(', 'sort()', 'heap', 'queue']):
            return 'O(n log n)'
        
        # Check for recursion/backtracking
        if any(pattern in code_lower for pattern in ['def ', 'fib', 'factorial', 'backtrack']):
            if code_lower.count('def ') > 1:
                return 'O(2ⁿ)'
        
        return 'O(1)'
    
    def _detect_space_complexity(self, code: str) -> str:
        """Detect space complexity from code patterns"""
        code_lower = code.lower()
        
        # Check for data structures
        if any(pattern in code_lower for pattern in 
               ['list(', 'dict(', 'set(', 'heap', 'queue', 'stack', 'array']):
            return 'O(n)'
        
        # Check for recursion
        if any(pattern in code_lower for pattern in ['def ', 'recursive', 'fib', 'factorial']):
            if code_lower.count('def ') > 1:
                return 'O(n)'  # Recursion stack
        
        return 'O(1)'
    
    def _complexity_matches(self, detected: str, optimal: str) -> bool:
        """Check if detected complexity matches optimal"""
        if not optimal:
            return True
        
        optimal_lower = optimal.lower()
        detected_lower = detected.lower()
        
        # Handle different notations
        if 'n²' in optimal_lower or 'n^2' in optimal_lower:
            return 'n²' in detected_lower or 'n^2' in detected_lower or 'quadratic' in detected_lower
        
        if 'log n' in optimal_lower:
            return 'log' in detected_lower
        
        if 'n log n' in optimal_lower:
            return 'n log' in detected_lower or 'log n' in detected_lower
        
        return detected_lower in optimal_lower or optimal_lower in detected_lower
    
    def _generate_complexity_explanation(self, code: str) -> str:
        """Generate explanation for complexity detection"""
        lines = code.split('\n')
        loops = 0
        nested = False
        prev_indent = -1
        
        for line in lines:
            if 'for ' in line or 'while ' in line:
                loops += 1
                indent = len(line) - len(line.lstrip())
                if prev_indent != -1 and indent > prev_indent:
                    nested = True
                prev_indent = indent
        
        if nested and loops > 1:
            return f"Detected {loops} loops with nesting → O(n²)"
        elif loops == 1:
            return f"Detected single loop → O(n)"
        elif loops > 1:
            return f"Detected {loops} sequential loops → O(n)"
        else:
            return "No loops detected → O(1)"
    
    def _analyze_edge_cases(self, code: str, problem_data: Dict) -> Dict:
        """Analyze edge case handling"""
        edge_cases = problem_data.get('key_edge_cases', [])
        code_lower = code.lower()
        
        handled = 0
        details = []
        
        for case in edge_cases[:10]:  # Check first 10
            case_lower = case.lower()
            handled_flag = False
            
            # Check various edge case patterns
            if 'empty' in case_lower:
                if any(pattern in code_lower for pattern in ['if not', 'len(', '== 0', 'is none']):
                    handled_flag = True
            elif 'single' in case_lower:
                if any(pattern in code_lower for pattern in ['len(', '== 1', 'range(1']):
                    handled_flag = True
            elif 'negative' in case_lower:
                if any(pattern in code_lower for pattern in ['< 0', 'negative', 'abs(']):
                    handled_flag = True
            elif 'large' in case_lower or 'maximum' in case_lower:
                if any(pattern in code_lower for pattern in ['range', 'for ', 'while ']):
                    handled_flag = True
            elif 'null' in case_lower or 'none' in case_lower:
                if any(pattern in code_lower for pattern in ['none', 'null', 'is not']):
                    handled_flag = True
            
            if handled_flag:
                handled += 1
                details.append({'case': case, 'handled': True})
            else:
                details.append({'case': case, 'handled': False})
        
        return {
            'total_edge_cases': len(edge_cases),
            'handled_edge_cases': handled,
            'coverage_percentage': (handled / len(edge_cases) * 100) if edge_cases else 100,
            'details': details
        }
    
    def _assess_code_quality(self, code: str) -> Dict:
        """Assess code quality metrics"""
        lines = code.split('\n')
        
        return {
            'line_count': len(lines),
            'has_comments': any(line.strip().startswith('#') for line in lines),
            'has_docstring': '"""' in code or "'''" in code,
            'has_error_handling': any(pattern in code for pattern in ['try:', 'except', 'raise']),
            'has_type_hints': any(pattern in code for pattern in ['->', ': List', ': Dict', ': Optional']),
            'function_count': code.count('def '),
            'class_count': code.count('class '),
            'import_count': code.count('import ') + code.count('from ')
        }
    
    def _calculate_scores(self, code: str, problem_data: Dict) -> Dict:
        """Calculate comprehensive scores"""
        # Run tests to get base score
        test_results = self._run_tests(code, problem_data)
        complexity = self._analyze_complexity(code, problem_data)
        edge_cases = self._analyze_edge_cases(code, problem_data)
        quality = self._assess_code_quality(code)
        
        # Test score (50%)
        test_score = (test_results['passed'] / test_results['total'] * 100) if test_results['total'] > 0 else 0
        
        # Complexity score (20%)
        complexity_score = 100 if complexity['matches_optimal'] else 50
        
        # Edge case score (20%)
        edge_case_score = edge_cases['coverage_percentage']
        
        # Code quality score (10%)
        quality_score = 0
        if quality['has_comments']:
            quality_score += 20
        if quality['has_error_handling']:
            quality_score += 30
        if quality['has_docstring']:
            quality_score += 20
        if quality['has_type_hints']:
            quality_score += 30
        
        overall_score = (
            test_score * 0.5 +
            complexity_score * 0.2 +
            edge_case_score * 0.2 +
            quality_score * 0.1
        )
        
        return {
            'test_score': test_score,
            'complexity_score': complexity_score,
            'edge_case_score': edge_case_score,
            'quality_score': quality_score,
            'overall_score': overall_score
        }
    
    def _would_pass_leetcode(self, results: Dict) -> bool:
        """Determine if solution would pass LeetCode"""
        # LeetCode criteria:
        # 1. All tests must pass
        # 2. Must have valid syntax
        # 3. Should handle edge cases reasonably
        
        if not results['syntax_valid']['valid']:
            return False
        
        if not results['test_results']['all_passed']:
            return False
        
        # Edge case coverage should be > 70%
        if results['edge_case_analysis']['coverage_percentage'] < 70:
            return False
        
        return True
    
    def print_evaluation(self, evaluation: Dict, prefix: str = ""):
        """Print detailed evaluation results"""
        print(f"\n{prefix}📊 DETAILED EVALUATION:")
        print(f"{prefix}{'='*50}")
        
        # Test Results
        print(f"{prefix}🧪 TEST RESULTS:")
        print(f"{prefix}  Passed: {evaluation['test_results']['passed']}/{evaluation['test_results']['total']}")
        if evaluation['test_results']['failed'] > 0:
            print(f"{prefix}  Failed: {evaluation['test_results']['failed']}")
        
        # Complexity Analysis
        print(f"{prefix}⚡ COMPLEXITY ANALYSIS:")
        print(f"{prefix}  Detected Time: {evaluation['complexity_analysis']['detected_time']}")
        print(f"{prefix}  Optimal Time: {evaluation['complexity_analysis']['optimal_time']}")
        print(f"{prefix}  Matches Optimal: {'✅' if evaluation['complexity_analysis']['matches_optimal'] else '❌'}")
        print(f"{prefix}  Explanation: {evaluation['complexity_analysis']['explanation']}")
        
        # Edge Case Analysis
        print(f"{prefix}🎯 EDGE CASE ANALYSIS:")
        print(f"{prefix}  Handled: {evaluation['edge_case_analysis']['handled_edge_cases']}/{evaluation['edge_case_analysis']['total_edge_cases']}")
        print(f"{prefix}  Coverage: {evaluation['edge_case_analysis']['coverage_percentage']:.1f}%")
        
        # Code Quality
        print(f"{prefix}✨ CODE QUALITY:")
        qual = evaluation['code_quality']
        print(f"{prefix}  Lines: {qual['line_count']}")
        print(f"{prefix}  Functions: {qual['function_count']}")
        print(f"{prefix}  Comments: {'✅' if qual['has_comments'] else '❌'}")
        print(f"{prefix}  Error Handling: {'✅' if qual['has_error_handling'] else '❌'}")
        
        # Scores
        print(f"{prefix}🏆 SCORES:")
        scores = evaluation['scores']
        print(f"{prefix}  Test Score: {scores['test_score']:.1f}%")
        print(f"{prefix}  Complexity Score: {scores['complexity_score']:.1f}%")
        print(f"{prefix}  Edge Case Score: {scores['edge_case_score']:.1f}%")
        print(f"{prefix}  Overall Score: {scores['overall_score']:.1f}%")
        
        # LeetCode Readiness
        print(f"{prefix}🏁 LEETCODE READINESS:")
        print(f"{prefix}  Would Pass LeetCode: {'✅ YES' if evaluation['would_pass_leetcode'] else '❌ NO'}")


def run_comparison():
    """Run comparison with enhanced evaluation"""
    print("🏁 LEETCODE HARD PROBLEMS COMPARISON - ENHANCED")
    print("="*80)
    
    # Initialize components
    direct_gen = DirectCodeGenerator()
    socratic_gen = SocraticCodeGenerator()
    evaluator = EnhancedLeetCodeEvaluator()
    
    results = {}
    detailed_results = []
    
    for problem in LEETCODE_HARD_PROBLEMS:
        problem_id = problem["id"]
        title = problem["title"]
        
        print(f"\n{'='*80}")
        print(f"PROBLEM {problem_id}: {title}")
        print(f"Difficulty: {problem['difficulty']}")
        print(f"{'='*80}")
        
        problem_result = {
            'problem_id': problem_id,
            'title': title,
            'direct': {},
            'socratic': {},
            'comparison': {}
        }
        
        # 1. DIRECT GENERATION
        print("\n1️⃣  DIRECT GENERATION")
        print("-"*40)
        try:
            direct_start = time.time()
            direct_code, direct_time = direct_gen.generate_for_problem(problem)
            direct_elapsed = time.time() - direct_start
            
            # Save
            direct_gen.save_result(direct_code, problem_id)
            
            print(f"  Time: {direct_elapsed:.1f}s")
            print(f"  Code length: {len(direct_code)} chars")
            
            # Evaluate
            print(f"  Evaluating...")
            direct_eval = evaluator.evaluate_solution(direct_code, problem)
            evaluator.print_evaluation(direct_eval, "  ")
            
            problem_result['direct'] = {
                'generation_time': direct_elapsed,
                'code_length': len(direct_code),
                'evaluation': direct_eval,
                'filename': f"direct_{problem_id}.py"
            }
            
        except Exception as e:
            print(f"  ❌ Direct generation failed: {e}")
            problem_result['direct']['error'] = str(e)
        
        time.sleep(5)  # Cooldown
        
        # 2. SOCRATIC GENERATION
        print("\n2️⃣  SOCRATIC GENERATION")
        print("-"*40)
        try:
            socratic_start = time.time()
            socratic_results, socratic_total_time = socratic_gen.generate_for_problem(problem)
            socratic_code = socratic_results.get('code', '')
            socratic_elapsed = time.time() - socratic_start
            
            # Save
            socratic_gen.save_results(socratic_results, problem_id)
            
            print(f"  Time: {socratic_elapsed:.1f}s")
            print(f"  Code length: {len(socratic_code)} chars")
            print(f"  Debate length: {len(socratic_results.get('debate', ''))} chars")
            
            # Evaluate
            print(f"  Evaluating...")
            socratic_eval = evaluator.evaluate_solution(socratic_code, problem)
            evaluator.print_evaluation(socratic_eval, "  ")
            
            problem_result['socratic'] = {
                'generation_time': socratic_elapsed,
                'total_time': socratic_total_time,
                'code_length': len(socratic_code),
                'debate_length': len(socratic_results.get('debate', '')),
                'evaluation': socratic_eval,
                'code_filename': f"socratic_{problem_id}.py",
                'debate_filename': f"debate_{problem_id}.txt"
            }
            
        except Exception as e:
            print(f"  ❌ Socratic generation failed: {e}")
            problem_result['socratic']['error'] = str(e)
        
        # 3. COMPARE RESULTS
        print("\n3️⃣  COMPARISON")
        print("-"*40)
        
        if 'evaluation' in problem_result['direct'] and 'evaluation' in problem_result['socratic']:
            direct_eval = problem_result['direct']['evaluation']
            socratic_eval = problem_result['socratic']['evaluation']
            direct_time = problem_result['direct']['generation_time']
            socratic_time = problem_result['socratic']['generation_time']
            
            # Calculate comparison metrics
            time_ratio = socratic_time / direct_time if direct_time > 0 else float('inf')
            score_diff = socratic_eval['scores']['overall_score'] - direct_eval['scores']['overall_score']
            
            # Determine winner
            winner = "tie"
            if score_diff > 5:
                winner = "socratic"
            elif score_diff < -5:
                winner = "direct"
            
            comparison = {
                'time_direct': direct_time,
                'time_socratic': socratic_time,
                'time_ratio': time_ratio,
                'score_direct': direct_eval['scores']['overall_score'],
                'score_socratic': socratic_eval['scores']['overall_score'],
                'score_difference': score_diff,
                'tests_direct': f"{direct_eval['test_results']['passed']}/{direct_eval['test_results']['total']}",
                'tests_socratic': f"{socratic_eval['test_results']['passed']}/{socratic_eval['test_results']['total']}",
                'complexity_match_direct': direct_eval['complexity_analysis']['matches_optimal'],
                'complexity_match_socratic': socratic_eval['complexity_analysis']['matches_optimal'],
                'edge_coverage_direct': direct_eval['edge_case_analysis']['coverage_percentage'],
                'edge_coverage_socratic': socratic_eval['edge_case_analysis']['coverage_percentage'],
                'would_pass_direct': direct_eval['would_pass_leetcode'],
                'would_pass_socratic': socratic_eval['would_pass_leetcode'],
                'winner': winner
            }
            
            problem_result['comparison'] = comparison
            
            # Print comparison
            print(f"  Time: Direct {direct_time:.1f}s vs Socratic {socratic_time:.1f}s")
            print(f"  Time Ratio: {time_ratio:.1f}x")
            print(f"  Score: Direct {direct_eval['scores']['overall_score']:.1f}% vs Socratic {socratic_eval['scores']['overall_score']:.1f}%")
            print(f"  Score Difference: {score_diff:+.1f}%")
            print(f"  Tests: Direct {comparison['tests_direct']} vs Socratic {comparison['tests_socratic']}")
            print(f"  Complexity Match: Direct {'✅' if comparison['complexity_match_direct'] else '❌'} vs Socratic {'✅' if comparison['complexity_match_socratic'] else '❌'}")
            print(f"  Edge Coverage: Direct {comparison['edge_coverage_direct']:.1f}% vs Socratic {comparison['edge_coverage_socratic']:.1f}%")
            print(f"  Would Pass LeetCode: Direct {'✅' if comparison['would_pass_direct'] else '❌'} vs Socratic {'✅' if comparison['would_pass_socratic'] else '❌'}")
            print(f"  🏆 WINNER: {winner.upper()}")
        
        results[problem_id] = problem_result
        detailed_results.append(problem_result)
        
        # Rate limiting
        print(f"\n⏳ Waiting before next problem (10s)...")
        time.sleep(10)
    
    # GENERATE FINAL REPORT
    generate_final_report(detailed_results)
    
    return results


def generate_final_report(detailed_results: List[Dict]):
    """Generate comprehensive final report"""
    report_filename = "leetcode_comparison_detailed_report.txt"
    
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("LEETCODE HARD PROBLEMS COMPARISON - DETAILED REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total problems tested: {len(detailed_results)}\n\n")
        
        # Summary Statistics
        f.write("SUMMARY STATISTICS:\n")
        f.write("-"*80 + "\n")
        
        direct_wins = sum(1 for r in detailed_results if r.get('comparison', {}).get('winner') == 'direct')
        socratic_wins = sum(1 for r in detailed_results if r.get('comparison', {}).get('winner') == 'socratic')
        ties = sum(1 for r in detailed_results if r.get('comparison', {}).get('winner') == 'tie')
        
        f.write(f"Direct wins: {direct_wins}\n")
        f.write(f"Socratic wins: {socratic_wins}\n")
        f.write(f"Ties: {ties}\n\n")
        
        # Calculate averages
        direct_scores = []
        socratic_scores = []
        time_ratios = []
        
        for result in detailed_results:
            if 'comparison' in result:
                comp = result['comparison']
                direct_scores.append(comp.get('score_direct', 0))
                socratic_scores.append(comp.get('score_socratic', 0))
                time_ratios.append(comp.get('time_ratio', 0))
        
        if direct_scores:
            f.write(f"Average Direct Score: {sum(direct_scores)/len(direct_scores):.1f}%\n")
            f.write(f"Average Socratic Score: {sum(socratic_scores)/len(socratic_scores):.1f}%\n")
            f.write(f"Average Time Ratio (Socratic/Direct): {sum(time_ratios)/len(time_ratios):.1f}x\n\n")
        
        # Detailed Results Per Problem
        f.write("DETAILED RESULTS PER PROBLEM:\n")
        f.write("="*80 + "\n")
        
        for result in detailed_results:
            f.write(f"\nProblem: {result['title']} (ID: {result['problem_id']})\n")
            f.write("-"*80 + "\n")
            
            if 'comparison' in result:
                comp = result['comparison']
                f.write(f"Winner: {comp.get('winner', 'unknown').upper()}\n")
                f.write(f"Direct Score: {comp.get('score_direct', 0):.1f}%\n")
                f.write(f"Socratic Score: {comp.get('score_socratic', 0):.1f}%\n")
                f.write(f"Time Ratio: {comp.get('time_ratio', 0):.1f}x\n\n")
                
                # Direct details
                if 'evaluation' in result['direct']:
                    eval_d = result['direct']['evaluation']
                    f.write("Direct Evaluation:\n")
                    f.write(f"  Tests: {eval_d['test_results']['passed']}/{eval_d['test_results']['total']}\n")
                    f.write(f"  Complexity: {eval_d['complexity_analysis']['detected_time']} (Optimal: {eval_d['complexity_analysis']['optimal_time']})\n")
                    f.write(f"  Edge Coverage: {eval_d['edge_case_analysis']['coverage_percentage']:.1f}%\n")
                    f.write(f"  Would Pass LeetCode: {'Yes' if eval_d['would_pass_leetcode'] else 'No'}\n")
                
                # Socratic details
                if 'evaluation' in result['socratic']:
                    eval_s = result['socratic']['evaluation']
                    f.write("\nSocratic Evaluation:\n")
                    f.write(f"  Tests: {eval_s['test_results']['passed']}/{eval_s['test_results']['total']}\n")
                    f.write(f"  Complexity: {eval_s['complexity_analysis']['detected_time']} (Optimal: {eval_s['complexity_analysis']['optimal_time']})\n")
                    f.write(f"  Edge Coverage: {eval_s['edge_case_analysis']['coverage_percentage']:.1f}%\n")
                    f.write(f"  Would Pass LeetCode: {'Yes' if eval_s['would_pass_leetcode'] else 'No'}\n")
            
            f.write("\n")
        
        # Recommendations
        f.write("\nOVERALL RECOMMENDATIONS:\n")
        f.write("="*80 + "\n")
        
        if socratic_wins > direct_wins:
            f.write("✅ SOCRATIC METHOD IS BETTER for hard LeetCode problems\n")
            f.write(f"   Wins by {socratic_wins}-{direct_wins}-{ties}\n")
            f.write("   The debate process helps catch edge cases and optimize solutions\n")
        elif direct_wins > socratic_wins:
            f.write("✅ DIRECT GENERATION IS BETTER for hard LeetCode problems\n")
            f.write(f"   Wins by {direct_wins}-{socratic_wins}-{ties}\n")
            f.write("   Faster generation with similar quality\n")
        else:
            f.write("⚖️  BOTH METHODS ARE COMPARABLE for hard LeetCode problems\n")
            f.write("   Consider using Direct for speed, Socratic for critical code\n")
        
        f.write("\nKEY INSIGHTS:\n")
        f.write("-"*80 + "\n")
        f.write("1. Problems where Socratic excels: Complex algorithms with many edge cases\n")
        f.write("2. Problems where Direct excels: Straightforward implementations\n")
        f.write("3. Time trade-off: Socratic is typically slower but may produce better code\n")
        f.write("4. For production code, consider Socratic method for critical components\n")
        f.write("5. For quick prototyping, Direct generation is more efficient\n")
    
    print(f"\n✅ Detailed report saved to {report_filename}")
    
    # Also save JSON for programmatic access
    with open("detailed_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, default=str)
    print(f"✅ JSON results saved to detailed_comparison_results.json")


if __name__ == "__main__":
    # Run the comparison
    results = run_comparison()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print("\n📁 Generated files:")
    print("  - direct_*.py - Direct generation code")
    print("  - socratic_*.py - Socratic generation code")
    print("  - debate_*.txt - Debate transcripts")
    print("  - leetcode_comparison_detailed_report.txt - Detailed report")
    print("  - detailed_comparison_results.json - Full results in JSON")