"""
Fixed LeetCode Comparison Runner
Properly evaluates Direct vs Socratic generation with accurate test execution
"""
import os
import sys
import time
import json
import ast
import re
import traceback
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Import from your files
try:
    from leetcode import LEETCODE_HARD_PROBLEMS
    print(f"✅ Loaded {len(LEETCODE_HARD_PROBLEMS)} LeetCode hard problems")
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

try:
    from direct_generation import DirectCodeGenerator
    print("✅ Loaded DirectCodeGenerator")
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

try:
    from socratic_gen import SocraticCodeGenerator
    print("✅ Loaded SocraticCodeGenerator")
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)


class FixedLeetCodeEvaluator:
    """Fixed evaluator with accurate test execution"""
    
    def evaluate_solution(self, code: str, problem_data: Dict) -> Dict[str, Any]:
        """Comprehensive evaluation with actual test execution"""
        results = {
            'problem_id': problem_data['id'],
            'problem_title': problem_data['title'],
            'syntax_valid': self._check_syntax(code),
            'test_results': self._run_tests_safely(code, problem_data),
            'complexity_analysis': self._analyze_complexity(code, problem_data),
            'edge_case_analysis': self._analyze_edge_cases(code, problem_data),
            'code_quality': self._assess_code_quality(code),
            'scores': {},
            'would_pass_leetcode': False
        }
        
        # Calculate scores after all metrics are computed
        results['scores'] = self._calculate_scores(results)
        results['would_pass_leetcode'] = self._would_pass_leetcode(results)
        
        return results
    
    def _check_syntax(self, code: str) -> Dict:
        """Check Python syntax"""
        try:
            ast.parse(code)
            return {'valid': True, 'error': None}
        except SyntaxError as e:
            return {'valid': False, 'error': str(e)}
        except Exception as e:
            return {'valid': False, 'error': f'Parse error: {str(e)}'}
    
    def _run_tests_safely(self, code: str, problem_data: Dict) -> Dict:
        """Run tests by executing code in isolated namespace"""
        test_cases = problem_data.get('test_cases', [])
        
        if not test_cases:
            return {
                'passed': 0, 
                'failed': 0, 
                'total': 0, 
                'all_passed': True, 
                'details': [],
                'error': None
            }
         
        # Execute code to get the function/class
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return {
                'passed': 0,
                'failed': len(test_cases),
                'total': len(test_cases),
                'all_passed': False,
                'details': [{'status': 'ERROR', 'message': f'Code execution failed: {str(e)}'}],
                'error': str(e)
            }
        
        # Find the main function or class
        func_name = self._extract_function_name(code, problem_data)
        class_name = self._extract_class_name(code)
        
        if not func_name and not class_name:
            return {
                'passed': 0,
                'failed': len(test_cases),
                'total': len(test_cases),
                'all_passed': False,
                'details': [{'status': 'ERROR', 'message': 'No function or class found'}],
                'error': 'No callable found'
            }
        
        # Run tests
        passed = 0
        failed = 0
        details = []
        
        for i, test_case in enumerate(test_cases):
            try:
                if func_name and func_name in namespace:
                    result = self._run_function_test(namespace[func_name], test_case, i)
                elif class_name and class_name in namespace:
                    result = self._run_class_test(namespace[class_name], test_case, i)
                else:
                    result = {'status': 'ERROR', 'message': 'Function/class not found in namespace'}
                
                if result['status'] == 'PASSED':
                    passed += 1
                else:
                    failed += 1
                details.append(result)
                
            except Exception as e:
                failed += 1
                details.append({
                    'status': 'ERROR',
                    'message': f"Test {i+1}: {str(e)}",
                    'traceback': traceback.format_exc()
                })
        
        return {
            'passed': passed,
            'failed': failed,
            'total': len(test_cases),
            'all_passed': failed == 0 and passed > 0,
            'details': details[:10],  # Limit to first 10
            'error': None
        }
    
    def _extract_function_name(self, code: str, problem_data: Dict) -> Optional[str]:
        """Extract main function name from code"""
        # Try to get from problem data first
        func_sig = problem_data.get('function_signature', '')
        if func_sig:
            match = re.search(r'def\s+(\w+)', func_sig)
            if match:
                return match.group(1)
        
        # Extract from code - get the first non-main function
        matches = re.findall(r'def\s+(\w+)', code)
        for func in matches:
            if func != 'main' and not func.startswith('_'):
                return func
        
        return matches[0] if matches else None
    
    def _extract_class_name(self, code: str) -> Optional[str]:
        """Extract main class name from code"""
        match = re.search(r'class\s+(\w+)', code)
        return match.group(1) if match else None
    
    def _run_function_test(self, func, test_case: Dict, test_num: int) -> Dict:
        """Run a single function test"""
        try:
            inputs = test_case.get('input', {})
            expected = test_case.get('expected')
            name = test_case.get('name', f'Test {test_num + 1}')
            
            # Handle different input formats
            if isinstance(inputs, dict):
                # Named parameters
                result = func(**inputs)
            elif isinstance(inputs, (list, tuple)):
                # Positional parameters
                result = func(*inputs)
            else:
                # Single parameter
                result = func(inputs)
            
            # Compare results
            if self._results_equal(result, expected):
                return {
                    'status': 'PASSED',
                    'message': f'{name}: ✓',
                    'input': inputs,
                    'expected': expected,
                    'actual': result
                }
            else:
                return {
                    'status': 'FAILED',
                    'message': f'{name}: Expected {expected}, got {result}',
                    'input': inputs,
                    'expected': expected,
                    'actual': result
                }
                
        except Exception as e:
            return {
                'status': 'ERROR',
                'message': f'{name}: {str(e)}',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _run_class_test(self, cls, test_case: Dict, test_num: int) -> Dict:
        """Run a single class-based test (like MedianFinder)"""
        try:
            operations = test_case.get('operations', [])
            args = test_case.get('args', [])
            expected = test_case.get('expected', [])
            name = test_case.get('name', f'Test {test_num + 1}')
            
            obj = None
            results = []
            
            for i, (op, arg_list) in enumerate(zip(operations, args)):
                if i == 0:
                    # First operation is constructor
                    obj = cls(*arg_list) if arg_list else cls()
                    results.append(None)
                else:
                    # Call method
                    if hasattr(obj, op):
                        method = getattr(obj, op)
                        result = method(*arg_list) if arg_list else method()
                        results.append(result)
                    else:
                        return {
                            'status': 'ERROR',
                            'message': f'{name}: Method {op} not found'
                        }
            
            if self._results_equal(results, expected):
                return {
                    'status': 'PASSED',
                    'message': f'{name}: ✓',
                    'expected': expected,
                    'actual': results
                }
            else:
                return {
                    'status': 'FAILED',
                    'message': f'{name}: Expected {expected}, got {results}',
                    'expected': expected,
                    'actual': results
                }
                
        except Exception as e:
            return {
                'status': 'ERROR',
                'message': f'{name}: {str(e)}',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _results_equal(self, actual, expected) -> bool:
        """Compare results with tolerance for floating point"""
        if isinstance(actual, float) and isinstance(expected, (int, float)):
            return abs(actual - expected) < 1e-5
        
        if isinstance(actual, list) and isinstance(expected, list):
            if len(actual) != len(expected):
                return False
            return all(self._results_equal(a, e) for a, e in zip(actual, expected))
        
        return actual == expected
    
    def _analyze_complexity(self, code: str, problem_data: Dict) -> Dict:
        """Analyze time and space complexity"""
        optimal_time = problem_data.get('optimal_time_complexity', '')
        optimal_space = problem_data.get('optimal_space_complexity', '')
        
        # Use AST analysis for more accurate detection
        ast_analysis = self._analyze_with_ast(code)
        if ast_analysis:
            detected_time = ast_analysis.get('time_complexity', 'O(1)')
            detected_space = ast_analysis.get('space_complexity', 'O(1)')
        else:
            # Fallback to pattern matching
            detected_time = self._detect_time_complexity(code)
            detected_space = self._detect_space_complexity(code)
        
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
            'explanation': self._generate_complexity_explanation(code, detected_time, detected_space)
        }
    
    def _analyze_with_ast(self, code: str) -> Optional[Dict]:
        """Use AST parsing for more accurate complexity analysis"""
        try:
            import ast
            
            class ComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.max_nesting = 0
                    self.current_nesting = 0
                    self.loop_vars = []
                    self.has_sorting = False
                    self.has_recursion = False
                    self.func_name = None
                    self.data_structures = set()
                    self.recursive_calls = 0
                    
                def visit_For(self, node):
                    self.current_nesting += 1
                    self.max_nesting = max(self.max_nesting, self.current_nesting)
                    self.generic_visit(node)
                    self.current_nesting -= 1
                    
                def visit_While(self, node):
                    self.current_nesting += 1
                    self.max_nesting = max(self.max_nesting, self.current_nesting)
                    self.generic_visit(node)
                    self.current_nesting -= 1
                    
                def visit_Call(self, node):
                    # Check for sorting operations
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ['sort', 'sorted']:
                            self.has_sorting = True
                    elif isinstance(node.func, ast.Name):
                        if node.func.id in ['sorted', 'heapq', 'bisect']:
                            self.has_sorting = True
                        # Check for recursive calls
                        if self.func_name and node.func.id == self.func_name:
                            self.has_recursion = True
                            self.recursive_calls += 1
                    self.generic_visit(node)
                    
                def visit_FunctionDef(self, node):
                    if not self.func_name:
                        self.func_name = node.name
                    self.generic_visit(node)
                    
                def visit_Assign(self, node):
                    # Check for data structure assignments
                    if isinstance(node.value, (ast.List, ast.Dict, ast.Set)):
                        self.data_structures.add(type(node.value).__name__)
                    self.generic_visit(node)
            
            tree = ast.parse(code)
            visitor = ComplexityVisitor()
            visitor.visit(tree)
            
            # Determine time complexity based on AST analysis
            time_complexity = self._determine_time_from_ast(visitor)
            space_complexity = self._determine_space_from_ast(visitor)
            
            return {
                'time_complexity': time_complexity,
                'space_complexity': space_complexity,
                'ast_metrics': {
                    'max_nesting': visitor.max_nesting,
                    'has_sorting': visitor.has_sorting,
                    'has_recursion': visitor.has_recursion,
                    'recursive_calls': visitor.recursive_calls,
                    'data_structures': list(visitor.data_structures)
                }
            }
            
        except:
            return None
    
    def _determine_time_from_ast(self, visitor) -> str:
        """Determine time complexity from AST metrics"""
        # Check for specific algorithm patterns first
        if visitor.has_recursion:
            if visitor.recursive_calls >= 2:
                # Multiple recursive calls (like fibonacci)
                return 'O(2ⁿ)'
            elif visitor.max_nesting > 0:
                # Recursive with loops
                if visitor.max_nesting == 1:
                    return 'O(n log n)' if visitor.has_sorting else 'O(n)'
                elif visitor.max_nesting == 2:
                    return 'O(n²)'
            return 'O(n)'  # Single recursion like factorial
        
        # Check for sorting operations
        if visitor.has_sorting:
            if visitor.max_nesting == 0:
                return 'O(n log n)'
            elif visitor.max_nesting == 1:
                return 'O(n log n)'  # Sort + single loop
            elif visitor.max_nesting == 2:
                return 'O(n² log n)'  # Sort + nested loops
        
        # Determine by loop nesting depth
        if visitor.max_nesting >= 3:
            return 'O(n³)'
        elif visitor.max_nesting == 2:
            return 'O(n²)'
        elif visitor.max_nesting == 1:
            return 'O(n)'
        else:
            return 'O(1)'
    
    def _determine_space_from_ast(self, visitor) -> str:
        """Determine space complexity from AST metrics"""
        if visitor.has_recursion:
            # Recursion uses stack space
            return 'O(n)'
        elif visitor.data_structures:
            # If using lists, dicts, or sets that scale with input
            return 'O(n)'
        else:
            return 'O(1)'
    
    def _detect_time_complexity(self, code: str) -> str:
        """Detect time complexity from code patterns"""
        code_lower = code.lower()
        lines = code.split('\n')
        
        # Improved loop nesting depth detection
        max_nesting = 0
        current_nesting = 0
        indent_stack = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
                
            current_indent = len(line) - len(line.lstrip())
            
            # Check if we've dedented
            while indent_stack and current_indent <= indent_stack[-1]:
                indent_stack.pop()
                current_nesting = max(0, current_nesting - 1)
            
            # Detect loops
            if re.match(r'^\s*(for|while)\b', stripped):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
                indent_stack.append(current_indent)
        
        # Check for specific algorithm patterns
        
        # Binary search pattern (improved)
        binary_search_patterns = [
            r'while\s+[a-z]+\s*<=\s*[a-z]+:',
            r'mid\s*=\s*\([a-z]+\s*\+\s*[a-z]+\)\s*//\s*2',
            r'[a-z]+\s*=\s*mid\s*[+-]\s*1'
        ]
        if all(re.search(pattern, code_lower) for pattern in binary_search_patterns[:2]):
            return 'O(log n)'
        
        # Check for sorting/heap operations
        if re.search(r'\.sort\s*\(|sorted\s*\(|heapq\.', code_lower):
            if max_nesting >= 1:
                return 'O(n log n)'
            return 'O(n log n)'
        
        # Check for matrix operations
        if re.search(r'for\s+.*\s+in\s+range\s*\(.*\)\s*:.*\n\s*for\s+.*\s+in\s+range\s*\(.*\)', code):
            if max_nesting == 2:
                return 'O(n²)'
            elif max_nesting == 3:
                return 'O(n³)'
        
        # Check for divide and conquer (like merge sort)
        if re.search(r'def\s+\w+\s*\(.*\):.*\n\s*return\s+\w+\s*\(\s*\w+\s*\[:\s*mid\s*\]\s*\)\s*\+\s*\w+\s*\(\s*\w+\s*\[\s*mid\s*:\s*\]\s*\)', code, re.DOTALL):
            return 'O(n log n)'
        
        # Check for two pointers/sliding window
        if re.search(r'left\s*=\s*0.*right\s*=\s*0|start\s*=\s*0.*end\s*=\s*0', code_lower):
            if max_nesting == 1:
                return 'O(n)'
        
        # Nested loops
        if max_nesting >= 3:
            return 'O(n³)'
        elif max_nesting >= 2:
            return 'O(n²)'
        elif max_nesting >= 1:
            return 'O(n)'
        
        # Recursive patterns (improved detection)
        func_match = re.search(r'def\s+(\w+)\s*\(', code)
        if func_match:
            func_name = func_match.group(1)
            # Count recursive calls more accurately
            recursive_pattern = rf'{func_name}\s*\(\s*.*[nN]\s*[-+]\s*\d+.*\)'
            if re.search(recursive_pattern, code):
                # Check for memoization
                if re.search(r'memo|dp\s*\[|@lru_cache|@cache', code_lower):
                    return 'O(n)'
                # Check for multiple recursive calls (like fibonacci)
                if code.count(func_name) > 3:
                    return 'O(2ⁿ)'
                return 'O(n)'
        
        return 'O(1)'
    
    def _detect_space_complexity(self, code: str) -> str:
        """Detect space complexity from code patterns"""
        code_lower = code.lower()
        
        # Check for data structures that scale with input
        # Improved pattern matching
        scaling_patterns = [
            # List comprehensions that iterate over input
            r'\[\s*.*\s*for\s+.*\s+in\s+\w+\s*\]',
            # Dict comprehensions
            r'\{\s*.*\s*:\s*.*\s*for\s+.*\s+in\s+\w+\s*\}',
            # Explicit list/dict creation with input size
            r'\[\s*\]\s*\*\s*\w+',
            r'\w+\s*=\s*\[\s*\]\s*;\s*for',
            r'\w+\.append\(|\w+\.extend\(',
            # Common scaling data structures
            r'defaultdict|Counter|deque|heapq',
            # Matrix/2D array creation
            r'\[\s*\[\s*[^\]]*\s*\]\s*for\s+.*\s+in\s+range',
        ]
        
        if any(re.search(pattern, code_lower, re.IGNORECASE) for pattern in scaling_patterns):
            return 'O(n)'
        
        # Check for recursion depth
        func_match = re.search(r'def\s+(\w+)\s*\(', code)
        if func_match:
            func_name = func_match.group(1)
            # Look for recursive calls with decreasing parameters
            recursive_calls = re.findall(rf'{func_name}\s*\(\s*.*[nN]\s*[-+]\s*\d+.*\)', code)
            if recursive_calls:
                return 'O(n)'  # Recursion stack
        
        # Check for BFS/DFS with queue/stack
        if re.search(r'queue\.|deque\(|stack\s*=\s*\[', code_lower):
            return 'O(n)'
        
        return 'O(1)'
    
    def _complexity_matches(self, detected: str, optimal: str) -> bool:
        """Check if detected complexity matches optimal"""
        if not optimal:
            return True
        
        # Normalize notations
        def normalize(comp: str) -> str:
            comp = comp.strip().lower()
            comp = re.sub(r'[²³^]', '', comp)  # Remove exponents
            comp = re.sub(r'\s+', ' ', comp)   # Normalize spaces
            comp = re.sub(r'o\((\d+)\)', r'o(1)', comp)  # Any constant to O(1)
            
            # Handle equivalent notations
            equivalences = {
                r'o\(n2\)': 'o(n^2)',
                r'o\(n\s*\*\s*2\)': 'o(n^2)',
                r'o\(n\s*log\s*n\)': 'o(n log n)',
                r'o\(log\s*n\)': 'o(log n)',
                r'o\(2\s*n\)': 'o(2^n)',
                r'o\(n!\s*\)': 'o(n!)',
                r'o\(n\s*\*\s*\*\s*2\)': 'o(n^2)',
            }
            
            for pattern, replacement in equivalences.items():
                comp = re.sub(pattern, replacement, comp)
            
            return comp
        
        detected_norm = normalize(detected)
        optimal_norm = normalize(optimal)
        
        # Direct match
        if detected_norm == optimal_norm:
            return True
        
        # Handle ranges (e.g., O(n) matches O(n log n) in some contexts)
        complexity_order = ['o(1)', 'o(log n)', 'o(n)', 'o(n log n)', 'o(n^2)', 'o(n^3)', 'o(2^n)', 'o(n!)']
        
        try:
            detected_idx = complexity_order.index(detected_norm)
            optimal_idx = complexity_order.index(optimal_norm)
            
            # Allow one level of difference for practical purposes
            if abs(detected_idx - optimal_idx) <= 1:
                return True
        except ValueError:
            pass
        
        # Partial matches
        if detected_norm in optimal_norm or optimal_norm in detected_norm:
            return True
        
        return False
    
    def _generate_complexity_explanation(self, code: str, detected_time: str = None, detected_space: str = None) -> str:
        """Generate explanation for complexity detection"""
        if not detected_time:
            detected_time = self._detect_time_complexity(code)
        if not detected_space:
            detected_space = self._detect_space_complexity(code)
        
        lines = code.split('\n')
        loop_count = sum(1 for line in lines if re.match(r'^\s*(for|while)\b', line.strip()))
        func_count = code.count('def ')
        
        explanations = []
        
        # Time complexity explanation
        if detected_time == 'O(1)':
            if loop_count == 0 and func_count <= 1:
                explanations.append("No loops or recursion → O(1)")
            else:
                explanations.append("Constant time operations → O(1)")
        elif detected_time == 'O(log n)':
            if 'mid' in code.lower() or 'binary' in code.lower():
                explanations.append("Binary search pattern → O(log n)")
            else:
                explanations.append("Halving search space → O(log n)")
        elif detected_time == 'O(n)':
            if loop_count == 1:
                explanations.append(f"Single loop over input → O(n)")
            elif func_count > 1 and 'memo' in code.lower():
                explanations.append("Memoized recursion → O(n)")
            else:
                explanations.append("Linear traversal → O(n)")
        elif detected_time == 'O(n log n)':
            if '.sort' in code.lower() or 'sorted' in code.lower():
                explanations.append("Sorting operation → O(n log n)")
            elif 'heapq' in code.lower():
                explanations.append("Heap operations → O(n log n)")
            else:
                explanations.append("Divide and conquer → O(n log n)")
        elif detected_time == 'O(n²)':
            if loop_count >= 2:
                explanations.append(f"Nested loops ({loop_count} levels) → O(n²)")
            else:
                explanations.append("Quadratic operations → O(n²)")
        elif detected_time == 'O(2ⁿ)':
            explanations.append("Exponential recursion (multiple branches) → O(2ⁿ)")
        elif detected_time == 'O(n³)':
            explanations.append("Triple nested loops → O(n³)")
        
        # Space complexity explanation
        if detected_space == 'O(1)':
            explanations.append("Constant extra space → O(1)")
        elif detected_space == 'O(n)':
            if '[]' in code or 'list(' in code.lower():
                explanations.append("Auxiliary array/list → O(n)")
            elif 'def ' in code and code.count('def ') > 1:
                explanations.append("Recursion stack → O(n)")
            else:
                explanations.append("Linear extra space → O(n)")
        
        return " | ".join(explanations) if explanations else "Complexity analysis inconclusive"
    
    def _analyze_edge_cases(self, code: str, problem_data: Dict) -> Dict:
        """Analyze edge case handling"""
        edge_cases = problem_data.get('key_edge_cases', [])
        code_lower = code.lower()
        
        handled = 0
        details = []
        
        for case in edge_cases:
            case_lower = case.lower()
            handled_flag = False
            
            # More accurate pattern matching
            if 'empty' in case_lower:
                if re.search(r'if\s+not\s+\w+|len\([^)]+\)\s*==\s*0|\[\]\s*:', code_lower):
                    handled_flag = True
            elif 'single' in case_lower or 'one element' in case_lower:
                if re.search(r'len\([^)]+\)\s*[<=>]+\s*1|range\(1\)', code_lower):
                    handled_flag = True
            elif 'negative' in case_lower:
                if re.search(r'[<>]=?\s*0|abs\(|\bnegative\b', code_lower):
                    handled_flag = True
            elif 'null' in case_lower or 'none' in case_lower:
                if re.search(r'is\s+none|==\s*none|if\s+not\s+', code_lower):
                    handled_flag = True
            elif 'zero' in case_lower:
                if re.search(r'==\s*0|!=\s*0', code_lower):
                    handled_flag = True
            else:
                # Generic check - if there's any conditional
                if 'if ' in code_lower:
                    handled_flag = True
            
            if handled_flag:
                handled += 1
            
            details.append({'case': case, 'handled': handled_flag})
        
        coverage = (handled / len(edge_cases) * 100) if edge_cases else 100
        
        return {
            'total_edge_cases': len(edge_cases),
            'handled_edge_cases': handled,
            'coverage_percentage': round(coverage, 1),
            'details': details
        }
    
    def _assess_code_quality(self, code: str) -> Dict:
        """Assess code quality metrics"""
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        
        return {
            'line_count': len(non_empty_lines),
            'total_lines': len(lines),
            'has_comments': any(line.strip().startswith('#') for line in lines),
            'has_docstring': '"""' in code or "'''" in code,
            'has_error_handling': 'try:' in code or 'except' in code or 'raise' in code,
            'has_type_hints': '->' in code or ': List' in code or ': Dict' in code or ': int' in code,
            'function_count': code.count('def '),
            'class_count': code.count('class '),
            'import_count': code.count('import ') + code.count('from ')
        }
    
    def _calculate_scores(self, results: Dict) -> Dict:
        """Calculate comprehensive scores using the metrics"""
        test_results = results['test_results']
        complexity = results['complexity_analysis']
        edge_cases = results['edge_case_analysis']
        quality = results['code_quality']
        
        # Test score (50%)
        if test_results['total'] > 0:
            test_score = (test_results['passed'] / test_results['total']) * 100
        else:
            test_score = 0
        
        # Complexity score (20%)
        complexity_score = 0
        if complexity['time_matches_optimal']:
            complexity_score += 50
        if complexity['space_matches_optimal']:
            complexity_score += 50
        
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
        
        # Overall score
        overall_score = (
            test_score * 0.5 +
            complexity_score * 0.2 +
            edge_case_score * 0.2 +
            quality_score * 0.1
        )
        
        return {
            'test_score': round(test_score, 1),
            'complexity_score': round(complexity_score, 1),
            'edge_case_score': round(edge_case_score, 1),
            'quality_score': round(quality_score, 1),
            'overall_score': round(overall_score, 1)
        }
    
    def _would_pass_leetcode(self, results: Dict) -> bool:
        """Determine if solution would pass LeetCode"""
        # Must have valid syntax
        if not results['syntax_valid']['valid']:
            return False
        
        # Must pass all tests
        if not results['test_results']['all_passed']:
            return False
        
        # Must handle most edge cases (70%+)
        if results['edge_case_analysis']['coverage_percentage'] < 70:
            return False
        
        return True
    
    def print_evaluation(self, evaluation: Dict, prefix: str = ""):
        """Print detailed evaluation results"""
        print(f"\n{prefix}📊 DETAILED EVALUATION:")
        print(f"{prefix}{'='*50}")
        
        # Test Results
        tr = evaluation['test_results']
        print(f"{prefix}🧪 TEST RESULTS:")
        print(f"{prefix}  Passed: {tr['passed']}/{tr['total']}")
        if tr['failed'] > 0:
            print(f"{prefix}  Failed: {tr['failed']}")
            # Show first failure
            for detail in tr['details'][:3]:
                if detail['status'] != 'PASSED':
                    print(f"{prefix}    ❌ {detail['message']}")
        
        # Complexity Analysis
        ca = evaluation['complexity_analysis']
        print(f"{prefix}⚡ COMPLEXITY ANALYSIS:")
        print(f"{prefix}  Detected Time: {ca['detected_time']}")
        print(f"{prefix}  Optimal Time: {ca['optimal_time']}")
        print(f"{prefix}  Time Match: {'✅' if ca['time_matches_optimal'] else '❌'}")
        print(f"{prefix}  Detected Space: {ca['detected_space']}")
        print(f"{prefix}  Optimal Space: {ca['optimal_space']}")
        print(f"{prefix}  Space Match: {'✅' if ca['space_matches_optimal'] else '❌'}")
        
        # Edge Case Analysis
        ea = evaluation['edge_case_analysis']
        print(f"{prefix}🎯 EDGE CASE ANALYSIS:")
        print(f"{prefix}  Coverage: {ea['handled_edge_cases']}/{ea['total_edge_cases']} ({ea['coverage_percentage']:.1f}%)")
        
        # Code Quality
        qual = evaluation['code_quality']
        print(f"{prefix}✨ CODE QUALITY:")
        print(f"{prefix}  Lines: {qual['line_count']} (non-empty)")
        print(f"{prefix}  Functions: {qual['function_count']}")
        print(f"{prefix}  Docstring: {'✅' if qual['has_docstring'] else '❌'}")
        print(f"{prefix}  Comments: {'✅' if qual['has_comments'] else '❌'}")
        print(f"{prefix}  Error Handling: {'✅' if qual['has_error_handling'] else '❌'}")
        print(f"{prefix}  Type Hints: {'✅' if qual['has_type_hints'] else '❌'}")
        
        # Scores
        scores = evaluation['scores']
        print(f"{prefix}🏆 SCORES:")
        print(f"{prefix}  Test: {scores['test_score']:.1f}% (50% weight)")
        print(f"{prefix}  Complexity: {scores['complexity_score']:.1f}% (20% weight)")
        print(f"{prefix}  Edge Cases: {scores['edge_case_score']:.1f}% (20% weight)")
        print(f"{prefix}  Quality: {scores['quality_score']:.1f}% (10% weight)")
        print(f"{prefix}  ⭐ OVERALL: {scores['overall_score']:.1f}%")
        
        # LeetCode Readiness
        print(f"{prefix}🏁 LEETCODE READINESS:")
        print(f"{prefix}  Would Pass: {'✅ YES' if evaluation['would_pass_leetcode'] else '❌ NO'}")


def run_comparison(num_problems: int = 5):
    """Run comparison with fixed evaluation"""
    print("\n" + "="*80)
    print("🏁 LEETCODE HARD PROBLEMS COMPARISON - FIXED VERSION")
    print("="*80)
    
    # Initialize
    direct_gen = DirectCodeGenerator()
    socratic_gen = SocraticCodeGenerator()
    evaluator = FixedLeetCodeEvaluator()
    
    results = []
    
    # Test on specified number of problems
    problems_to_test = LEETCODE_HARD_PROBLEMS[:num_problems]
    
    for problem in problems_to_test:
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
        
        # DIRECT GENERATION
        print("\n1️⃣  DIRECT GENERATION")
        print("-"*40)
        try:
            start_time = time.time()
            direct_code, _ = direct_gen.generate_for_problem(problem)
            direct_time = time.time() - start_time
            
            direct_gen.save_result(direct_code, problem_id)
            
            print(f"  ✓ Generated in {direct_time:.1f}s")
            print(f"  Code length: {len(direct_code)} chars")
            
            # Evaluate
            direct_eval = evaluator.evaluate_solution(direct_code, problem)
            evaluator.print_evaluation(direct_eval, "  ")
            
            problem_result['direct'] = {
                'time': direct_time,
                'code_length': len(direct_code),
                'evaluation': direct_eval
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            problem_result['direct']['error'] = str(e)
        
        time.sleep(2)  # API cooldown
        
        # SOCRATIC GENERATION
        print("\n2️⃣  SOCRATIC GENERATION")
        print("-"*40)
        try:
            start_time = time.time()
            socratic_results, _ = socratic_gen.generate_for_problem(problem)
            socratic_time = time.time() - start_time
            socratic_code = socratic_results.get('code', '')
            
            socratic_gen.save_results(socratic_results, problem_id)
            
            print(f"  ✓ Generated in {socratic_time:.1f}s")
            print(f"  Code length: {len(socratic_code)} chars")
            
            # Evaluate
            socratic_eval = evaluator.evaluate_solution(socratic_code, problem)
            evaluator.print_evaluation(socratic_eval, "  ")
            
            problem_result['socratic'] = {
                'time': socratic_time,
                'code_length': len(socratic_code),
                'evaluation': socratic_eval
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            problem_result['socratic']['error'] = str(e)
        
        # COMPARISON
        if 'evaluation' in problem_result['direct'] and 'evaluation' in problem_result['socratic']:
            print("\n3️⃣  COMPARISON")
            print("-"*40)
            
            d_eval = problem_result['direct']['evaluation']
            s_eval = problem_result['socratic']['evaluation']
            d_time = problem_result['direct']['time']
            s_time = problem_result['socratic']['time']
            
            # Determine winner based on scores
            d_score = d_eval['scores']['overall_score']
            s_score = s_eval['scores']['overall_score']
            
            if abs(d_score - s_score) < 5:
                winner = "tie"
            elif s_score > d_score:
                winner = "socratic"
            else:
                winner = "direct"
            
            comparison = {
                'direct_time': round(d_time, 1),
                'socratic_time': round(s_time, 1),
                'time_ratio': round(s_time / d_time if d_time > 0 else 0, 1),
                'direct_score': d_score,
                'socratic_score': s_score,
                'score_diff': round(s_score - d_score, 1),
                'direct_tests': f"{d_eval['test_results']['passed']}/{d_eval['test_results']['total']}",
                'socratic_tests': f"{s_eval['test_results']['passed']}/{s_eval['test_results']['total']}",
                'winner': winner
            }
            
            problem_result['comparison'] = comparison
            
            # Print comparison
            print(f"  Time: Direct {comparison['direct_time']}s vs Socratic {comparison['socratic_time']}s")
            print(f"  Score: Direct {comparison['direct_score']:.1f}% vs Socratic {comparison['socratic_score']:.1f}%")
            print(f"  Tests: Direct {comparison['direct_tests']} vs Socratic {comparison['socratic_tests']}")
            print(f"  🏆 WINNER: {winner.upper()}")
        
        results.append(problem_result)
        
        # Cooldown between problems
        if problem != problems_to_test[-1]:
            print(f"\n⏳ Cooling down (10s)...")
            time.sleep(10)
    
    # Generate report
    generate_report(results)
    
    return results


def generate_report(results: List[Dict]):
    """Generate final comparison report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"comparison_report_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LEETCODE COMPARISON REPORT - FIXED EVALUATION\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Problems: {len(results)}\n\n")
        
        # Summary Statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        
        direct_wins = sum(1 for r in results if r.get('comparison', {}).get('winner') == 'direct')
        socratic_wins = sum(1 for r in results if r.get('comparison', {}).get('winner') == 'socratic')
        ties = sum(1 for r in results if r.get('comparison', {}).get('winner') == 'tie')
        
        f.write(f"Direct Wins: {direct_wins}\n")
        f.write(f"Socratic Wins: {socratic_wins}\n")
        f.write(f"Ties: {ties}\n\n")
        
        # Calculate averages
        direct_scores = []
        socratic_scores = []
        direct_times = []
        socratic_times = []
        
        for result in results:
            if 'comparison' in result and 'evaluation' in result.get('direct', {}) and 'evaluation' in result.get('socratic', {}):
                comp = result['comparison']
                direct_scores.append(comp['direct_score'])
                socratic_scores.append(comp['socratic_score'])
                direct_times.append(comp['direct_time'])
                socratic_times.append(comp['socratic_time'])
        
        if direct_scores:
            f.write(f"Average Direct Score: {sum(direct_scores)/len(direct_scores):.1f}%\n")
            f.write(f"Average Socratic Score: {sum(socratic_scores)/len(socratic_scores):.1f}%\n")
            f.write(f"Average Direct Time: {sum(direct_times)/len(direct_times):.1f}s\n")
            f.write(f"Average Socratic Time: {sum(socratic_times)/len(socratic_times):.1f}s\n")
            f.write(f"Average Time Ratio: {sum(socratic_times)/sum(direct_times):.1f}x\n\n")
        
        # Detailed Results
        f.write("DETAILED RESULTS PER PROBLEM\n")
        f.write("="*80 + "\n")
        
        for result in results:
            f.write(f"\nProblem: {result['title']} (ID: {result['problem_id']})\n")
            f.write("-"*80 + "\n")
            
            if 'comparison' in result:
                comp = result['comparison']
                f.write(f"Winner: {comp['winner'].upper()}\n\n")
                
                # Direct results
                if 'evaluation' in result['direct']:
                    d_eval = result['direct']['evaluation']
                    f.write("DIRECT GENERATION:\n")
                    f.write(f"  Time: {result['direct']['time']:.1f}s\n")
                    f.write(f"  Tests: {d_eval['test_results']['passed']}/{d_eval['test_results']['total']}\n")
                    f.write(f"  Overall Score: {d_eval['scores']['overall_score']:.1f}%\n")
                    f.write(f"  - Test Score: {d_eval['scores']['test_score']:.1f}%\n")
                    f.write(f"  - Complexity Score: {d_eval['scores']['complexity_score']:.1f}%\n")
                    f.write(f"  - Edge Case Score: {d_eval['scores']['edge_case_score']:.1f}%\n")
                    f.write(f"  - Quality Score: {d_eval['scores']['quality_score']:.1f}%\n")
                    f.write(f"  Would Pass LeetCode: {'YES' if d_eval['would_pass_leetcode'] else 'NO'}\n\n")
                
                # Socratic results
                if 'evaluation' in result['socratic']:
                    s_eval = result['socratic']['evaluation']
                    f.write("SOCRATIC GENERATION:\n")
                    f.write(f"  Time: {result['socratic']['time']:.1f}s\n")
                    f.write(f"  Tests: {s_eval['test_results']['passed']}/{s_eval['test_results']['total']}\n")
                    f.write(f"  Overall Score: {s_eval['scores']['overall_score']:.1f}%\n")
                    f.write(f"  - Test Score: {s_eval['scores']['test_score']:.1f}%\n")
                    f.write(f"  - Complexity Score: {s_eval['scores']['complexity_score']:.1f}%\n")
                    f.write(f"  - Edge Case Score: {s_eval['scores']['edge_case_score']:.1f}%\n")
                    f.write(f"  - Quality Score: {s_eval['scores']['quality_score']:.1f}%\n")
                    f.write(f"  Would Pass LeetCode: {'YES' if s_eval['would_pass_leetcode'] else 'NO'}\n\n")
        
        # Recommendations
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        
        if socratic_wins > direct_wins:
            f.write("✅ SOCRATIC METHOD WINS\n")
            f.write(f"   Score: {socratic_wins}-{direct_wins}-{ties}\n")
            f.write("   Recommendation: Use Socratic for hard LeetCode problems\n")
        elif direct_wins > socratic_wins:
            f.write("✅ DIRECT GENERATION WINS\n")
            f.write(f"   Score: {direct_wins}-{socratic_wins}-{ties}\n")
            f.write("   Recommendation: Use Direct for speed without sacrificing quality\n")
        else:
            f.write("⚖️  BOTH METHODS ARE COMPARABLE\n")
            f.write("   Recommendation: Choose based on time constraints\n")
        
        if direct_scores and socratic_scores:
            avg_quality_diff = (sum(socratic_scores) - sum(direct_scores)) / len(direct_scores)
            avg_time_ratio = sum(socratic_times) / sum(direct_times)
            
            f.write(f"\nKey Metrics:\n")
            f.write(f"  - Quality Difference: {avg_quality_diff:+.1f}%\n")
            f.write(f"  - Time Trade-off: {avg_time_ratio:.1f}x slower\n")
            f.write(f"  - Worth it: {'YES' if avg_quality_diff > 10 else 'NO' if avg_quality_diff < -10 else 'MAYBE'}\n")
    
    print(f"\n✅ Report saved to {report_file}")
    
    # Also save JSON
    json_file = f"comparison_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ JSON results saved to {json_file}")


if __name__ == "__main__":
    print("="*80)
    print("LEETCODE COMPARISON RUNNER - FIXED VERSION")
    print("="*80)
    print("\nThis runner will:")
    print("  ✅ Actually execute test cases")
    print("  ✅ Accurately detect complexity")
    print("  ✅ Properly evaluate edge cases")
    print("  ✅ Give meaningful scores")
    print("\nUsage:")
    print("  python comparison_runner_fixed.py")
    print("\nOr in Python:")
    print("  from comparison_runner_fixed import run_comparison")
    print("  results = run_comparison(num_problems=3)  # Test on 3 problems")
    print("  results = run_comparison(num_problems=5)  # Test on all 5 problems")
    print("\n" + "="*80)
    
    # Run comparison
    results = run_comparison(num_problems=5)