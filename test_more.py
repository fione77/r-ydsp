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
            "max_tokens": 600,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, verify=False, timeout=30)
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

def analyze_code_quality(code: str, problem: str) -> dict:
    """Analyze the quality of generated code"""
    
    metrics = {
        "lines": code.count('\n') + 1,
        "has_comments": '#' in code,
        "has_docstring": '"""' in code or "'''" in code,
        "has_error_handling": any(x in code.lower() for x in ['try:', 'except', 'if not', 'assert', 'raise']),
        "has_type_hints": '->' in code or ': int' in code or ': str' in code or ': float' in code or ': bool' in code,
        "has_tests": 'test' in code.lower() or 'example' in code.lower() or 'print(' in code,
        "mentions_edge": 'edge' in code.lower() or 'corner' in code.lower(),
        "complexity_score": 0
    }
    
    # Simple complexity estimate
    complexity = 0
    if 'for ' in code: complexity += 1
    if 'while ' in code: complexity += 1
    if 'if ' in code: complexity += 1
    if 'def ' in code: complexity += 1
    if 'class ' in code: complexity += 2
    if 'import ' in code: complexity += 1
    metrics["complexity_score"] = complexity
    
    return metrics

# HARDER TEST PROBLEMS (Fewer to save time)
HARD_PROBLEMS = [
    {
        "difficulty": "Hard",
        "problem": "Implement a thread-safe LRU cache with TTL expiration"
    },
    {
        "difficulty": "Hard", 
        "problem": "Write a function to serialize and deserialize a binary tree"
    },
    {
        "difficulty": "Very Hard",
        "problem": "Implement a concurrent web crawler with rate limiting"
    },
    {
        "difficulty": "Medium-Hard",
        "problem": "Create a Python decorator that memoizes function results"
    }
]

def main():
    print("=" * 70)
    print("🧠 SOCRATIC CODER - HARD PROBLEMS CHALLENGE")
    print("=" * 70)
    print("Testing Socratic dialogue on complex tasks\n")
    
    coder = SocraticCoder()
    results = []
    
    for i, item in enumerate(HARD_PROBLEMS, 1):
        difficulty = item["difficulty"]
        problem = item["problem"]
        
        print(f"\n{'#' * 70}")
        print(f"TEST {i}: {difficulty}")
        print(f"Problem: {problem}")
        print(f"{'#' * 70}\n")
        
        # Generate code
        print("🤔 Generating code...")
        start_time = time.time()
        code = coder.generate_code(problem)
        elapsed = time.time() - start_time
        
        line_count = code.count('\n') + 1
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"📝 Code length: {len(code)} characters, {line_count} lines")
        
        # Analyze quality
        metrics = analyze_code_quality(code, problem)
        
        print("\n📊 QUALITY METRICS:")
        print(f"  • Has documentation: {'✅' if metrics['has_docstring'] else '❌'}")
        print(f"  • Has comments: {'✅' if metrics['has_comments'] else '❌'}")
        print(f"  • Has error handling: {'✅' if metrics['has_error_handling'] else '❌'}")
        print(f"  • Has type hints: {'✅' if metrics['has_type_hints'] else '❌'}")
        print(f"  • Includes tests/examples: {'✅' if metrics['has_tests'] else '❌'}")
        print(f"  • Mentions edge cases: {'✅' if metrics['mentions_edge'] else '❌'}")
        print(f"  • Complexity score: {metrics['complexity_score']}/10")
        
        # Save result
        results.append({
            "test": i,
            "difficulty": difficulty,
            "problem": problem,
            "code": code,
            "time": elapsed,
            "metrics": metrics
        })
        
        # Save to file
        filename = f"hard_test_{i}.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Problem: {problem}\n")
            f.write(f"# Difficulty: {difficulty}\n")
            f.write(f"# Generated in: {elapsed:.2f}s\n\n")
            f.write(code)
        
        print(f"💾 Saved to: {filename}")
        
        # Show first 10 lines of code
        print("\n📄 CODE PREVIEW (first 10 lines):")
        lines = code.split('\n')
        for j, line in enumerate(lines[:10]):
            print(f"  {j+1:2d}: {line}")
        if len(lines) > 10:
            print(f"  ... and {len(lines)-10} more lines")
        
        # Small pause between tests
        if i < len(HARD_PROBLEMS):
            print("\n" + "-" * 50)
            input(f"⏸️  Press Enter for next test ({i+1}/{len(HARD_PROBLEMS)})...")
    
    # Final summary
    print("\n" + "=" * 70)
    print("📈 FINAL SUMMARY")
    print("=" * 70)
    
    total_time = sum(r["time"] for r in results)
    avg_time = total_time / len(results)
    
    quality_scores = []
    for r in results:
        metrics = r["metrics"]
        score = sum(1 for k,v in metrics.items() if v and k not in ['lines', 'complexity_score'])
        quality_scores.append(score)
    
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    print(f"Total tests: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per test: {avg_time:.2f}s")
    print(f"Average quality score: {avg_quality:.1f}/6.0")
    
    # Generate report
    with open("hard_tests_report.txt", "w") as f:
        f.write("SOCRATIC CODER - HARD PROBLEMS TEST\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Tests completed: {len(results)}\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Average quality: {avg_quality:.1f}/6.0\n\n")
        
        for r in results:
            f.write(f"Test {r['test']} - {r['difficulty']}\n")
            f.write(f"Problem: {r['problem']}\n")
            f.write(f"Time: {r['time']:.2f}s\n")
            f.write("-" * 40 + "\n")
    
    print(f"\n📄 Report saved to: hard_tests_report.txt")
    print("🎯 Test completed!")

if __name__ == "__main__":
    main()