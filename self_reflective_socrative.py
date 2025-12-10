import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

class SelfReflectiveSocraticCoder:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def plan_debate(self, problem: str) -> str:
        """First, plan WHAT to debate about"""
        planning_prompt = f"""Analyze this coding problem and plan a debate:

PROBLEM: {problem}

As a debate planner, identify:
1. What are the KEY DECISIONS needed?
2. What are the MAIN TRADE-OFFS?
3. What EDGE CASES should be debated?
4. What ALTERNATIVE APPROACHES exist?

Think step by step. Then create a debate outline with 3-5 specific debate points.

DEBATE PLAN:
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": planning_prompt}],
            "max_tokens": 400,
            "temperature": 0.8
        }

        response = requests.post(self.api_url, headers=headers, json=data, verify=False)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def have_debate(self, problem: str, debate_plan: str) -> str:
        """Have the actual debate based on the plan"""

        debate_prompt = f"""Three expert programmers debate this problem:

PROBLEM: {problem}

DEBATE PLAN (what to discuss):
{debate_plan}

ROLES:
- ARCHITECT: Loves elegant solutions. Values simplicity and clarity.
- TESTER: Paranoid about failures. Looks for every possible edge case.
- OPTIMIZER: Pragmatic about trade-offs. Cares about performance and maintainability.

RULES:
1. Each persona must take a DISTINCT position
2. They must reference the debate plan points
3. They must challenge each other's assumptions
4. The debate must lead to a BETTER solution

Start the debate (3-4 rounds):

🧠 ARCHITECT:
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": debate_prompt}],
            "max_tokens": 800,
            "temperature": 0.9  # Higher temp for more creative debate
        }

        response = requests.post(self.api_url, headers=headers, json=data, verify=False)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def synthesize_code(self, problem: str, debate_plan: str, debate: str) -> str:
        """Synthesize code from the debate"""

        synthesis_prompt = f"""Based on this debate, write the final code:

PROBLEM: {problem}

DEBATE PLAN:
{debate_plan}

FULL DEBATE:
{debate}

SYNTHESIS INSTRUCTIONS:
1. Extract KEY DECISIONS from the debate
2. Address ALL concerns raised by tester
3. Implement OPTIMIZATIONS discussed
4. Include ERROR HANDLING for debated edge cases
5. Add COMMENTS explaining design choices
6. Make it PRODUCTION-READY

FINAL PYTHON CODE:
```python
"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": synthesis_prompt}],
            "max_tokens": 600,
            "temperature": 0.7  # Lower temp for more consistent code
        }
        
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
    
    def generate_code(self, problem: str) -> dict:
        """Complete self-reflective Socratic process"""
        
        print("🔍 STEP 1: Planning the debate...")
        debate_plan = self.plan_debate(problem)
        print(f"   Debate plan created ({len(debate_plan.split())} words)")
        
        print("💬 STEP 2: Having the debate...")
        debate = self.have_debate(problem, debate_plan)
        print(f"   Debate completed ({len(debate.split())} words)")
        
        print("⚡ STEP 3: Synthesizing code...")
        code = self.synthesize_code(problem, debate_plan, debate)
        print(f"   Code synthesized ({len(code.split())} words)")
        
        return {
            "debate_plan": debate_plan,
            "debate": debate,
            "code": code,
            "process": "plan → debate → synthesize"
        }

class DirectCoder:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def generate_code(self, problem: str) -> dict:
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
        
        return {
            "code": code,
            "process": "direct generation"
        }

def analyze_code_quality(code: str) -> dict:
    """Analyze code quality comprehensively"""
    
    analysis = {
        "error_handling_score": 0,
        "documentation_score": 0,
        "edge_case_coverage": [],
        "design_decisions": []
    }
    
    code_lower = code.lower()
    
    # Error handling analysis
    error_indicators = [
        ('try:', 2), ('except', 2), ('zerodivisionerror', 3),
        ('typeerror', 2), ('valueerror', 2), ('assert', 1),
        ('if not', 1), ('isinstance', 1), ('raise', 2)
    ]
    
    for indicator, score in error_indicators:
        if indicator in code_lower:
            analysis["error_handling_score"] += score
    
    # Documentation analysis
    if '"""' in code or "'''" in code:
        analysis["documentation_score"] += 3
    if '#' in code:
        analysis["documentation_score"] += min(code.count('#'), 5)  # Up to 5 points for comments
    
    # Edge case detection
    edge_case_keywords = ['edge', 'corner', 'boundary', 'empty', 'null', 'none', 
                         'zero', 'negative', 'large', 'small', 'invalid', 'valid']
    
    for keyword in edge_case_keywords:
        if keyword in code_lower:
            analysis["edge_case_coverage"].append(keyword)
    
    # Design decisions detection
    design_patterns = ['def ', 'class ', 'import ', 'from ', 'return ', 
                      'yield ', 'async ', 'await ', 'decorator', 'generator']
    
    for pattern in design_patterns:
        if pattern in code_lower:
            analysis["design_decisions"].append(pattern.replace(' ', ''))
    
    return analysis

def compare_methods(problem: str):
    """Compare self-reflective Socratic vs Direct"""
    
    print("=" * 70)
    print("🧠 SELF-REFLECTIVE SOCRATIC vs DIRECT GENERATION")
    print("=" * 70)
    print(f"Problem: {problem}")
    print("-" * 70)
    
    # Self-Reflective Socratic
    print("\n🎯 SELF-REFLECTIVE SOCRATIC PROCESS:")
    socratic_coder = SelfReflectiveSocraticCoder()
    
    start_time = time.time()
    socratic_result = socratic_coder.generate_code(problem)
    socratic_time = time.time() - start_time
    
    print(f"\n⏱️  Total time: {socratic_time:.2f}s")
    print(f"📊 Process: {socratic_result['process']}")
    
    socratic_analysis = analyze_code_quality(socratic_result['code'])
    
    print("\n📄 SOCRATIC CODE (first 20 lines):")
    lines = socratic_result['code'].split('\n')
    for i, line in enumerate(lines[:20]):
        print(f"{i+1:3d}: {line}")
    if len(lines) > 20:
        print(f"... and {len(lines)-20} more lines")
    
    # Direct
    print("\n" + "=" * 70)
    print("🤖 DIRECT GENERATION:")
    direct_coder = DirectCoder()
    
    start_time = time.time()
    direct_result = direct_coder.generate_code(problem)
    direct_time = time.time() - start_time
    
    print(f"⏱️  Time: {direct_time:.2f}s")
    print(f"📊 Process: {direct_result['process']}")
    
    direct_analysis = analyze_code_quality(direct_result['code'])
    
    print("\n📄 DIRECT CODE (first 20 lines):")
    lines = direct_result['code'].split('\n')
    for i, line in enumerate(lines[:20]):
        print(f"{i+1:3d}: {line}")
    if len(lines) > 20:
        print(f"... and {len(lines)-20} more lines")
    
    # Comparison
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'Self-Reflective Socratic':<25} {'Direct':<15} {'Winner':<10}")
    print("-" * 80)
    
    # Time
    time_ratio = socratic_time / direct_time if direct_time > 0 else 0
    print(f"{'Time (seconds)':<30} {socratic_time:.2f}s{'':<15} {direct_time:.2f}s{'':<7} {'Direct' if direct_time < socratic_time else 'Socratic'}")
    print(f"{'Time Ratio (Socratic/Direct)':<30} {time_ratio:.1f}x{'':<18} 1.0x{'':<9} {'Slower' if time_ratio > 1 else 'Faster'}")
    
    # Error handling
    print(f"{'Error Handling Score (0-20)':<30} {socratic_analysis['error_handling_score']}/20{'':<14} {direct_analysis['error_handling_score']}/20{'':<7} {'Socratic ✓' if socratic_analysis['error_handling_score'] > direct_analysis['error_handling_score'] else 'Direct' if direct_analysis['error_handling_score'] > socratic_analysis['error_handling_score'] else 'Tie'}")
    
    # Documentation
    print(f"{'Documentation Score (0-10)':<30} {socratic_analysis['documentation_score']}/10{'':<15} {direct_analysis['documentation_score']}/10{'':<7} {'Socratic ✓' if socratic_analysis['documentation_score'] > direct_analysis['documentation_score'] else 'Direct' if direct_analysis['documentation_score'] > socratic_analysis['documentation_score'] else 'Tie'}")
    
    # Edge cases
    socratic_edges = len(socratic_analysis['edge_case_coverage'])
    direct_edges = len(direct_analysis['edge_case_coverage'])
    print(f"{'Edge Cases Covered':<30} {socratic_edges}{'':<18} {direct_edges}{'':<9} {'Socratic ✓' if socratic_edges > direct_edges else 'Direct' if direct_edges > socratic_edges else 'Tie'}")
    
    # Design complexity
    socratic_designs = len(socratic_analysis['design_decisions'])
    direct_designs = len(direct_analysis['design_decisions'])
    print(f"{'Design Elements':<30} {socratic_designs}{'':<18} {direct_designs}{'':<9} {'Socratic' if socratic_designs > direct_designs else 'Direct' if direct_designs > socratic_designs else 'Tie'}")
    
    # Code length
    socratic_lines = socratic_result['code'].count('\n') + 1
    direct_lines = direct_result['code'].count('\n') + 1
    print(f"{'Code Lines':<30} {socratic_lines}{'':<18} {direct_lines}{'':<9} {'More detailed' if socratic_lines > direct_lines else 'More concise'}")
    
    print("-" * 80)
    
    # Value calculation
    quality_improvement = (socratic_analysis['error_handling_score'] + socratic_analysis['documentation_score']) - \
                         (direct_analysis['error_handling_score'] + direct_analysis['documentation_score'])
    
    time_cost = socratic_time - direct_time
    
    if time_cost > 0:
        value_per_second = quality_improvement / time_cost
    else:
        value_per_second = float('inf')
    
    print(f"\n💎 VALUE ANALYSIS:")
    print(f"   Quality Improvement: {quality_improvement} points")
    print(f"   Additional Time Cost: {time_cost:.2f}s")
    print(f"   Value per Second: {value_per_second:.2f} quality points/second")
    
    if value_per_second > 1:
        print(f"   🎯 VERDICT: Socratic is WORTH the extra time!")
    elif value_per_second > 0:
        print(f"   ⚖️  VERDICT: Socratic has marginal benefits")
    else:
        print(f"   ⚠️  VERDICT: Direct is better for this problem")
    
    # Save results
    with open("self_reflective_results.txt", "w") as f:
        f.write(f"Problem: {problem}\n")
        f.write(f"Socratic Time: {socratic_time:.2f}s\n")
        f.write(f"Direct Time: {direct_time:.2f}s\n")
        f.write(f"Quality Improvement: {quality_improvement}\n")
        f.write(f"Value per Second: {value_per_second:.2f}\n\n")
        
        f.write("SOCRATIC DEBATE PLAN:\n")
        f.write(socratic_result['debate_plan'][:500] + "...\n\n")
        
        f.write("SOCRATIC CODE:\n")
        f.write(socratic_result['code'][:1000] + "...\n\n")
        
        f.write("DIRECT CODE:\n")
        f.write(direct_result['code'][:1000] + "...\n")
    
    print(f"\n📁 Results saved to: self_reflective_results.txt")
    
    return {
        "socratic": socratic_result,
        "direct": direct_result,
        "socratic_analysis": socratic_analysis,
        "direct_analysis": direct_analysis,
        "socratic_time": socratic_time,
        "direct_time": direct_time,
        "value_per_second": value_per_second
    }

if __name__ == "__main__":
    # Test with different problem difficulties
    problems = [
        "Write a function to safely divide two numbers",
        "Implement a function to validate and parse email addresses",
        "Create a thread-safe counter that can be incremented from multiple threads"
    ]
    
    all_results = []
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{'#' * 80}")
        print(f"TEST {i}/{len(problems)}")
        print(f"{'#' * 80}")
        
        result = compare_methods(problem)
        all_results.append(result)
        
        if i < len(problems):
            input(f"\n⏸️  Press Enter for next test...")
    
    # Final analysis
    print("\n" + "=" * 80)
    print("📈 OVERALL FINDINGS")
    print("=" * 80)
    
    avg_value = sum(r["value_per_second"] for r in all_results) / len(all_results)
    avg_time_ratio = sum(r["socratic_time"] / r["direct_time"] for r in all_results) / len(all_results)
    
    print(f"Average Value per Second: {avg_value:.2f}")
    print(f"Average Time Ratio (Socratic/Direct): {avg_time_ratio:.1f}x")
    
    if avg_value > 1:
        print("\n🎯 OVERALL VERDICT: Self-Reflective Socratic is VALUABLE!")
        print("   The extra thinking time produces significantly better code.")
    elif avg_value > 0:
        print("\n⚖️  OVERALL VERDICT: Socratic has modest benefits")
        print("   Benefits exist but may not justify time for simple problems.")
    else:
        print("\n⚠️  OVERALL VERDICT: Direct generation is often sufficient")
        print("   For these problems, simpler approaches work well.")