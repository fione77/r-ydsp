"""
Socratic Code Generator - FOR LEETCODE HARD PROBLEMS - FIXED VERSION
Generates code through debate and synthesis
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


class MultiTurnDebate:
    """Manages multi-turn debate between three personas"""
    
    def __init__(self, api_caller):
        self._make_api_call = api_caller
        
        # System prompts for each persona (unchanged)
        self.architect_system = """You are THE ARCHITECT - a systems thinking expert..."""
        self.tester_system = """You are THE TESTER - a skeptical empiricist focused on robustness..."""
        self.optimizer_system = """You are THE OPTIMIZER - a performance realist and pragmatist..."""
    
    def conduct_multi_turn_debate(self, problem: str, plan: str) -> str:
        """Run multi-turn debate with separate API calls per persona"""
        print("  (Each persona responds separately in sequence)")
        
        debate_log = []
        debate_log.append(f"PROBLEM:\n{problem}\n")
        debate_log.append(f"\nDEBATE AGENDA:\n{plan}\n")
        debate_log.append("\n" + "="*70)
        debate_log.append("DEBATE BEGINS - MULTI-TURN CONVERSATION")
        debate_log.append("="*70 + "\n")
        
        context = f"Problem: {problem}\n\nAgenda: {plan}\n\n"
        
        # ROUND 1
        debate_log.append("\n--- ROUND 1: INITIAL EXPLORATION ---\n")
        
        print("  🏗️  ARCHITECT thinking...")
        architect_r1 = self._make_api_call(
            f"{self.architect_system}\n\nDEBATE CONTEXT:\n{context}\n\nYOUR TURN: Start the debate...",
            max_tokens=300, temperature=0.85
        )
        debate_log.append(f"🏗️ ARCHITECT:\n{architect_r1}\n")
        context += f"ARCHITECT: {architect_r1}\n\n"
        
        time.sleep(2)
        
        print("  🔬 TESTER thinking...")
        tester_r1 = self._make_api_call(
            f"{self.tester_system}\n\nDEBATE CONTEXT:\n{context}\n\nYOUR TURN: Respond to ARCHITECT...",
            max_tokens=300, temperature=0.85
        )
        debate_log.append(f"🔬 TESTER:\n{tester_r1}\n")
        context += f"TESTER: {tester_r1}\n\n"
        
        time.sleep(2)
        
        print("  ⚡ OPTIMIZER thinking...")
        optimizer_r1 = self._make_api_call(
            f"{self.optimizer_system}\n\nDEBATE CONTEXT:\n{context}\n\nYOUR TURN: Respond to both...",
            max_tokens=300, temperature=0.85
        )
        debate_log.append(f"⚡ OPTIMIZER:\n{optimizer_r1}\n")
        context += f"OPTIMIZER: {optimizer_r1}\n\n"
        
        time.sleep(2)
        
        # ROUND 2
        debate_log.append("\n--- ROUND 2: EXPLORING TRADE-OFFS ---\n")
        
        print("  🏗️  ARCHITECT refining...")
        architect_r2 = self._make_api_call(
            f"{self.architect_system}\n\nDEBATE CONTEXT:\n{context}\n\nYOUR TURN: Refine your thinking...",
            max_tokens=300, temperature=0.85
        )
        debate_log.append(f"🏗️ ARCHITECT:\n{architect_r2}\n")
        context += f"ARCHITECT: {architect_r2}\n\n"
        
        time.sleep(2)
        
        print("  🔬 TESTER probing...")
        tester_r2 = self._make_api_call(
            f"{self.tester_system}\n\nDEBATE CONTEXT:\n{context}\n\nYOUR TURN: Probe deeper...",
            max_tokens=300, temperature=0.85
        )
        debate_log.append(f"🔬 TESTER:\n{tester_r2}\n")
        context += f"TESTER: {tester_r2}\n\n"
        
        time.sleep(2)
        
        print("  ⚡ OPTIMIZER evaluating...")
        optimizer_r2 = self._make_api_call(
            f"{self.optimizer_system}\n\nDEBATE CONTEXT:\n{context}\n\nYOUR TURN: Evaluate trade-offs...",
            max_tokens=300, temperature=0.85
        )
        debate_log.append(f"⚡ OPTIMIZER:\n{optimizer_r2}\n")
        
        debate_log.append("\n" + "="*70)
        debate_log.append("DEBATE CONCLUDED")
        debate_log.append("="*70)
        
        return "\n".join(debate_log)


class SocraticCodeGenerator:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        print(f"🤖 Socratic Generator initialized")
        
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
            "explanation:", "note:", "important:", "example:"
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
                if stripped and not stripped.startswith('#') and len(stripped) > 100:
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
            if (len(words) > 8 and 
                not stripped.startswith('#') and
                not '=' in stripped and
                not ':' in stripped and
                not stripped.endswith(':')):
                # Looks like a sentence, not code
                continue
                
            final_lines.append(line)
        
        return '\n'.join(final_lines).strip()
    
    def _validate_code_syntax(self, code: str) -> bool:
        """Quick syntax validation"""
        try:
            import ast
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _retry_with_stricter_prompt(self, problem: str, debate: str, attempt: int) -> str:
        """Retry code generation with stricter prompt"""
        stricter_prompt = f"""Write Python code to solve this LeetCode problem:

{problem}

Based on debate insights: {debate[-500:]}

IMPORTANT: Provide ONLY the Python code. NO explanations, NO comments outside code blocks, 
NO text before or after the code. The code must be syntactically correct and complete.

Start with imports or function/class definition. End with the last line of code.
DO NOT add any explanatory text.

Code:"""
        
        return self._make_api_call(stricter_prompt, max_tokens=2500, temperature=0.2)
    
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
        
        max_retries = 3
        
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
                    time.sleep(2 * (attempt + 1))
                    continue
                
                if response.status_code != 200:
                    continue
                
                response_data = response.json()
                if "choices" not in response_data:
                    continue
                
                return response_data["choices"][0]["message"]["content"].strip()
                
            except Exception:
                time.sleep(1)
        
        return ""
    
    def generate_for_problem(self, problem_data: dict) -> Tuple[dict, float]:
        """Full Socratic generation process for a specific problem"""
        problem_id = problem_data.get("id", "Unknown")
        title = problem_data.get("title", "")
        description = problem_data.get("description", "")
        
        print(f"\n{'='*70}")
        print(f"SOCRATIC GENERATION - Problem {problem_id}: {title}")
        print(f"{'='*70}")
        
        total_start = time.time()
        results = {}
        
        full_problem = f"{title}\n\n{description}"
        
        # Step 1: Plan
        print("\n📋 Planning debate...")
        plan_prompt = f"""Analyze this coding problem and create a debate agenda:

{full_problem}

List key approaches, trade-offs, and edge cases to debate."""
        
        plan = self._make_api_call(plan_prompt, max_tokens=400, temperature=0.8)
        results['plan'] = plan
        print(f"  ✓ Planning completed")
        
        time.sleep(2)
        
        # Step 2: Debate
        print("\n💬 Conducting debate...")
        debate_manager = MultiTurnDebate(self._make_api_call)
        debate = debate_manager.conduct_multi_turn_debate(full_problem, plan)
        results['debate'] = debate
        print(f"  ✓ Debate completed")
        
        time.sleep(2)
        
        # Step 3: Synthesize with retry logic
        print("\n⚙️  Synthesizing code...")
        
        # First attempt
        synth_prompt = f"""Based on this debate, write Python code to solve:

{full_problem}

DEBATE INSIGHTS:
{debate[-1000:]}

Write complete, executable Python code. Include all necessary imports and handle edge cases.

IMPORTANT: Provide ONLY the Python code. No explanations before or after.

Code:"""
        
        code = self._make_api_call(synth_prompt, max_tokens=2500, temperature=0.3)
        code = self._clean_generated_code(code)
        
        # Validate and retry if needed
        max_retries = 2
        for attempt in range(max_retries):
            if self._validate_code_syntax(code):
                break
            print(f"  ⚠️  Syntax error detected, retry {attempt + 1}/{max_retries}...")
            code = self._retry_with_stricter_prompt(full_problem, debate, attempt)
            code = self._clean_generated_code(code)
        
        results['code'] = code
        results['total_time'] = time.time() - total_start
        
        print(f"\n📊 Generation completed in {results['total_time']:.1f}s")
        print(f"  Code length: {len(code)} characters")
        print(f"  Syntax valid: {self._validate_code_syntax(code)}")
        
        return results, results['total_time']
    
    def save_results(self, results: dict, problem_id: int):
        """Save generated code and debate"""
        # Save code
        code_file = f"socratic_{problem_id}.py"
        with open(code_file, "w", encoding='utf-8') as f:
            f.write(f"# SOCRATIC GENERATION - Problem {problem_id}\n")
            f.write("# Generated after debate synthesis\n\n")
            f.write(results['code'])
        print(f"  ✓ Saved code to {code_file}")
        
        # Save debate
        debate_file = f"debate_{problem_id}.txt"
        with open(debate_file, "w", encoding='utf-8') as f:
            f.write(f"DEBATE TRANSCRIPT - Problem {problem_id}\n")
            f.write("="*70 + "\n\n")
            f.write(results.get('debate', 'No debate generated'))
        print(f"  ✓ Saved debate to {debate_file}")


if __name__ == "__main__":
    generator = SocraticCodeGenerator()
    
    # Example problem
    test_problem = {
        "id": 42,
        "title": "Trapping Rain Water",
        "description": "Given n non-negative integers representing an elevation map..."
    }
    
    results, timing = generator.generate_for_problem(test_problem)
    
    print("\n" + "="*70)
    print("CODE PREVIEW (First 500 chars):")
    print("="*70)
    print(results['code'][:500] + "..." if len(results['code']) > 500 else results['code'])
    
    print(f"\nTotal time: {timing:.2f}s")
    
    generator.save_results(results, test_problem["id"])