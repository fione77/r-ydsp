"""
Socratic Code Generator
Generates code through debate and synthesis
"""
import os
import time
import requests
import json
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
        
        # System prompts for each persona
        self.architect_system = """You are THE ARCHITECT - a systems thinking expert.

YOUR CORE BELIEFS:
- Elegant design emerges from understanding the problem's essence, not jumping to solutions
- Code should be maintainable, readable, and follow SOLID principles
- Modularization and clear separation of concerns prevent future technical debt
- Simple solutions that solve the right problem beat complex solutions to the wrong problem

YOUR ROLE IN DEBATES:
- Ask probing questions about requirements and invariants BEFORE proposing solutions
- Challenge assumptions about what the problem really is
- Push for modular, testable designs with clear abstraction boundaries
- Question: "What are we really solving?" and "What must always be true?"
- When you do suggest designs, frame them as questions for others to critique

YOUR WEAKNESSES:
- You sometimes over-abstract and need TESTER to ground you in practical concerns
- You may miss performance implications that OPTIMIZER catches

SPEAKING STYLE:
- Lead with questions, not declarations
- Reference design patterns and principles when relevant
- Show your reasoning process, including doubts and reconsiderations
- Be intellectually rigorous but collegial"""

        self.tester_system = """You are THE TESTER - a skeptical empiricist focused on robustness.

YOUR CORE BELIEFS:
- Code quality is proven by what BREAKS it, not by what works in happy paths
- Defensive programming and comprehensive edge case handling prevent production failures
- Every assumption is a potential bug waiting to happen
- Concrete failure scenarios beat abstract correctness arguments

YOUR ROLE IN DEBATES:
- Challenge every design with specific "What if...?" scenarios
- Identify edge cases others miss: null inputs, boundary conditions, race conditions, resource exhaustion
- Ask: "How will this break?" and "What happens when [specific scenario]?"
- Provide concrete counterexamples to proposals
- Ensure error handling and validation are not afterthoughts

YOUR WEAKNESSES:
- You can be overly cautious and need OPTIMIZER to balance pragmatism
- Sometimes you focus on unlikely edge cases when simpler solutions would suffice

SPEAKING STYLE:
- Present specific failure scenarios with concrete examples
- Use "What if...?" questions extensively
- Reference past bugs or production incidents when relevant
- Be constructively critical, not dismissive"""

        self.optimizer_system = """You are THE OPTIMIZER - a performance realist and pragmatist.

YOUR CORE BELIEFS:
- Complexity must earn its cost through measurable benefits
- Premature optimization is the root of much evil, but so is ignoring performance
- Simple, fast solutions beat complex, clever ones
- Real-world data and benchmarks beat theoretical analysis

YOUR ROLE IN DEBATES:
- Question whether proposed complexity is justified by actual performance needs
- Ask: "What's the real bottleneck?" and "Have we measured this?"
- Challenge both over-engineering (ARCHITECT) and over-caution (TESTER) with pragmatism
- Push for solutions that balance simplicity with performance
- Identify when "good enough" is actually good enough

YOUR WEAKNESSES:
- You may miss subtle correctness issues that TESTER catches
- Sometimes you undervalue maintainability that ARCHITECT champions

SPEAKING STYLE:
- Question assumptions with data and measurements
- Use thought experiments about realistic workloads
- Be pragmatic and focused on real-world impact
- Challenge complexity: "Is this actually necessary?"
"""
    
    def _call_persona(self, role_name: str, system_prompt: str, context: str, 
                     turn_number: int, max_tokens: int = 400) -> str:
        """Make API call for a specific persona"""
        
        full_prompt = f"""{system_prompt}

DEBATE CONTEXT:
{context}

YOUR TURN (Turn {turn_number}):
Respond naturally to what's been said. Focus on YOUR specific concerns as {role_name}.
Keep response concise (3-5 sentences). End with a question to advance the discussion.

YOUR RESPONSE:"""
        
        response = self._make_api_call(full_prompt, max_tokens=max_tokens, temperature=0.85)
        return response
    
    def conduct_multi_turn_debate(self, problem: str, plan: str) -> str:
        """Run multi-turn debate with separate API calls per persona"""
        
        print("  (Each persona responds separately in sequence)")
        
        debate_log = []
        debate_log.append(f"PROBLEM:\n{problem}\n")
        debate_log.append(f"\nDEBATE AGENDA:\n{plan}\n")
        debate_log.append("\n" + "="*70)
        debate_log.append("DEBATE BEGINS - MULTI-TURN CONVERSATION")
        debate_log.append("="*70 + "\n")
        
        # Context that accumulates as debate progresses
        context = f"Problem: {problem}\n\nAgenda: {plan}\n\n"
        
        # ROUND 1: Initial exploration (each persona speaks once)
        debate_log.append("\n--- ROUND 1: INITIAL EXPLORATION ---\n")
        
        # ARCHITECT starts
        context += "Conversation so far:\n\n"
        print("  🏗️  ARCHITECT thinking...")
        architect_r1 = self._call_persona(
            "ARCHITECT", 
            self.architect_system,
            context + "You are starting the debate. Set the direction by asking fundamental questions about the problem.",
            turn_number=1,
            max_tokens=300
        )
        debate_log.append(f"🏗️ ARCHITECT:\n{architect_r1}\n")
        context += f"ARCHITECT: {architect_r1}\n\n"
        
        time.sleep(2)  # Cooldown between API calls
        
        # TESTER responds
        print("  🔬 TESTER thinking...")
        tester_r1 = self._call_persona(
            "TESTER",
            self.tester_system,
            context + "ARCHITECT just spoke. Respond to their points with specific edge cases or failure scenarios.",
            turn_number=2,
            max_tokens=300
        )
        debate_log.append(f"🔬 TESTER:\n{tester_r1}\n")
        context += f"TESTER: {tester_r1}\n\n"
        
        time.sleep(2)
        
        # OPTIMIZER responds
        print("  ⚡ OPTIMIZER thinking...")
        optimizer_r1 = self._call_persona(
            "OPTIMIZER",
            self.optimizer_system,
            context + "ARCHITECT and TESTER have spoken. Respond with pragmatic concerns about complexity vs. benefits.",
            turn_number=3,
            max_tokens=300
        )
        debate_log.append(f"⚡ OPTIMIZER:\n{optimizer_r1}\n")
        context += f"OPTIMIZER: {optimizer_r1}\n\n"
        
        time.sleep(2)
        
        # ROUND 2: Deeper exploration (each persona speaks again)
        debate_log.append("\n--- ROUND 2: EXPLORING TRADE-OFFS ---\n")
        
        print("  🏗️  ARCHITECT refining...")
        architect_r2 = self._call_persona(
            "ARCHITECT",
            self.architect_system,
            context + "TESTER and OPTIMIZER raised important points. Refine or adapt your thinking based on their feedback.",
            turn_number=4,
            max_tokens=300
        )
        debate_log.append(f"🏗️ ARCHITECT:\n{architect_r2}\n")
        context += f"ARCHITECT: {architect_r2}\n\n"
        
        time.sleep(2)
        
        print("  🔬 TESTER probing...")
        tester_r2 = self._call_persona(
            "TESTER",
            self.tester_system,
            context + "ARCHITECT adapted their position. Are your edge case concerns addressed? Probe deeper or raise new concerns.",
            turn_number=5,
            max_tokens=300
        )
        debate_log.append(f"🔬 TESTER:\n{tester_r2}\n")
        context += f"TESTER: {tester_r2}\n\n"
        
        time.sleep(2)
        
        print("  ⚡ OPTIMIZER evaluating...")
        optimizer_r2 = self._call_persona(
            "OPTIMIZER",
            self.optimizer_system,
            context + "The debate has evolved. Evaluate the trade-offs being discussed. Are we overcomplicating or missing something?",
            turn_number=6,
            max_tokens=300
        )
        debate_log.append(f"⚡ OPTIMIZER:\n{optimizer_r2}\n")
        context += f"OPTIMIZER: {optimizer_r2}\n\n"
        
        time.sleep(2)
        
        # ROUND 3: Convergence (final thoughts from each)
        debate_log.append("\n--- ROUND 3: CONVERGING ON SOLUTION ---\n")
        
        print("  🏗️  ARCHITECT synthesizing...")
        architect_r3 = self._call_persona(
            "ARCHITECT",
            self.architect_system,
            context + "Final round. Synthesize what you've learned. What's your recommended approach given everyone's input?",
            turn_number=7,
            max_tokens=350
        )
        debate_log.append(f"🏗️ ARCHITECT:\n{architect_r3}\n")
        context += f"ARCHITECT: {architect_r3}\n\n"
        
        time.sleep(2)
        
        print("  🔬 TESTER final checks...")
        tester_r3 = self._call_persona(
            "TESTER",
            self.tester_system,
            context + "Final checks. Are the critical edge cases covered in the emerging solution? What must not be forgotten?",
            turn_number=8,
            max_tokens=350
        )
        debate_log.append(f"🔬 TESTER:\n{tester_r3}\n")
        context += f"TESTER: {tester_r3}\n\n"
        
        time.sleep(2)
        
        print("  ⚡ OPTIMIZER final verdict...")
        optimizer_r3 = self._call_persona(
            "OPTIMIZER",
            self.optimizer_system,
            context + "Final verdict. Is the emerging solution pragmatic and performant? What's your final recommendation?",
            turn_number=9,
            max_tokens=350
        )
        debate_log.append(f"⚡ OPTIMIZER:\n{optimizer_r3}\n")
        
        debate_log.append("\n" + "="*70)
        debate_log.append("DEBATE CONCLUDED")
        debate_log.append("="*70)
        
        full_debate = "\n".join(debate_log)
        return full_debate


class SocraticCodeGenerator:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        print(f"🤖 Socratic Generator initialized")
        
    def _clean_generated_code(self, code: str) -> str:
        """Clean common issues in LLM-generated code"""
        code = re.sub(r'^```python\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        code = code.replace("```python", "").replace("```", "")
        
        lines = code.split('\n')
        cleaned_lines = []
        
        skip_phrases = [
            "here's", "here is", "i'll", "let me", "this code",
            "the following", "below is", "this implementation"
        ]
        
        for line in lines:
            line_lower = line.lower().strip()
            if any(phrase in line_lower for phrase in skip_phrases) and not line.strip().startswith('#'):
                continue
            if not cleaned_lines and not line.strip():
                continue
            cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines).strip()
        
        last_def_index = -1
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                last_def_index = i
        
        if last_def_index > 0:
            for i in range(last_def_index + 1, len(lines)):
                if lines[i] and not lines[i][0].isspace() and not lines[i].startswith('#'):
                    if not lines[i].strip().startswith('def ') and not lines[i].strip().startswith('class '):
                        code = '\n'.join(lines[:i]).strip()
                        break
        
        return code
    
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
                        try:
                            error_data = response.json()
                            if 'message' in error_data:
                                match = re.search(r'try again in (\d+)ms', error_data['message'])
                                if match:
                                    wait_ms = int(match.group(1))
                                    wait_time = (wait_ms / 1000.0) + 1.0
                                else:
                                    wait_time = base_delay * (2 ** attempt)
                            else:
                                wait_time = base_delay * (2 ** attempt)
                        except:
                            wait_time = base_delay * (2 ** attempt)
                        
                        print(f"  ⏳ Rate limit (attempt {attempt+1}/{max_retries}), waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "Error: Rate limit exceeded"
                
                if response.status_code != 200:
                    return f"Error: API Error {response.status_code}"
                
                response_data = response.json()
                if "choices" not in response_data:
                    return "Error: Unexpected API response"
                
                return response_data["choices"][0]["message"]["content"].strip()
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (attempt + 1)
                    time.sleep(wait_time)
                    continue
                return f"Error: {str(e)}"
        
        return "Error: Max retries exceeded"
    
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
        """Run multi-turn debate between three personas"""
        print("\n💬 Conducting multi-turn debate...")
        start = time.time()
        
        # Create debate manager
        debate_manager = MultiTurnDebate(self._make_api_call)
        
        # Run multi-turn debate
        debate = debate_manager.conduct_multi_turn_debate(problem, plan)
        
        elapsed = time.time() - start
        print(f"  ✓ Multi-turn debate completed ({elapsed:.1f}s)")
        
        return debate, elapsed
    
    def synthesize_code(self, problem: str, debate: str) -> Tuple[str, float]:
        """Generate code from debate"""
        print("\n⚙️  Synthesizing code from debate...")
        start = time.time()
        
        prompt = f"""You are an expert Python programmer implementing the solution agreed upon in a debate.

ORIGINAL PROBLEM:
{problem}

DEBATE CONSENSUS (last part):
{debate[-1500:]}

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
    
    def generate(self, problem: str) -> Tuple[dict, float]:
        """Full Socratic generation process"""
        print(f"\n{'='*70}")
        print(f"SOCRATIC CODE GENERATION")
        print(f"{'='*70}")
        
        total_start = time.time()
        results = {}
        
        # Step 1: Plan
        plan, t1 = self.plan_debate(problem)
        results['plan'] = plan
        results['plan_time'] = t1
        
        time.sleep(3)  # Cooldown
        
        # Step 2: Debate
        debate, t2 = self.conduct_debate(problem, plan)
        results['debate'] = debate
        results['debate_time'] = t2
        
        time.sleep(3)  # Cooldown
        
        # Step 3: Synthesize
        code, t3 = self.synthesize_code(problem, debate)
        results['code'] = code
        results['synthesis_time'] = t3
        
        results['total_time'] = time.time() - total_start
        
        return results, results['total_time']
    
    def save_results(self, results: dict, code_file: str = "socratic_code.py", 
                     debate_file: str = "debate_transcript.txt"):
        """Save generated code and debate"""
        # Save code
        with open(code_file, "w", encoding='utf-8') as f:
            f.write("# SOCRATIC METHOD CODE\n")
            f.write("# Generated after debate synthesis\n\n")
            f.write(results['code'])
        print(f"  ✓ Saved code to {code_file}")
        
        # Save debate transcript
        with open(debate_file, "w", encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("SOCRATIC DEBATE TRANSCRIPT\n")
            f.write("="*70 + "\n\n")
            f.write("DEBATE PLAN:\n")
            f.write("-"*70 + "\n")
            f.write(results['plan'] + "\n\n")
            f.write("-"*70 + "\n")
            f.write("FULL DEBATE:\n")
            f.write("-"*70 + "\n")
            f.write(results['debate'] + "\n")
        print(f"  ✓ Saved debate to {debate_file}")


if __name__ == "__main__":
    generator = SocraticCodeGenerator()
    
    problem = """
Implement a thread-safe LRU (Least Recently Used) cache with TTL (Time To Live).

Requirements:
- Support get(key) and put(key, value) operations
- Maximum capacity that evicts least recently used items when full
- Each entry has a TTL; expired entries should not be returned
- Must be thread-safe for concurrent access
- O(1) time complexity for both operations
"""
    
    results, timing = generator.generate(problem)
    
    print("\n" + "="*70)
    print("DEBATE EXCERPT:")
    print("="*70)
    print(results['debate'][:500] + "...")
    
    print("\n" + "="*70)
    print("GENERATED CODE:")
    print("="*70)
    print(results['code'][:500] + "..." if len(results['code']) > 500 else results['code'])
    
    print(f"\nTotal time: {timing:.2f}s")
    
    generator.save_results(results)