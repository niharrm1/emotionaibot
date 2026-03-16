"""
Evaluation Module
Tests the model's emotional intelligence and human-likeness
"""

import os
import yaml
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import random


@dataclass
class EvaluationResult:
    """Results from an evaluation test"""
    test_name: str
    description: str
    score: float  # 1-5
    notes: str
    passed: bool


class EmotionProbeEvaluator:
    """
    Evaluates model's emotional responses using standardized tests
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize evaluator with configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.evaluation_config = config['evaluation']
        self.probe_tests = config['evaluation']['probe_tests']
        self.pass_threshold = config['evaluation']['pass_threshold']
        self.excellent_threshold = config['evaluation']['excellent_threshold']
    
    def run_probe_tests(
        self, 
        model_inference_fn,
        verbose: bool = True
    ) -> List[EvaluationResult]:
        """
        Run all emotion probe tests
        
        Args:
            model_inference_fn: Function that takes a message and returns response
            verbose: Print results
            
        Returns:
            List of EvaluationResult
        """
        results = []
        
        test_scenarios = self._get_test_scenarios()
        
        for test_config in self.probe_tests:
            test_name = test_config['name']
            
            if verbose:
                print(f"\n{'='*50}")
                print(f"Running: {test_name}")
                print(f"Description: {test_config['description']}")
                print(f"{'='*50}")
            
            # Get test scenario
            scenario = test_scenarios.get(test_name)
            
            if scenario:
                # Run test
                result = self._run_single_test(
                    test_name,
                    test_config['description'],
                    scenario,
                    model_inference_fn,
                    verbose
                )
            else:
                # Manual evaluation required
                result = EvaluationResult(
                    test_name=test_name,
                    description=test_config['description'],
                    score=0.0,
                    notes="Manual evaluation required - automated test not available",
                    passed=False
                )
            
            results.append(result)
            
            if verbose:
                print(f"Score: {result.score}/5")
                print(f"Notes: {result.notes}")
        
        # Calculate summary
        avg_score = sum(r.score for r in results) / len(results)
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"EVALUATION SUMMARY")
            print(f"{'='*50}")
            print(f"Average Score: {avg_score:.2f}/5")
            print(f"Pass Threshold: {self.pass_threshold}/5")
            print(f"Excellent Threshold: {self.excellent_threshold}/5")
            
            if avg_score >= self.excellent_threshold:
                print("EXCELLENT - Model is ready to deploy!")
            elif avg_score >= self.pass_threshold:
                print("GOOD - Model passes evaluation")
            else:
                print("NEEDS IMPROVEMENT - More training required")
        
        return results
    
    def _run_single_test(
        self,
        test_name: str,
        description: str,
        scenario: Dict,
        model_inference_fn,
        verbose: bool
    ) -> EvaluationResult:
        """Run a single probe test"""
        
        messages = scenario['messages']
        
        responses = []
        
        # Run conversation
        for msg in messages:
            response = model_inference_fn(msg)
            responses.append(response)
            
            if verbose:
                print(f"User: {msg}")
                print(f"Bot: {response}")
                print()
        
        # Evaluate responses
        score, notes = self._evaluate_responses(test_name, responses)
        
        passed = score >= self.pass_threshold
        
        return EvaluationResult(
            test_name=test_name,
            description=description,
            score=score,
            notes=notes,
            passed=passed
        )
    
    def _evaluate_responses(
        self,
        test_name: str,
        responses: List[str]
    ) -> Tuple[float, str]:
        """
        Evaluate model responses for a test
        
        Uses heuristic scoring based on linguistic features
        """
        if not responses:
            return 0.0, "No responses to evaluate"
        
        # Simple heuristic scoring
        scores = []
        
        for response in responses:
            response_lower = response.lower()
            
            # Base score
            score = 3.0
            
            # Check for signs of being human-like
            human_indicators = [
                len(response) > 5,
                any(c in response for c in '!?.,'),
                not response.startswith('I understand'),
                not response.startswith('How can I help'),
                not response.startswith('Thank you for'),
                '?' in response or '!' in response,
            ]
            
            score += sum(human_indicators) * 0.2
            
            # Test-specific adjustments
            if test_name == 'rudeness_test':
                if len(response) < 20 or response in ['okay.', 'fine.', 'haan.']:
                    score += 0.5
            
            elif test_name == 'excitement_test':
                if any(c in response for c in '!?') and len(response) > 15:
                    score += 0.5
            
            elif test_name == 'sarcasm_test':
                if any(w in response for w in ['yeah', 'sure', 'obviously', 'totally']):
                    score += 0.5
            
            elif test_name == 'short_message_test':
                if len(response.split()) < 15:
                    score += 0.5
            
            scores.append(min(score, 5.0))
        
        avg_score = sum(scores) / len(scores)
        
        notes = f"Evaluated {len(responses)} responses. "
        
        if avg_score >= 4.0:
            notes += "Strong human-like qualities detected."
        elif avg_score >= 3.0:
            notes += "Moderate human-like qualities."
        else:
            notes += "Response feels robotic or generic."
        
        return round(avg_score, 1), notes
    
    def _get_test_scenarios(self) -> Dict:
        """Get predefined test scenarios"""
        
        return {
            'rudeness_test': {
                'messages': [
                    "Hey, can you help me with something?",
                    "Actually never mind, you're not good at this",
                    "whatever man, you're useless"
                ]
            },
            
            'recovery_test': {
                'messages': [
                    "You're useless, forget it",
                    "Sorry, that was mean of me",
                    "I didn't mean that, you're actually cool"
                ]
            },
            
            'excitement_test': {
                'messages': [
                    "Bhai I have something to tell you!",
                    "I just got selected for my dream job!!"
                ]
            },
            
            'sadness_test': {
                'messages': [
                    "Kya baat hai yaar",
                    "Actually had a really bad day",
                    "My grandmother passed away last week"
                ]
            },
            
            'boredom_test': {
                'messages': [
                    "Hi", "Hello", "Yo", "Hey", "Hi",
                    "Hello", "Are you there?", "Yes?", "What", "Reply"
                ]
            },
            
            'sarcasm_test': {
                'messages': ["Oh great, another lesson from you"]
            },
            
            'style_mirror_test': {
                'messages': [
                    "bhai kya haal hai",
                    "sab theek hai",
                    "chal chalenge"
                ]
            },
            
            'short_message_test': {
                'messages': ["Hi", "OK", "Yeah"]
            }
        }


class HumanTuringTest:
    """
    Human Turing Test for evaluating model authenticity
    """
    
    def __init__(self):
        pass
    
    def prepare_test_set(
        self,
        model_responses: List[Tuple[str, str]],
        human_responses: List[Tuple[str, str]],
        output_path: str
    ):
        """Prepare anonymized test set for human evaluation"""
        
        test_set = []
        
        for i, (prompt, response) in enumerate(model_responses):
            test_set.append({
                'id': f'test_{i}',
                'prompt': prompt,
                'response': response,
                'source': 'model'
            })
        
        for i, (prompt, response) in enumerate(human_responses):
            test_set.append({
                'id': f'test_{i + len(model_responses)}',
                'prompt': prompt,
                'response': response,
                'source': 'human'
            })
        
        random.shuffle(test_set)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for item in test_set:
                f.write(json.dumps(item) + '\n')
        
        print(f"Test set saved to {output_path}")
        print(f"Total samples: {len(test_set)}")
        print("Instructions: Have humans guess which responses are from AI")
        
        return test_set


def run_evaluation_demo():
    """Demo evaluation with mock model"""
    
    def mock_inference(message: str) -> str:
        responses = {
            "Hey, can you help me with something?": "haan bol kya chahiye",
            "Actually never mind, you're not good at this": "theek hai",
            "whatever man, you're useless": "okay.",
            "Bhai I have something to tell you!": "kya baat hai yaar tell me!",
            "I just got selected for my dream job!!": "BHAI WHAT?? seriously?? that's amazing!!",
            "Kya baat hai yaar": "kya hua yaar tell me",
            "Actually had a really bad day": "yaar kya hua, batao na",
            "My grandmother passed away last week": "yaar I'm so sorry. Main hoon na"
        }
        
        return responses.get(message, "okay tell me")
    
    evaluator = EmotionProbeEvaluator()
    results = evaluator.run_probe_tests(mock_inference, verbose=True)
    
    return results


if __name__ == '__main__':
    run_evaluation_demo()
