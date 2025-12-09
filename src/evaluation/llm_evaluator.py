from openai import OpenAI
import os
from typing import List, Optional
from tqdm import tqdm
import pandas as pd
import time


class LLMEvaluator:
    """Use LLM as a judge to evaluate text quality"""
    
    def __init__(self, 
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize LLM evaluator
        
        Args:
            model: Model name to use (e.g., gpt-4, gpt-3.5-turbo)
            api_key: OpenAI API key (if not provided, read from environment variable)
            base_url: API base URL (optional, for custom endpoints)
        """
        self.model = model
        
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable or provide api_key parameter")
        
        # Initialize OpenAI client
        client_kwargs = {'api_key': api_key}
        if base_url:
            client_kwargs['base_url'] = base_url
            
        self.client = OpenAI(**client_kwargs)
        print(f"LLM evaluator initialized successfully, using model: {model}")
    
    def create_evaluation_prompt(self, original_text: str, rewritten_text: str, target_level: str = "A2") -> str:
        """
        Create evaluation prompt
        
        Args:
            original_text: Original text
            rewritten_text: Rewritten text
        
        Returns:
            Evaluation prompt
        """
        prompt = f"""You are a professional English language educator and CEFR (Common European Framework of Reference for Languages) expert. 

## CEFR Level Definitions:
- A2 (Elementary): Simple, clear language about familiar topics
- B1 (Intermediate): Main points of clear standard input, simple connected text
- B2 (Upper-Intermediate): Complex text, detailed on wide range of subjects

## Task:
The original text has been simplified to **{target_level} level**. Evaluate how well it matches.

**Original Text:**
{original_text}

**Simplified Text:**
{rewritten_text}

## Evaluation Criteria (rate each 0-5):
1. **Level Appropriateness**: Does the simplified text match the target CEFR level in terms of vocabulary and grammar complexity?
2. **Semantic Preservation**: Does it preserve the core meaning and key information from the original?
3. **Sentence Length**: Are sentence lengths appropriate for the target level (shorter for A1/A2, longer allowed for B1/B2)?
4. **Fluency**: Is the language natural and readable for learners at this level?
5. **Accuracy**: Is the information accurate without distortion or errors?

## Scoring Guide:
- 5: Excellent - Perfectly matches target level, fully preserves meaning, natural and accurate
- 4: Good - Mostly appropriate level, preserves main meaning, minor issues
- 3: Fair - Somewhat appropriate, some meaning loss or level mismatch
- 2: Poor - Significant level mismatch or meaning distortion
- 1: Very Poor - Major issues with level, meaning, or accuracy
- 0: Unacceptable - Completely inappropriate or incomprehensible

Please provide ONLY a single numeric score (0-5, decimals allowed like 3.5) as your overall assessment. No explanation needed."""

        return prompt
    
    def evaluate_single(self, 
                       original_text: str, 
                       rewritten_text: str,
                       max_retries: int = 3) -> Optional[float]:
        """
        Evaluate a single text pair
        
        Args:
            original_text: Original text
            rewritten_text: Rewritten text
            max_retries: Maximum number of retries
        
        Returns:
            Score (0-5), or None if failed
        """
        prompt = self.create_evaluation_prompt(original_text, rewritten_text)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                                {"role": "system", "content": "You are a professional English language educator specializing in CEFR-based text simplification assessment."},
                                {"role": "user", "content": prompt}
                            ],
                    temperature=0.7,
                    max_tokens=10
                )
                
                # Extract score
                score_text = response.choices[0].message.content.strip()
                
                # Try to parse the number
                import re
                numbers = re.findall(r'\d+\.?\d*', score_text)
                if numbers:
                    score = float(numbers[0])
                    # Ensure score is within 0-5 range
                    score = max(0, min(5, score))
                    return score
                else:
                    print(f"Unable to extract score from response: {score_text}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return None
                    
            except Exception as e:
                print(f"Evaluation error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None
        
        return None
    
    def evaluate_batch(self,
                      original_texts: List[str],
                      rewritten_texts: List[str],
                      show_progress: bool = True,
                      delay: float = 2.0) -> List[Optional[float]]:
        """
        Evaluate multiple text pairs in batch
        
        Args:
            original_texts: List of original texts
            rewritten_texts: List of rewritten texts
            show_progress: Whether to show progress bar
            delay: Delay between requests (seconds), to avoid rate limits
        
        Returns:
            List of scores
        """
        assert len(original_texts) == len(rewritten_texts), \
            "Number of original texts and rewritten texts must be equal"
        
        scores = []
        
        iterator = zip(original_texts, rewritten_texts)
        if show_progress:
            iterator = tqdm(iterator, total=len(original_texts),
                          desc="LLM evaluation")
        
        for i, (orig, rewritten) in enumerate(iterator):
            # Handle empty text cases
            if not orig or not rewritten or pd.isna(orig) or pd.isna(rewritten):
                scores.append(None)
            else:
                score = self.evaluate_single(str(orig), str(rewritten))
                scores.append(score)
            
            # Add delay to avoid rate limits (except for the last one)
            if i < len(original_texts) - 1 and delay > 0:
                time.sleep(delay)
        
        return scores
    
    def normalize_scores(self, scores: List[Optional[float]]) -> List[Optional[float]]:
        """
        Normalize scores to 0-1 range using min-max normalization
        
        Args:
            scores: List of scores (0-5 range)
        
        Returns:
            List of normalized scores (0-1 range)
        """
        # Filter out None values for calculation
        valid_scores = [s for s in scores if s is not None]
        
        if not valid_scores:
            return scores
        
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        
        # Avoid division by zero
        if max_score == min_score:
            return [1.0 if s is not None else None for s in scores]
        
        # Normalize each score
        normalized = []
        for score in scores:
            if score is None:
                normalized.append(None)
            else:
                norm_score = (score - min_score) / (max_score - min_score)
                normalized.append(norm_score)
        
        return normalized



