import pandas as pd
import numpy as np
import os
from typing import Optional, List
import argparse
from cosine_sim import CosineEvaluator
from llm_evaluator import LLMEvaluator
from dotenv import load_dotenv

# Import level appropriateness functions
from combine import (
    compute_level_appropriateness,
    CEFR_EXPECTED_FLESCH
)


class TextEvaluator:
    """Main text evaluation class"""
    
    def __init__(self, 
                 use_cosine: bool = True,
                 use_llm: bool = True,
                 cosine_model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 llm_model: str = 'gpt-4'):
        """
        Initialize evaluator
        
        Args:
            use_cosine: Whether to use cosine similarity evaluation
            use_llm: Whether to use LLM evaluation
            cosine_model: Model to use for cosine similarity
            llm_model: Model to use for LLM evaluation
        """
        self.use_cosine = use_cosine
        self.use_llm = use_llm
        
        # Initialize evaluators
        if self.use_cosine:
            self.cosine_sim = CosineEvaluator(model_name=cosine_model)
        
        if self.use_llm:
            self.llm_evaluator = LLMEvaluator(model=llm_model)
    
    def evaluate_csv(self,
                    input_path: str,
                    output_path: Optional[str] = None,
                    id_column: str = None,
                    original_column: str = None) -> pd.DataFrame:
        """
        Evaluate texts in CSV file
        
        Args:
            input_path: Input CSV file path
            output_path: Output CSV file path (auto-generated if None)
            id_column: ID column name (default: first column)
            original_column: Original text column name (default: second column)
        
        Returns:
            DataFrame containing evaluation results
        """
        # Read CSV file
        df = pd.read_csv(input_path)
        print(f"Columns: {list(df.columns)}\n")
        
        # Determine column names
        columns = df.columns.tolist()
        first_col_lower = columns[0].lower()
        is_first_col_original = any(keyword in first_col_lower for keyword in ['original', 'input', 'source', 'text'])
        
        if len(columns) == 2:
            original_column = columns[0]
            rewritten_columns = [columns[1]]
            print(f"Original text: {original_column}")
            print(f"Target text: {rewritten_columns[0]}\n")
        elif is_first_col_original or original_column == columns[0]:
            if original_column is None:
                original_column = columns[0]
            rewritten_columns = columns[1:]
            print(f"Original text: {original_column}")
            print(f"Detected {len(rewritten_columns)} rewritten text: {rewritten_columns}\n")
        else:
            # Standard format: ID, original, rewrite1, rewrite2, ...
            if id_column is None:
                id_column = columns[0]
            if original_column is None:
                original_column = columns[1]
            
            print(f"ID: {id_column}")
            print(f"Original text: {original_column}")
            
            # Get rewritten text columns (all columns starting from the third)
            id_col_idx = columns.index(id_column)
            original_col_idx = columns.index(original_column)
            
            # Find all rewritten text columns (excluding ID and original text columns)
            rewritten_columns = [col for i, col in enumerate(columns) 
                                if i != id_col_idx and i != original_col_idx]
            
            print(f"Detected {len(rewritten_columns)} rewritten text columns: {rewritten_columns}\n")
        
        # Create result DataFrame (preserve original data)
        result_df = df.copy()
        
        # Evaluate each rewritten text column
        for rewritten_col in rewritten_columns:
            print(f"eval the column: {rewritten_col}")
            
            # Cosine similarity evaluation
            if self.use_cosine:
                cosine_scores = self.cosine_sim.compute_similarities_batch(
                    df[original_column].tolist(),
                    df[rewritten_col].tolist()
                )
                result_df[f'{rewritten_col}_cosine_score'] = cosine_scores
                
                # Calculate statistics
                valid_scores = [s for s in cosine_scores if pd.notna(s)]
                if valid_scores:
                    print(f"avg score: {sum(valid_scores)/len(valid_scores):.4f}")
                    print(f"max score: {max(valid_scores):.4f}")
                    print(f"min score: {min(valid_scores):.4f}")
            
            # LLM evaluation
            if self.use_llm:
                llm_scores = self.llm_evaluator.evaluate_batch(
                    df[original_column].tolist(),
                    df[rewritten_col].tolist()
                )
                
                # Normalize scores to 0-1 range
                print(f"normalize the score to 0-1")
                normalized_scores = self.llm_evaluator.normalize_scores(llm_scores)
                result_df[f'{rewritten_col}_llm_score'] = normalized_scores
                
                # Calculate statistics (using normalized scores)
                valid_scores = [s for s in normalized_scores if s is not None]
                if valid_scores:
                    print(f"avg score: {sum(valid_scores)/len(valid_scores):.4f}")
                    print(f"max score: {max(valid_scores):.4f}")
                    print(f"min score: {min(valid_scores):.4f}")
            target_level = None
            for level in ['A2', 'B1', 'B2']:
                if rewritten_col.startswith(level):
                    target_level = level
                    break
            
            if target_level:
                print(f"Expected Flesch: {CEFR_EXPECTED_FLESCH[target_level]['center']} (±{CEFR_EXPECTED_FLESCH[target_level]['tolerance']})")
                
                # Compute level appropriateness for each text
                flesch_scores = []
                level_match_scores = []
                appropriateness_scores = []
                
                for text in df[rewritten_col]:
                    result = compute_level_appropriateness(text, target_level)
                    flesch_scores.append(result.get('flesch', np.nan))
                    level_match_scores.append(result.get('level_match', np.nan))
                    appropriateness_scores.append(result.get('appropriateness_score', np.nan))
                
                # Add columns formatted name
                result_df[f'{rewritten_col}_flesch_reading_ease'] = flesch_scores
                result_df[f'{rewritten_col}_level_match_score'] = level_match_scores
                result_df[f'{rewritten_col}_appropriateness_score'] = appropriateness_scores
                valid_flesch = [s for s in flesch_scores if pd.notna(s)]
                if valid_flesch:
                    avg_flesch = np.mean(valid_flesch)
                    expected = CEFR_EXPECTED_FLESCH[target_level]['center']
                    print(f"Actual Flesch: {avg_flesch:.1f} (deviation: {abs(avg_flesch - expected):.1f})")
                    print(f"Appropriateness Score: {np.nanmean(appropriateness_scores):.4f}")
                
                # Compute total average score
                if self.use_llm:
                    llm_col = f'{rewritten_col}_llm_score'
                    cosine_col = f'{rewritten_col}_cosine_score'
                    
                    if llm_col in result_df.columns and cosine_col in result_df.columns:
                        total_avg_scores = []
                        
                        for i in range(len(result_df)):
                            cosine_val = result_df[cosine_col].iloc[i]
                            approp_val = appropriateness_scores[i]
                            llm_val = result_df[llm_col].iloc[i]
                            
                            # Calculate average of all three scores
                            if pd.notna(cosine_val) and pd.notna(approp_val) and pd.notna(llm_val):
                                total_avg = (cosine_val + approp_val + llm_val) / 3
                            else:
                                total_avg = np.nan
                            
                            total_avg_scores.append(total_avg)
                        
                        result_df[f'{rewritten_col}_total_avg'] = total_avg_scores
                        print(f"   Total Average Score (Cosine+Approp+LLM)/3: {np.nanmean(total_avg_scores):.4f}")
        
        # Save results
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_evaluated.csv"
        
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return result_df


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(
        description='Text Rewriting Quality Evaluation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # both evaluation methods
  python evaluator.py input.csv
  
  # only cosine similarity
  python evaluator.py input.csv --no-llm
  
  # only LLM evaluation
  python evaluator.py input.csv --no-cosine
  
  # output file directory
  python evaluator.py input.csv -o output.csv
  
  # different LLM model
  python evaluator.py input.csv --llm-model gpt-3.5-turbo
        """
    )
    
    parser.add_argument('input_csv', help='input file path')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('--no-cosine', action='store_true', 
                       help='Do not use cosine similarity evaluation')
    parser.add_argument('--no-llm', action='store_true',
                       help='no LLM eval')
    parser.add_argument('--cosine-model', 
                       default='paraphrase-multilingual-MiniLM-L12-v2',
                       help='Cosine similarity model name')
    parser.add_argument('--llm-model', default='gpt-4o-mini',
                       help='LLM model name')
    parser.add_argument('--id-column', help='ID column name')
    parser.add_argument('--original-column', help='Original text column name')
    
    args = parser.parse_args()
    load_dotenv()
    
    # Check the input file
    if not os.path.exists(args.input_csv):
        print(f"Input file '{args.input_csv}' not found")
        return
    
    # evaluator
    evaluator = TextEvaluator(
        use_cosine=not args.no_cosine,
        use_llm=not args.no_llm,
        cosine_model=args.cosine_model,
        llm_model=args.llm_model
    )
    
    # evaluation
    evaluator.evaluate_csv(
        input_path=args.input_csv,
        output_path=args.output,
        id_column=args.id_column,
        original_column=args.original_column
    )


if __name__ == '__main__':
    main()

