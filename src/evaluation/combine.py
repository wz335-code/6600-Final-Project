import pandas as pd
import numpy as np
import textstat

#compute the flesch score based on the pre-defined range.
CEFR_EXPECTED_FLESCH = {
    'A2': {'center': 78, 'tolerance': 12},
    'B1': {'center': 65, 'tolerance': 10},
    'B2': {'center': 55, 'tolerance': 10},
}


def calculate_readability_metrics(text):
    """Calculate readability metrics for a text"""
    if pd.isna(text) or not text or len(str(text).strip()) == 0:
        return {'flesch_reading_ease': np.nan, 'flesch_kincaid_grade': np.nan}
    
    text = str(text)
    
    try:
        flesch_reading = textstat.flesch_reading_ease(text)
        flesch_grade = textstat.flesch_kincaid_grade(text)
    except:
        flesch_reading = np.nan
        flesch_grade = np.nan
    
    return {
        'flesch_reading_ease': flesch_reading,
        'flesch_kincaid_grade': flesch_grade
    }


def compute_level_appropriateness(text, target_level):
    """
    Compute how well a text matches its target CEFR level
    """
    metrics = calculate_readability_metrics(text)
    flesch = metrics['flesch_reading_ease']
    
    if pd.isna(flesch) or target_level not in CEFR_EXPECTED_FLESCH:
        return {
            'flesch': np.nan,
            'expected': np.nan,
            'deviation': np.nan,
            'level_match': np.nan,
            'appropriateness_score': np.nan
        }
    
    expected = CEFR_EXPECTED_FLESCH[target_level]['center']
    tolerance = CEFR_EXPECTED_FLESCH[target_level]['tolerance']
    
    # Calculate deviation from expected
    deviation = abs(flesch - expected)
    
    # Readability score
    readability_score = max(0, min(1, (flesch - 30) / 70))
    
    # Level match score
    if deviation <= tolerance:
        level_match = 1.0
    else:
        extra = deviation - tolerance
        level_match = max(0.7, 1.0 - (extra / 40))
    
    # 70% readability + 30% level match
    appropriateness_score = 0.7 * readability_score + 0.3 * level_match
    
    return {
        'flesch': flesch,
        'expected': expected,
        'deviation': deviation,
        'level_match': level_match,
        'appropriateness_score': appropriateness_score
    }
