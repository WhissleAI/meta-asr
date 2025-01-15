from typing import List, Dict, Union, Optional, Set
from dataclasses import dataclass
import pandas as pd
import json
import re
from pathlib import Path
from functools import lru_cache

@dataclass(frozen=True)
class NERTags:
    """Constants for NER tag processing with validation methods."""
    PREFIX: str = 'NER_'
    END: str = 'END'
    SEPARATOR: str = ' '
    
    @classmethod
    def is_special_tag(cls, word: str) -> bool:
        """
        Check if a word is a special tag that should always be uppercase.
        
        Args:
            word: Word to check
            
        Returns:
            bool: True if word is a special tag
        """
        return (word.upper() == cls.END or 
                word.upper().startswith(cls.PREFIX) or
                word.upper().startswith('AGE_') or
                word.upper().startswith('GENDER_') or
                word.upper().startswith('EMOTION_') or
                word.upper() == 'SPEAKER_CHANGE')

class TextProcessor:
    """Handles text processing and NER tag formatting with optimization."""
    
    def __init__(self) -> None:
        """Initialize regex patterns and cached tag sets."""
        self._ner_pattern = re.compile(rf'{NERTags.PREFIX}\w+|{NERTags.END}')
        self._special_words: Set[str] = {NERTags.END, 'SPEAKER_CHANGE'}
    
    @staticmethod
    def _normalize_case(word: str) -> str:
        """
        Normalize case for special tags and regular words.
        
        Args:
            word: Input word
            
        Returns:
            str: Word with normalized case
        """
        if NERTags.is_special_tag(word):
            return word.upper()
        return word.lower()
    
    @lru_cache(maxsize=1024)
    def format_ner_text(self, text: Union[str, float, int]) -> str:
        """
        Format NER tags ensuring uppercase preservation while normalizing other text.
        Implements caching for repeated text patterns.
        
        Args:
            text: Input text containing NER tags (can be string, float, or int)
            
        Returns:
            str: Formatted text with uppercase NER tags and END markers
        """
        # Handle non-string input types
        if not isinstance(text, str):
            return str(text)
        
        # Split and process text
        words: List[str] = text.strip().split()
        formatted_words: List[str] = []
        i: int = 0
        
        while i < len(words):
            current_word = words[i]
            
            if current_word.upper().startswith(NERTags.PREFIX):
                # Process NER tag sequence
                formatted_words.append(current_word.upper())
                
                # Handle the entity if it exists
                if i + 1 < len(words):
                    formatted_words.append(words[i + 1].lower())
                    i += 2
                    
                    # Ensure END tag is uppercase if present
                    if (i < len(words) and 
                        words[i].upper() == NERTags.END):
                        formatted_words.append(NERTags.END)  # Explicitly uppercase
                        i += 1
                else:
                    i += 1
            else:
                # Apply case normalization for all words
                formatted_words.append(self._normalize_case(current_word))
                i += 1
        
        return NERTags.SEPARATOR.join(formatted_words)

def get_age_bucket(age: float) -> str:
    """
    Map age to predefined age buckets with validation.
    
    Args:
        age: Age value to be categorized (will be rounded to two decimals)
        
    Returns:
        str: Age bucket identifier in uppercase
    """
    actual_age: float = round(age * 100, 2)  # Convert from percentage to actual age
    
    age_brackets: List[tuple[float, str]] = [
        (18, "0_18"),
        (30, "18_30"),
        (45, "30_45"),
        (60, "45_60"),
        (float('inf'), "60PLUS")
    ]
    
    for threshold, bracket in age_brackets:
        if actual_age < threshold:
            return bracket.upper()
    return "60PLUS"  # Ensure consistent uppercase for age buckets

def process_csv_files(dir_path: Union[str, Path]) -> None:
    """
    Process all CSV files in the directory and convert to JSON format.
    
    Args:
        dir_path: Directory containing CSV files
        
    Raises:
        FileNotFoundError: If no CSV files are found in the directory
    """
    path = Path(dir_path)
    csv_files = list(path.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the specified directory.")
    
    text_processor = TextProcessor()
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        json_data: List[Dict[str, str]] = []
        
        for audio_path, group in df.groupby('Audio File Path'):
            text_segments: List[str] = []
            
            for _, row in group.iterrows():
                ner_text = text_processor.format_ner_text(row['NER_Tagged_Text'])
                
                text_segment = (
                    f"{ner_text} "
                    f"AGE_{get_age_bucket(row['Age'])} "
                    f"GENDER_{'MALE' if row['Gender'] == 1 else 'FEMALE'} "
                    f"EMOTION_{str(row['emotion']).upper()} "
                    f"SPEAKER_CHANGE"
                )
                text_segments.append(text_segment)
            
            audio_stem = Path(audio_path).stem
            json_data.append({
                "audio_filepath": f"audio_chunks/{audio_stem}/{audio_path}",
                "text": " ".join(text_segments).strip()
            })
        
        json_path = csv_file.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)
        
        print(f"Conversion complete. JSON saved as {json_path.name} in {dir_path}")

def main() -> None:
    """Main execution function with error handling."""
    try:
        dir_path = 'output2/results1'
        process_csv_files(dir_path)
    except Exception as e:
        print(f"Error processing files: {str(e)}")

if __name__ == "__main__":
    main()