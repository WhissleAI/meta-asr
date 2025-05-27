# import json
# from pathlib import Path

# def separate_jsonl(input_file, simple_output, complex_output, error_output):
#     """
#     Separates JSONL records into different files based on their structure and captures error cases.
    
#     Args:
#         input_file (str): Path to input JSONL file
#         simple_output (str): Path to output file for simple records
#         complex_output (str): Path to output file for complex records
#         error_output (str): Path to output file for lines that caused errors
#     """
#     base_fields = {'audio_filepath', 'text','duration'}
    
#     # Keep track of statistics
#     stats = {
#         'total_lines': 0,
#         'simple_records': 0,
#         'complex_records': 0,
#         'error_lines': 0
#     }
    
#     with open(input_file, 'r', encoding='utf-8') as f_in, \
#          open(simple_output, 'w', encoding='utf-8') as f_simple, \
#          open(complex_output, 'w', encoding='utf-8') as f_complex, \
#          open(error_output, 'w', encoding='utf-8') as f_error:
        
#         for line_number, line in enumerate(f_in, 1):
#             stats['total_lines'] += 1
#             try:
#                 record = json.loads(line.strip())
#                 record_fields = set(record.keys())
                
#                 # If record has only base fields, write to simple file
#                 if record_fields <= base_fields:
#                     json.dump(record, f_simple, ensure_ascii=False)
#                     f_simple.write('\n')
#                     stats['simple_records'] += 1
#                 # If record has additional fields, write to complex file
#                 else:
#                     json.dump(record, f_complex, ensure_ascii=False)
#                     f_complex.write('\n')
#                     stats['complex_records'] += 1
                    
#             except json.JSONDecodeError as e:
#                 stats['error_lines'] += 1
#                 # Write the problematic line along with error details to error file
#                 error_info = {
#                     'line_number': line_number,
#                     'error_message': str(e),
#                     'problematic_line': line.strip()
#                 }
#                 json.dump(error_info, f_error, ensure_ascii=False)
#                 f_error.write('\n')
#                 print(f"Error at line {line_number}: {e}")
    
#     # Print statistics
#     print("\nProcessing Statistics:")
#     print(f"Total lines processed: {stats['total_lines']}")
#     print(f"Simple records: {stats['simple_records']}")
#     print(f"Complex records: {stats['complex_records']}")
#     print(f"Error lines: {stats['error_lines']}")

# def main():
#     # Input and output file paths
#     input_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/europeen/cv/italain_cv_gem.jsonl"
#     simple_output = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/europeen/cv/simpe_it.jsonl"
#     complex_output = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/europeen/cv/comp_it_data.jsonl"
#     error_output = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/europeen/cv/error_it_data.jsonl"
    
#     # Ensure input file exists
#     if not Path(input_file).exists():
#         print(f"Error: Input file '{input_file}' does not exist")
#         return
    
#     # Create output directory if it doesn't exist
#     Path(simple_output).parent.mkdir(parents=True, exist_ok=True)
    
#     separate_jsonl(input_file, simple_output, complex_output, error_output)

# if __name__ == '__main__':
#     main()

import json
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import words
import nltk
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt')
    nltk.download('words')

# Create a set of English words for faster lookup
ENGLISH_WORDS = set(words.words())
# Add common words that might not be in NLTK's word list
ADDITIONAL_WORDS = {'ok', 'yeah', 'etc', 'gonna', 'wanna', 'hey', 'hi', 'hello', 'bye'}
ENGLISH_WORDS.update(ADDITIONAL_WORDS)

def is_entity_tag(word):
    """Check if a word is an entity tag or END tag."""
    return word.startswith('ENTITY_') or word == 'END'

def is_metadata_tag(word):
    """Check if a word is a metadata tag (e.g., AGE_0_18, GER_MALE, etc.)."""
    return any(word.startswith(prefix) for prefix in ['AGE_', 'GER_', 'EMOTION_', 'INTENT_'])

def remove_repeated_words(text):
    """Remove excessive repetitions of words while preserving entity tags and metadata."""
    words = text.split()
    cleaned_words = []
    i = 0
    
    while i < len(words):
        current_word = words[i]
        
        # If it's an entity tag or metadata, preserve it
        if is_entity_tag(current_word) or is_metadata_tag(current_word):
            cleaned_words.append(current_word)
            i += 1
            continue
            
        # Count consecutive occurrences of the current word
        count = 1
        while i + count < len(words) and words[i + count] == current_word:
            count += 1
            
        # Add the word (maximum 2 times if repeated)
        cleaned_words.extend([current_word] * min(count, 2))
        i += count
        
    return ' '.join(cleaned_words)

def is_english_word(word):
    """Check if a word is English or a special tag/number."""
    # Allow entity tags and metadata tags
    if is_entity_tag(word) or is_metadata_tag(word):
        return True
    
    # Allow numbers and basic punctuation
    if word.isdigit() or word in {',', '.', '!', '?', ';', ':', '-'}:
        return True
    
    # Check if it's an English word (case-insensitive)
    return word.lower() in ENGLISH_WORDS

def clean_text(text):
    # First, fix split entity tags by removing spaces within them
    text = re.sub(r'ENTITY_([A-Z]+)\s+([A-Z])\s*', r'ENTITY_\1\2 ', text)
    
    # Fix the placement of END tags
    text = re.sub(r'(ENTITY_[A-Z]+)\s+(\w+)\s+END', r'\1 \2 END', text)
    
    # Remove duplicated entity patterns
    pattern = r'(ENTITY_[A-Z]+\s+\w+\s+END)\s+\1'
    while re.search(pattern, text):
        text = re.sub(pattern, r'\1', text)
    
    # Remove excessive word repetitions
    text = remove_repeated_words(text)
    
    # Filter out non-English words while preserving entity tags and metadata
    words = text.split()
    cleaned_words = [word for word in words if is_english_word(word)]
    
    return ' '.join(cleaned_words).strip()

def process_jsonl_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line_number, line in enumerate(fin, 1):
            try:
                data = json.loads(line.strip())
                # Clean the text field
                data['text'] = clean_text(data['text'])
                # Write the cleaned data back to the output file
                fout.write(json.dumps(data) + '\n')
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line {line_number}: {line.strip()}")
            except Exception as e:
                print(f"Error processing line {line_number}: {str(e)}")

if __name__ == "__main__":
    input_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/english/vils_data_deduplicated.jsonl"
    output_file = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/english/vils.jsonl"
    process_jsonl_file(input_file, output_file)