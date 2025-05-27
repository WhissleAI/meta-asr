# import json
# import re
# import os
# import langdetect
# from collections import Counter

# def is_repetitive(text, threshold=10):
#     """
#     Detect if text contains significant repetition.
    
#     Args:
#         text (str): Text to check for repetition
#         threshold (int): Number of repetitions to consider problematic
        
#     Returns:
#         bool: True if repetitive content detected, False otherwise
#     """
#     # Split text into words
#     words = text.split()
    
#     # Check for empty text
#     if len(words) <= 1:
#         return False
        
#     # Count word frequencies
#     word_counts = Counter(words)
#     most_common_word, count = word_counts.most_common(1)[0]
    
#     # If any single word appears too frequently relative to text length
#     if count > len(words) * 0.3 and count >= threshold:
#         return True
    
#     # Check for consecutive repetitions
#     repetition_count = 1
#     max_repetition = 1
    
#     for i in range(1, len(words)):
#         if words[i] == words[i-1]:
#             repetition_count += 1
#         else:
#             max_repetition = max(max_repetition, repetition_count)
#             repetition_count = 1
    
#     # Check one more time for sequence ending with repetition
#     max_repetition = max(max_repetition, repetition_count)
    
#     # Check for repeating phrases
#     for phrase_len in range(2, min(6, len(words)//2 + 1)):
#         phrases = [" ".join(words[i:i+phrase_len]) for i in range(len(words) - phrase_len + 1)]
#         phrase_counts = Counter(phrases)
        
#         if phrase_counts and phrase_counts.most_common(1)[0][1] >= threshold:
#             return True
    
#     return max_repetition >= threshold

# def contains_sound_description(text):
#     """
#     Check if text contains any kind of sound descriptions.
#     Enhanced to catch more variations of sound descriptions.
#     """
#     # Check for square bracket sound descriptions
#     if re.search(r'\[SOUND\s+of', text, re.IGNORECASE) or re.search(r'\[SOUND\]', text, re.IGNORECASE):
#         return True
    
#     # Check for parenthetical sound descriptions
#     parentheses_pattern = r'\([^)]*(?:No|p|P|Pppppp|Sound|SOUND|Inaudible|Silence|sound|noise|music|cough|laugh|background|rustling)[^)]*\)'
#     if re.search(parentheses_pattern, text, re.IGNORECASE):
#         return True
    
#     # Check for "Pppppp" outside of parentheses
#     if re.search(r'P', text, re.IGNORECASE):
#         return True
    
#     # Check for other common sound indicators
#     other_indicators = [
#         r'\b(?:sound\s+of|background\s+noise)\b',
#         r'\[(?:inaudible|silence|music|noise)\]',
#         r'\*(?:sound|noise|music)\*'
#     ]
    
#     for pattern in other_indicators:
#         if re.search(pattern, text, re.IGNORECASE):
#             return True
    
#     return False

# def is_english(text):
#     """
#     Detect if text is in English.
#     Uses langdetect library with error handling.
#     """
#     # Skip empty or very short texts
#     if not text or len(text.strip()) < 3:
#         return True
    
#     try:
#         # Check if text contains non-Latin characters
#         if re.search(r'[^\x00-\x7F\s]', text):
#             return False
            
#         # For longer text, use language detection
#         if len(text.strip()) > 20:
#             return langdetect.detect(text) == 'en'
#         return True
#     except langdetect.LangDetectException:
#         # If detection fails, let's be cautious and keep the text
#         return True

# def clean_jsonl_file(input_file, output_file, removed_log_file):
#     """
#     Clean a JSONL file by removing problematic entries.
    
#     Args:
#         input_file (str): Path to input JSONL file
#         output_file (str): Path to output cleaned JSONL file
#         removed_log_file (str): Path to log file for removed entries
#     """
#     removed_count = 0
#     kept_count = 0
#     removal_reasons = {
#         "repetitive": 0,
#         "non_english": 0,
#         "sound_description": 0
#     }
    
#     with open(input_file, 'r', encoding='utf-8') as f_in, \
#          open(output_file, 'w', encoding='utf-8') as f_out, \
#          open(removed_log_file, 'w', encoding='utf-8') as f_log:
        
#         f_log.write("Removed file paths and reasons:\n")
        
#         for line_num, line in enumerate(f_in, 1):
#             try:
#                 # Skip empty lines
#                 if not line.strip():
#                     continue
                
#                 # Parse the JSON object
#                 data = json.loads(line.strip())
                
#                 # Extract text and file path
#                 text = data.get("text", "")
#                 file_path = data.get("audio_filepath", f"Unknown path (line {line_num})")
                
#                 # Track if this entry should be removed
#                 remove_entry = False
#                 reason = ""
                
#                 # Check for repetitive content
#                 if is_repetitive(text):
#                     remove_entry = True
#                     reason = "repetitive text"
#                     removal_reasons["repetitive"] += 1
                
#                 # Check for sound descriptions (enhanced version)
#                 elif contains_sound_description(text):
#                     remove_entry = True
#                     reason = "contains sound description"
#                     removal_reasons["sound_description"] += 1
                
#                 # Check if not English
#                 elif not is_english(text):
#                     remove_entry = True
#                     reason = "non-English text"
#                     removal_reasons["non_english"] += 1
                
#                 if remove_entry:
#                     # Log the removed file with reason
#                     f_log.write(f"{file_path} - {reason}\n")
#                     removed_count += 1
#                 else:
#                     # Keep this entry
#                     f_out.write(line)
#                     kept_count += 1
                    
#             except json.JSONDecodeError:
#                 print(f"Warning: Skipping invalid JSON on line {line_num}")
    
#     print(f"Processing complete!")
#     print(f"Entries kept: {kept_count}")
#     print(f"Entries removed: {removed_count}")
#     print(f"Removal reasons:")
#     print(f"  - Repetitive text: {removal_reasons['repetitive']}")
#     print(f"  - Sound descriptions: {removal_reasons['sound_description']}")
#     print(f"  - Non-English text: {removal_reasons['non_english']}")
#     print(f"Removed file paths logged to: {removed_log_file}")


# if __name__ == "__main__":
#     # File paths
#     input_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/cleaned1_hf.jsonl"  # Replace with your input file path
#     output_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/cleaned2_hf.jsonl"
#     removed_log_file = "removed1_files.txt"
    
#     # Run the cleaning process
#     clean_jsonl_file(input_file, output_file, removed_log_file)

import json
import re
import os

def contains_p_pattern(text):
    """
    Check if text contains any of the specified P patterns.
    More precise matching to avoid removing too much data.
    
    Args:
        text (str): Text to check for P patterns
        
    Returns:
        bool: True if any P pattern is found, False otherwise
    """
    # Define the specific patterns to look for
    p_patterns = [
        r'\bP\s+p\s+p\s+p\s+p\s+p\b',  # Exact "P p p p p p" pattern
        r'\bP\s+P\s+P\b',              # Exact "P P P" pattern
        r'\bP\s+p\s+p\b',              # Exact "P p p" pattern
        r'\bP\s+p\b',                  # Exact "P p" pattern
        r'"P"',                        # "P" in quotes
        r'\(P\)',
        r'\bp',
        r'\bp\s+p\s+p\b',
        r'\b\s+p\s+p\s+p\s+p\s+p\b',
        # (P) in parentheses
    ]
    
    # Check each pattern
    for pattern in p_patterns:
        if re.search(pattern, text):
            return True
    
    # Special case for standalone P - be more careful
    # Only match if it's a standalone P (not part of a word)
    if re.search(r'\b[P]\b', text):
        # Make sure it's not part of a sentence
        words = text.split()
        if 'P' in words:
            # Check if it's surrounded by punctuation or appears alone
            for i, word in enumerate(words):
                if word == 'P':
                    # Check if it's alone or surrounded by punctuation
                    if len(words) == 1 or (i > 0 and words[i-1][-1] in '.!?,:;') or (i < len(words)-1 and words[i+1][0] in '.!?,:;'):
                        return True
    
    return False

def clean_jsonl_file(input_file, output_file, removed_log_file):
    """
    Clean a JSONL file by removing entries with P patterns.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output cleaned JSONL file
        removed_log_file (str): Path to log file for removed entries
    """
    removed_count = 0
    kept_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out, \
         open(removed_log_file, 'w', encoding='utf-8') as f_log:
        
        f_log.write("Removed file paths containing P patterns:\n")
        
        for line_num, line in enumerate(f_in, 1):
            try:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Parse the JSON object
                data = json.loads(line.strip())
                
                # Extract text and file path
                text = data.get("text", "")
                file_path = data.get("audio_filepath", f"Unknown path (line {line_num})")
                
                # Check if text contains any P pattern
                if contains_p_pattern(text):
                    # Log the removed file with sample text
                    f_log.write(f"{file_path} - text sample: {text[:100]}...\n")
                    removed_count += 1
                else:
                    # Keep this entry
                    f_out.write(line)
                    kept_count += 1
                    
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON on line {line_num}")
    
    print(f"Processing complete!")
    print(f"Entries kept: {kept_count}")
    print(f"Entries removed: {removed_count}")
    print(f"Removed file paths logged to: {removed_log_file}")

def clean_text_file(input_file, output_file, removed_log_file):
    """
    Clean a plain text file by removing lines with P patterns.
    
    Args:
        input_file (str): Path to input text file
        output_file (str): Path to output cleaned text file
        removed_log_file (str): Path to log file for removed lines
    """
    removed_count = 0
    kept_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out, \
         open(removed_log_file, 'w', encoding='utf-8') as f_log:
        
        f_log.write("Removed lines containing P patterns:\n")
        
        for line_num, line in enumerate(f_in, 1):
            # Skip empty lines
            if not line.strip():
                continue
            
            # Check if line contains any P pattern
            if contains_p_pattern(line):
                # Log the removed line
                f_log.write(f"Line {line_num} - text: {line.strip()}\n")
                removed_count += 1
            else:
                # Keep this line
                f_out.write(line)
                kept_count += 1
    
    print(f"Processing complete!")
    print(f"Lines kept: {kept_count}")
    print(f"Lines removed: {removed_count}")
    print(f"Removed lines logged to: {removed_log_file}")

def main():
    """
    Main function to process either JSONL or TXT files.
    """
    print("P-Pattern Removal Tool")
    print("=====================")
    
    # Get input file
    input_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/main_hf_Data.jsonl"
    
    # Determine file type
    file_extension = os.path.splitext(input_file)[1].lower()
    
    # Create output file path
    output_file = os.path.splitext(input_file)[0] + "_cleaned" + file_extension
    removed_log_file = os.path.splitext(input_file)[0] + "_removed_p_patterns.log"
    
    # Process according to file type
    if file_extension == '.jsonl':
        clean_jsonl_file(input_file, output_file, removed_log_file)
    else:
        # Treat as text file
        clean_text_file(input_file, output_file, removed_log_file)
    
    print(f"Cleaned file saved to: {output_file}")
    print(f"Removal log saved to: {removed_log_file}")

if __name__ == "__main__":
    main()