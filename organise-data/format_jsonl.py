import json
import re
import os

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            # Parse the input JSON line
            entry = json.loads(line.strip())
            
            # Initialize the output record with filepath and duration
            output_record = {
                "audio_filepath": entry["audio_filepath"],
                "duration": entry["duration"]
            }
            
            # Extract metadata from the text
            text = entry["text"]
            
            # Extract age group - handle both formats: AGE_XX_YY and AGE_60PLUS
            age_range_match = re.search(r'AGE_(\d+)_(\d+)', text)
            age_plus_match = re.search(r'AGE_(\d+)PLUS', text)
            
            if age_range_match:
                age_start, age_end = age_range_match.groups()
                output_record["age_group"] = f"{age_start}-{age_end}"
                text = re.sub(r'AGE_\d+_\d+', '', text)
            elif age_plus_match:
                age_start = age_plus_match.group(1)
                output_record["age_group"] = f"{age_start}+"
                text = re.sub(r'AGE_\d+PLUS', '', text)
            
            # Extract gender
            gender_match = re.search(r'GER_(FEMALE|MALE)', text)
            if gender_match:
                gender = gender_match.group(1)
                output_record["gender"] = "Female" if gender == "FEMALE" else "Male"
                text = re.sub(r'GER_(FEMALE|MALE)', '', text)
            
            # Extract emotion
            emotion_match = re.search(r'EMOTION_(\w+)', text)
            if emotion_match:
                emotion = emotion_match.group(1)
                output_record["emotion"] = emotion
                text = re.sub(r'EMOTION_\w+', '', text)
            
            # Extract intent
            intent_match = re.search(r'INTENT_(\w+)', text)
            if intent_match:
                intent = intent_match.group(1)
                output_record["intent"] = intent
                text = re.sub(r'INTENT_\w+', '', text)
            
            # Clean up extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            output_record["text"] = text
            
            # Write to output file
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')

def main():
    input_file = "input.jsonl"
    output_file = "output.jsonl"
    
    
    process_jsonl(input_file, output_file)
    print(f"Conversion complete. Output saved to {output_file}")
    
    # Display samples of the output
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print("\nSample outputs:")
        for i, line in enumerate(lines[:2]):
            print(f"\nSample {i+1}:")
            print(json.dumps(json.loads(line), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()