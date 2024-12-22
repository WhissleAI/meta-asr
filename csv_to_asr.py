import pandas as pd
import numpy as np

def clean_csv(csv_path):
    df = pd.read_csv(csv_path)

    valid_rows = (
        df['Transcription'].notna() & 
        (df['Transcription'] != '') &
        df['Age'].notna() &
        (df['Age'] >= 0) & 
        (df['Age'] <= 100) &
        df['Gender'].isin([0, 1]) &
        df['emotion'].notna() &
        df['emotion'].str.lower().isin(['hap', 'ang', 'neu', 'sad'])
    )

    clean_df = df[valid_rows].copy()

    clean_csv_path = csv_path.replace('.csv', '_cleaned.csv')
    clean_df.to_csv(clean_csv_path, index=False)
    
    print(f"Original rows: {len(df)}")
    print(f"Cleaned rows: {len(clean_df)}")
    print(f"Removed {len(df) - len(clean_df)} invalid rows")
    
    return clean_csv_path

def get_age_bucket(age):
    age = float(age)
    if age < 18:
        return "0_18"
    elif age < 30:
        return "18_30"
    elif age < 45:
        return "30_45"
    elif age < 60:
        return "45_60"
    else:
        return "60plus"

def get_gender(gender):
    return "MALE" if int(gender) == 1 else "FEMALE"

def get_emotion(emotion):
    emotion_dict = {
        'hap': 'HAPPY',
        'ang': 'ANGRY',
        'neu': 'NEUTRAL',
        'sad': 'SAD'
    }
    return emotion_dict.get(emotion.lower(), emotion.upper())

def format_conversation(csv_path):
    df = pd.read_csv(csv_path)
    
    df = df.sort_values('Start Time')
    formatted_text = []
    current_speaker = None
    
    for _, row in df.iterrows():
        text = str(row['Transcription']).strip()
        if not text or text == 'nan':
            continue
            
        text = text.lower()
        age_bucket = f"AGE_{get_age_bucket(row['Age'])}"
        gender = f"GENDER_{get_gender(row['Gender'])}"
        emotion = f"EMOTION_{get_emotion(row['emotion'])}"
        
        speaker_change = ""
        if current_speaker != row['Speaker']:
            speaker_change = "SPEAKER_CHANGE "
            current_speaker = row['Speaker']
            
        formatted_segment = f"{text} {age_bucket} {gender} {emotion} {speaker_change}".strip()
        formatted_text.append(formatted_segment)
    
    return " ".join(formatted_text)

def process_csv_files(directory):
    import os
    
    os.makedirs(directory, exist_ok=True)
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            csv_path = os.path.join(directory, filename)
            print(f"\nProcessing {filename}...")
                
            clean_csv_path = clean_csv(csv_path)

            output_path = os.path.join(directory, f"{os.path.splitext(filename)[0]}_formatted.txt")
            formatted_text = format_conversation(clean_csv_path)
                
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
                    
            print(f"Created formatted text file: {output_path}")

if __name__ == "__main__":
    process_csv_files("csv_dir")
