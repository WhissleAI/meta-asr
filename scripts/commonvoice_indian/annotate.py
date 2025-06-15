import os
import json
import openai
import time
import csv

start_time = time.time()

# Set your OpenAI API key.
openai.api_key = ''

# File paths for input and output.
input_jsonl_path = "/external4/datasets/cv_scripts_data/ur/ur_validated_manifest.jsonl"
output_jsonl_path = "/external4/datasets/cv_scripts_data/ur/ur_validated_manifest_annotated.jsonl"
output_csv_path = "/external4/datasets/cv_scripts_data/ur/ur_validated_manifest_annotated.csv"

# Load records from the input JSONL file.
with open(input_jsonl_path, 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f if line.strip()]

# Initialize annotated records.
annotated_records = []
if os.path.exists(output_jsonl_path):
    with open(output_jsonl_path, 'r', encoding='utf-8') as f_json:
        annotated_records = [json.loads(line) for line in f_json if line.strip()]
    print(f"Loaded {len(annotated_records)} existing annotated records.")
else:
    print("No existing annotated records found. Starting fresh.")

# Create a set of already annotated record identifiers (using the 'audio_filepath' field).
annotated_filepaths = set(record['audio_filepath'] for record in annotated_records if 'audio_filepath' in record)

total_records = len(records)
print(f"Total records to process: {total_records}")
batch_size = 10  # Process 10 records per batch.
total_batches = (total_records + batch_size - 1) // batch_size
print(f"Processing in batches of {batch_size} records.")

def format_tag(tag_type, value):
    """
    Format a tag string, e.g., 'DURATION_3.5s'.
    """
    return f"{tag_type}_{value.upper().replace(' ', '_').replace('-', '_').replace('/', '_')}"

def annotate_sentences_with_intent(sentences):
    """
    Annotate each  sentence using the OpenAI gpt-4o-mini model.
    The annotation process marks entity phrases by inserting start tags ("ENTITY_<TYPE>")
    and a literal "END" after each entity phrase found in the sentence.
    
    The prompt instructs the model to only consider the sentence content.
    The function returns a list of annotated sentence strings.
    """
    
    prompt = f'''
Annotate each Urdu sentence from the input list by marking entity phrases with start and end tags using only the provided entity types. The annotation process MUST consider only the sentence content.

IMPORTANT: All instructions provided below must be followed exactly and strictly. Do not add any extra text, explanations, or formatting.

Instructions:
    1. Process each sentence independently.
    2. Identify all entity phrases that match one of the provided entity types. When determining which entities to annotate, consider both the sentence content and the provided intent.
    3. For each identified entity phrase, insert a start tag and an end tag:
         - The start tag must be in the format: ENTITY_<ENTITY_TYPE> (replace spaces with underscores).
         - The end tag is the literal string: END.
    4. Use only the entity types provided in the list. Do not introduce any additional entity types.
    5. Ensure the output is a JSON array containing only the annotated sentences in the same order as the input sentences, without any markdown or formatting.
    6. Do not add any additional text, explanations, or formatting.

Entity Types:
[ "PERSON NAME", "ORGANIZATION", "LOCATION", "DATE", "TIME", "DURATION", "EMAIL", "PHONE NUMBER", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP CODE", "CURRENCY", "PRICE", "PRODUCT", "SERVICE", "BRAND", "EVENT", "PERCENTAGE", "AGE", "TEMPERATURE", "MEASUREMENT", "DISTANCE", "WEIGHT", "HEIGHT", "VOLUME", "SPEED", "LANGUAGE", "NATIONALITY", "RELIGION", "JOB TITLE", "COMPANY NAME", "DEVICE NAME", "OPERATING SYSTEM", "SOFTWARE VERSION", "COLOR", "SHAPE", "MATERIAL", "MODEL NUMBER", "LICENSE PLATE", "VEHICLE MAKE", "VEHICLE MODEL", "VEHICLE TYPE", "FLIGHT NUMBER", "HOTEL NAME", "BOOKING REFERENCE", "PAYMENT METHOD", "CREDIT CARD NUMBER", "ACCOUNT NUMBER", "INSURANCE PROVIDER", "POLICY NUMBER", "BANK NAME", "TAX ID", "SOCIAL SECURITY NUMBER", "DRIVER'S LICENSE", "PASSPORT NUMBER", "WEBSITE", "URL", "IP ADDRESS", "MAC ADDRESS", "USERNAME", "PASSWORD", "FOOD ITEM", "DRINK ITEM", "CUISINE", "INGREDIENT", "DISH NAME", "MENU ITEM", "ORDER NUMBER", "PAYMENT AMOUNT", "DELIVERY TIME", "DELIVERY DATE", "APPOINTMENT DATE", "APPOINTMENT TIME", "ROOM NUMBER", "HOSPITAL NAME", "DOCTOR NAME", "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST NAME", "TEST RESULT", "INSURANCE PLAN", "CLAIM NUMBER", "POLICY HOLDER", "BENEFICIARY", "RELATIONSHIP", "EMERGENCY CONTACT", "PROJECT NAME", "TASK", "MEETING", "AGENDA", "ACTION ITEM", "DEADLINE", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT", "QUESTION", "RESPONSE" ]

Example:


Input: 
[
    "یہی تناسب یوتھ کا بھی ہے۔ AGE_18_30 GENDER_FEMALE EMOTION_NEUTRAL",
    "یہاں سیکڑوں کارواں اور بھی ہیں AGE_60PLUS GENDER_MALE EMOTION_NEUTRAL",
    "صبح کرنا شام کا، لانا ہے جوئےشِیر کا AGE_18_30 GENDER_FEMALE EMOTION_NEUTRAL",
    "اے آرزو شہیدِ وفا! خوں بہا نہ مانگ AGE_18_30 GENDER_MALE EMOTION_NEUTRAL",
    "دل میں ذوقِ وصل و یادِ یار تک باقی نہیں AGE_18_30 GENDER_FEMALE EMOTION_NEUTRAL"
]

Expected Output:
[
    "یہی ENTITY_PERCENTAGE تناسب END ENTITY_AGE_GROUP یوتھ END کا بھی ہے۔ AGE_18_30 GENDER_FEMALE EMOTION_NEUTRAL",
    "یہاں ENTITY_QUANTITY سیکڑوں END ENTITY_GROUP کارواں END اور بھی ہیں AGE_60PLUS GENDER_MALE EMOTION_NEUTRAL",
    "ENTITY_TIME صبح END کرنا ENTITY_TIME شام END کا، ENTITY_TASK لانا END ہے ENTITY_FOOD_ITEM جوئےشِیر END کا AGE_18_30 GENDER_FEMALE EMOTION_NEUTRAL",
    "اے ENTITY_FEELING آرزو END ENTITY_PERSON_NAME شہیدِ وفا END! ENTITY_PAYMENT خوں بہا END نہ ENTITY_REQUEST مانگ END AGE_18_30 GENDER_MALE EMOTION_NEUTRAL",
    "ENTITY_EMOTION دل میں ذوقِ وصل END و ENTITY_MEMORY یادِ یار END تک باقی نہیں AGE_18_30 GENDER_FEMALE EMOTION_NEUTRAL"
]    



Phrases to Annotate:
{json.dumps(sentences, ensure_ascii=False)}
'''
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            n=1,
            temperature=0.5,
        )
        assistant_reply = response.choices[0].message.content.strip()
        try:
            annotated_results = json.loads(assistant_reply)
            if isinstance(annotated_results, list):
                return annotated_results
            else:
                print("Assistant did not return a list. Retrying individual annotations...")
                new_results = []
                for sentence in sentences:
                    individual_result = annotate_sentences_with_intent([sentence])
                    if isinstance(individual_result, list) and len(individual_result) == 1:
                        new_results.append(individual_result[0])
                    else:
                        new_results.append(sentence)
                return new_results
        except json.JSONDecodeError:
            print("JSON decoding failed. Retrying individual annotations...")
            new_results = []
            for sentence in sentences:
                individual_result = annotate_sentences_with_intent([sentence])
                if isinstance(individual_result, list) and len(individual_result) == 1:
                    new_results.append(individual_result[0])
                else:
                    new_results.append(sentence)
            return new_results
    except Exception as e:
        print(f"Error annotating batch: {e}")
        return sentences

# Process records in batches.
for batch_num in range(total_batches):
    start_idx = batch_num * batch_size
    end_idx = min((batch_num + 1) * batch_size, total_records)
    print(f"\nProcessing batch {batch_num + 1}/{total_batches} (records {start_idx + 1} to {end_idx})...")

    sentences_to_annotate = []
    filepaths = []
    batch_records = []

    # Gather records in this batch that have not been processed yet.
    for idx, record in enumerate(records[start_idx:end_idx], start=start_idx + 1):
        audio_filepath = record.get('audio_filepath')
        if audio_filepath in annotated_filepaths:
            print(f"Skipping already annotated record {idx}/{total_records}: {audio_filepath}")
            continue
        batch_records.append(record)
        sentences_to_annotate.append(record.get('text', ""))
        filepaths.append(audio_filepath)

    if not sentences_to_annotate:
        continue  # All records in this batch are already annotated.

    # Annotate the batch.
    annotated_batch = annotate_sentences_with_intent(sentences_to_annotate)

    # If there is a mismatch, reattempt annotation individually.
    if len(annotated_batch) != len(sentences_to_annotate):
        print("Mismatch in annotated sentence count. Retrying individual annotations...")
        new_results = []
        for sentence in sentences_to_annotate:
            individual_result = annotate_sentences_with_intent([sentence])
            if isinstance(individual_result, list) and len(individual_result) == 1:
                new_results.append(individual_result[0])
            else:
                new_results.append(sentence)
        annotated_batch = new_results

    # Update each record with the annotated text.
    for i, annotated_sentence in enumerate(annotated_batch):
        audio_filepath = filepaths[i]
        # Update the record's "text" field with the annotated sentence.
        batch_records[i]['text'] = annotated_sentence
        # Append the updated record to the annotated records list.
        annotated_records.append({
            'audio_filepath': audio_filepath,
            'text': annotated_sentence,
            'duration': batch_records[i].get('duration', "")
        })
        annotated_filepaths.add(audio_filepath)
        # Append immediately to the JSONL output.
        with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
            f_out.write(json.dumps({
                'audio_filepath': audio_filepath,
                'text': annotated_sentence,
                'duration': batch_records[i].get('duration', "")
            }, ensure_ascii=False) + "\n")

    # Update the CSV output after each batch.
    csv_columns = ['audio_filepath', 'text', 'duration']
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(annotated_records)
    print(f"Processed and saved batch {batch_num + 1}.")

end_time = time.time()
total_time = end_time - start_time
print("\nAnnotation complete.")
print(f"Files saved as '{os.path.basename(output_jsonl_path)}' and '{os.path.basename(output_csv_path)}'.")
print(f"\nTotal execution time: {total_time:.2f} seconds.")
