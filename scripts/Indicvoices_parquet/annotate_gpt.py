import os
import json
import openai
import time
import csv

start_time = time.time()

# Set your OpenAI API key.
openai.api_key = ''

# File paths for input and output.
input_jsonl_path = "/external3/databases/ai4bharat_indicvoices/assamese/valid_intermediate.jsonl"
output_jsonl_path = "/external3/databases/ai4bharat_indicvoices/assamese/assamese_valid_nemo.jsonl"

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
Annotate each Assamese sentence from the input list by marking entity phrases with start and end tags using only the provided entity types. The annotation process MUST consider only the sentence content.

IMPORTANT: All instructions provided below must be followed exactly and strictly. Do not add any extra text, explanations, or formatting.

Instructions:
    1. Process each sentence independently.
    2. Identify all entity phrases that match one of the provided entity types. When determining which entities to annotate, consider the sentence content.
    3. For each identified entity phrase, insert a start tag and an end tag always as below:
         - The start tag must be in the format: ENTITY_<ENTITY_TYPE> (replace spaces with underscores).
         - The end tag is the literal string: END.
    4. Use only the entity types provided in the list. Do not introduce any additional entity types.
    5. Classify the intent of each sentence and add an INTENT_<INTENT_TYPE> tag at the end of the sentence.
    6. Ensure the output is a JSON array containing only the annotated sentences in the same order as the input sentences, without any markdown or formatting.
    7. Do not add any additional text, explanations, or formatting.
    8. For every occurrence of a phrase matching an entity type, ensure it is enclosed with a ENTITY_<TYPE> tag followed by END. Never skip END, even for repeated or adjacent entities.
    9. All entity phrases — regardless of length — MUST be wrapped with a start (ENTITY_<TYPE>) and END.
    10.Tagging Enforcement:
        - Every time you use a tag like ENTITY_<TYPE>, it must be followed by the phrase and the word END.
        - Missing END is not allowed under any condition.
        - Even short phrases (one word) must have END.
        - If a tag appears multiple times, each must be followed by its own END.
        - Do not skip any required END tags.

Entity Types:
[   "PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE",
    "DATE", "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME",
    "EVENT", "MEETING", "TASK", "PROJECT_NAME", "ACTION_ITEM", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT",
    "QUESTION", "RESPONSE", "NOTIFICATION_TYPE", "AGENDA", "REMINDER", "NOTE", "RECORD", "ANNOUNCEMENT", "UPDATE",
    "SCHEDULE", "BOOKING_REFERENCE", "APPOINTMENT_NUMBER", "ORDER_NUMBER", "INVOICE_NUMBER", "PAYMENT_METHOD",
    "PAYMENT_AMOUNT", "BANK_NAME", "ACCOUNT_NUMBER", "CREDIT_CARD_NUMBER", "TAX_ID", "SOCIAL_SECURITY_NUMBER",
    "DRIVER'S_LICENSE", "PASSPORT_NUMBER", "INSURANCE_PROVIDER", "POLICY_NUMBER", "INSURANCE_PLAN", "CLAIM_NUMBER",
    "POLICY_HOLDER", "BENEFICIARY", "RELATIONSHIP", "EMERGENCY_CONTACT", "PROJECT_PHASE", "VERSION", "DEVELOPMENT_STAGE",
    "DEVICE_NAME", "OPERATING_SYSTEM", "SOFTWARE_VERSION", "BRAND", "MODEL_NUMBER", "LICENSE_PLATE", "VEHICLE_MAKE",
    "VEHICLE_MODEL", "VEHICLE_TYPE", "FLIGHT_NUMBER", "HOTEL_NAME", "ROOM_NUMBER", "TRANSACTION_ID", "TICKET_NUMBER",
    "SEAT_NUMBER", "GATE", "TERMINAL", "TRANSACTION_TYPE", "PAYMENT_STATUS", "PAYMENT_REFERENCE", "INVOICE_STATUS",
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST_NAME", "TEST_RESULT", "MEDICAL_RECORD",
    "HEALTH_STATUS", "HEALTH_METRIC", "VITAL_SIGN", "DOCTOR_NAME", "HOSPITAL_NAME", "DEPARTMENT", "WARD", "CLINIC_NAME",
    "WEBSITE", "URL", "IP_ADDRESS", "MAC_ADDRESS", "USERNAME", "PASSWORD", "LANGUAGE", "CODE_SNIPPET", "DATABASE_NAME",
    "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN",
    "PRODUCT", "SERVICE", "CATEGORY", "BRAND", "ORDER_STATUS", "DELIVERY_METHOD", "RETURN_STATUS", "WARRANTY_PERIOD",
    "CANCELLATION_REASON", "REFUND_AMOUNT", "EXCHANGE_ITEM", "GIFT_OPTION", "GIFT_MESSAGE",
    "FOOD_ITEM", "DRINK_ITEM", "CUISINE", "MENU_ITEM", "ORDER_NUMBER", "DELIVERY_ESTIMATE", "RECIPE", "INGREDIENT",
    "DISH_NAME", "PORTION_SIZE", "COOKING_TIME", "PREPARATION_METHOD",
    "NATIONALITY", "RELIGION", "MARITAL_STATUS", "OCCUPATION", "EDUCATION_LEVEL", "DEGREE",
    "SKILL", "EXPERIENCE", "YEARS_OF_EXPERIENCE", "CERTIFICATION",
    "MEASUREMENT", "DISTANCE", "WEIGHT", "HEIGHT", "VOLUME", "TEMPERATURE", "SPEED", "CAPACITY", "DIMENSION", "AREA",
    "SHAPE", "COLOR", "MATERIAL", "TEXTURE", "PATTERN", "STYLE",
    "WEATHER_CONDITION", "TEMPERATURE_SETTING", "HUMIDITY_LEVEL", "WIND_SPEED", "RAIN_INTENSITY", "AIR_QUALITY",
    "POLLUTION_LEVEL", "UV_INDEX",
    "QUESTION_TYPE", "REQUEST_TYPE", "SUGGESTION_TYPE", "ALERT_TYPE", "REMINDER_TYPE", "STATUS", "ACTION", "COMMAND"
]

Example:


Input: 
[
    "আজিৰ ভাৰতবৰ্ষই সন্মুখীন হোৱা প্ৰত্যাহ্বানসমূহ তথা সমস্যাসমূহৰ কথা যদি আমি ক'বলৈ যাওঁ তাৰ ভিতৰত এটা ডাঙৰ বা অন্যতম সমস্যা হৈছে সীমাৰ সমস্যা AGE_30_45 GENDER_FEMALE EMOTION_NEU",
    "চীন দেশত আজিলৈকে ডকলাম সীমান্তত এটা সীমাকলৈ মাজে মাজে সীমান্তৰক্ষীৰ মাজত এটা সংঘৰ্ষ হৈ থাকে আৰু তাৰ লগতে পাকিস্তান আৰু ভাৰতবৰ্ষৰ মাজতো সীমাৰ লগত এটা সমস্যা হৈ থাকে AGE_30_45 GENDER_FEMALE EMOTION_NEU",
    "পাকিস্তান আৰু পাকিস্তান আৰু ভাৰতবৰ্ষত বিশেষকৈ জম্মু আৰু কাশ্মীৰকলৈ এটা সীমাৰ সমস্যা প্ৰায়ে চলি থাকে AGE_30_45 GENDER_FEMALE EMOTION_NEU",
    "সেই সমস্যাসমূহ সমাধানৰ বাবে চৰকাৰে মাজে মাজে চীন মাজে মাজে চীনৰ চৰকাৰে বা পাকিস্তানৰ চৰকাৰৰ লগতো কথা বতৰা মিলিত কথা বতৰা বা আলোচনাত মিলিত হৈ থাকে AGE_30_45 GENDER_FEMALE EMOTION_NEU",
    "কিন্তু পাকিস্তান আৰু চীনৰ ফালৰ পৰা কিন্তু সময়ে সময়ে পাকিস্তান আৰু চীনৰ ফালৰ পৰা আগ্রাসনৰ বাবে সৈন্যবাহিনী আহি থাকে গতিকে সমস্যাকেইটা সমাধানৰ চেষ্টা চলি আছে যদিও এতিয়ালৈকে সম্পূৰ্ণৰূপে সমাধান হোৱা নাই AGE_30_45 GENDER_FEMALE EMOTION_NEU"  
]

Expected Output:
[
    "আজিৰ ENTITY_COUNTRY ভাৰতবৰ্ষই END সন্মুখীন হোৱা প্ৰত্যাহ্বানসমূহ তথা সমস্যাসমূহৰ কথা যদি আমি ক'বলৈ যাওঁ তাৰ ভিতৰত এটা ডাঙৰ বা অন্যতম ENTITY_COMPLAINT সমস্যা END হৈছে সীমাৰ সমস্যা AGE_30_45 GENDER_FEMALE EMOTION_NEU INTENT_STATEMENT",
    "ENTITY_COUNTRY চীন END দেশত আজিলৈকে ENTITY_LOCATION ডকলাম সীমান্তত END এটা সীমাকলৈ মাজে মাজে সীমান্তৰক্ষীৰ মাজত এটা ENTITY_EVENT সংঘৰ্ষ END হৈ থাকে আৰু তাৰ লগতে ENTITY_COUNTRY পাকিস্তান END আৰু ENTITY_COUNTRY ভাৰতবৰ্ষৰ END মাজতো সীমাৰ লগত এটা ENTITY_COMPLAINT সমস্যা END হৈ থাকে AGE_30_45 GENDER_FEMALE EMOTION_NEU INTENT_REPORT",
    "ENTITY_COUNTRY পাকিস্তান END আৰু ENTITY_COUNTRY পাকিস্তান END আৰু ENTITY_COUNTRY ভাৰতবৰ্ষত END বিশেষকৈ ENTITY_STATE জম্মু আৰু কাশ্মীৰকলৈ END এটা সীমাৰ ENTITY_COMPLAINT সমস্যা END প্ৰায়ে চলি থাকে AGE_30_45 GENDER_FEMALE EMOTION_NEU INTENT_REPORT",
    "সেই সমস্যাসমূহ ENTITY_ACTION সমাধানৰ বাবে END ENTITY_ORGANIZATION চৰকাৰে END মাজে মাজে ENTITY_COUNTRY চীন END মাজে মাজে ENTITY_ORGANIZATION চীনৰ চৰকাৰে END বা ENTITY_ORGANIZATION পাকিস্তানৰ চৰকাৰৰ END লগতো কথা বতৰা মিলিত কথা বতৰা বা ENTITY_MEETING আলোচনাত END মিলিত হৈ থাকে AGE_30_45 GENDER_FEMALE EMOTION_NEU INTENT_DISCUSSION",
    "কিন্তু ENTITY_COUNTRY পাকিস্তান END আৰু ENTITY_COUNTRY চীনৰ END ফালৰ পৰা কিন্তু সময়ে সময়ে ENTITY_COUNTRY পাকিস্তান END আৰু ENTITY_COUNTRY চীনৰ END ফালৰ পৰা আগ্রাসনৰ বাবে ENTITY_ORGANIZATION সৈন্যবাহিনী END আহি থাকে গতিকে সমস্যাকেইটা ENTITY_ACTION সমাধানৰ চেষ্টা END চলি আছে যদিও এতিয়ালৈকে সম্পূৰ্ণৰূপে সমাধান হোৱা নাই AGE_30_45 GENDER_FEMALE EMOTION_NEU INTENT_UPDATE"   
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
            ]
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

    
    print(f"Processed and saved batch {batch_num + 1}.")

end_time = time.time()
total_time = end_time - start_time
print("\nAnnotation complete.")
print(f"Files saved as {os.path.basename(output_jsonl_path)}")
print(f"\nTotal execution time: {total_time:.2f} seconds.")


#screen -r 968088.pts-0.hydra2