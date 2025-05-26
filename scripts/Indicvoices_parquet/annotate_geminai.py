import os
import json
import time
import csv
import google.generativeai as genai

start_time = time.time()

# Set your Google API key
GOOGLE_API_KEY = ''  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# File paths for input and output
input_jsonl_path = "/external3/databases/ai4bharat_indicvoices/dogri/train_intermediate.jsonl"
output_jsonl_path = "/external3/databases/ai4bharat_indicvoices/dogri/dogri_train_nemo.jsonl"

# Load records from the input JSONL file
with open(input_jsonl_path, 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f if line.strip()]

# Initialize annotated records
annotated_records = []
if os.path.exists(output_jsonl_path):
    with open(output_jsonl_path, 'r', encoding='utf-8') as f_json:
        annotated_records = [json.loads(line) for line in f_json if line.strip()]
    print(f"Loaded {len(annotated_records)} existing annotated records.")
else:
    print("No existing annotated records found. Starting fresh.")

# Create a set of already annotated record identifiers (using the 'audio_filepath' field)
annotated_filepaths = set(record['audio_filepath'] for record in annotated_records if 'audio_filepath' in record)

total_records = len(records)
print(f"Total records to process: {total_records}")
batch_size = 10  # Process 10 records per batch
total_batches = (total_records + batch_size - 1) // batch_size
print(f"Processing in batches of {batch_size} records.")


def annotate_sentences_with_intent(sentences):
    """
    Annotate each sentence using the Google Gemini 2.0 Flash Lite model.
    The annotation process marks entity phrases by inserting start tags ("ENTITY_<TYPE>")
    and a literal "END" after each entity phrase found in the sentence.
    
    The prompt instructs the model to only consider the sentence content.
    The function returns a list of annotated sentence strings.
    """
    
    prompt = f'''
Annotate each Dogri sentence from the input list by marking entity phrases with start and end tags using only the provided entity types. The annotation process MUST consider only the sentence content.

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
10. Tagging Enforcement:
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
"QUESTION_TYPE", "REQUEST_TYPE", "SUGGESTION_TYPE", "ALERT_TYPE", "REMINDER_TYPE", "STATUS", "ACTION", "COMMAND",
"FEATURE", "COMPONENT"
]

Example:

Input: 
[
    "अपने उत्पादें दी पैकेजिंग आस्तै इक अनोखी शैली बनाओ तां जे तुं'दे ब्रांड गी प्रतिस्पर्धा कोला बक्खरा रक्खेआ जाई सकै AGE_60+ GENDER_FEMALE EMOTION_HAP",
    "जेह्का तुस सवाल पुच्छै दे के पंज डेटां सनानियां जेह्कियां मी चेता न ते हां मी चेता हैन AGE_18_30 GENDER_MALE EMOTION_HAP",
    "मी चेता हैन पंज डेटां इक पैह्ली डेट एह् ऐ के एह् नौ तरीक त्रीआ म्हीना दो ज्हार चार ओह् मेरा जनमदिन औंदा उस दिन में पैदा होएया हा इक अगली तरीक ऐ AGE_18_30 GENDER_MALE EMOTION_HAP",
    "पंदरा तरीक अठमां म्हीना दो ज्हार सत्त उस दिन मेरा भ्राऽ पैदा होएया हा मेरे दोस्त दा जनमदिन दी तरीक बी पता ऐ मी ते बीह् तरीक चौथा म्हीना दो ज्हार चार AGE_18_30 GENDER_MALE EMOTION_SAD",
    "इक मेरे दोस्त दे इक होर दा जनमदिन बी चेता ऐ मी ऐ चौबी तरीक पैह्ला म्हीना दो ज्हार चार AGE_18_30 GENDER_MALE EMOTION_SAD"
]

Expected Output:
[
    "अपने ENTITY_PRODUCT उत्पादें END दी ENTITY_PACKAGING पैकेजिंग END आस्तै इक अनोखी ENTITY_STYLE शैली END बनाओ तां जे तुं'दे ENTITY_BRAND ब्रांड END गी ENTITY_COMPETITION प्रतिस्पर्धा END कोला बक्खरा रक्खेआ जाई सकै AGE_60+ GENDER_FEMALE EMOTION_HAP INTENT_SUGGESTION",
    "जेह्का तुस ENTITY_QUESTION सवाल END पुच्छै दे के पंज ENTITY_DATE डेटां END सनानियां जेह्कियां मी चेता न ते हां मी चेता हैन AGE_18_30 GENDER_MALE EMOTION_HAP INTENT_QUESTION",
    "मी चेता हैन पंज ENTITY_DATE डेटां END इक ENTITY_DATE पैह्ली डेट END एह् ऐ के एह् ENTITY_DATE नौ तरीक त्रीआ म्हीना दो ज्हार चार END ओह् मेरा ENTITY_EVENT जनमदिन END औंदा उस दिन में ENTITY_BIRTH पैदा होएया हा इक अगली तरीक ऐ AGE_18_30 GENDER_MALE EMOTION_HAP INTENT_MEMORY",
    "ENTITY_DATE पंदरा तरीक अठमां म्हीना दो ज्हार सत्त END उस दिन मेरा ENTITY_RELATIONSHIP भ्राऽ END ENTITY_BIRTH पैदा होएया हा मेरे ENTITY_RELATIONSHIP दोस्त END दा ENTITY_EVENT जनमदिन END दी ENTITY_DATE तरीक END बी पता ऐ मी ते ENTITY_DATE बीह् तरीक चौथा म्हीना दो ज्हार चार END AGE_18_30 GENDER_MALE EMOTION_SAD INTENT_MEMORY",
    "इक मेरे ENTITY_RELATIONSHIP दोस्त END दे इक होर दा ENTITY_EVENT जनमदिन END बी चेता ऐ मी ऐ ENTITY_DATE चौबी तरीक पैह्ला म्हीना दो ज्हार चार END AGE_18_30 GENDER_MALE EMOTION_SAD INTENT_MEMORY"
]    

Phrases to Annotate:
{json.dumps(sentences, ensure_ascii=False)}
'''
    try:
        # Create Gemini model instance
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate response
        response = model.generate_content(
            prompt
        )
        
        assistant_reply = response.text.strip()
        
        try:
            # Try to parse the response as JSON
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
            # If JSON parsing fails, try to extract JSON from the text
            print("JSON decoding failed. Attempting to extract JSON from the response...")
            
            # Look for array markers in the text
            start_idx = assistant_reply.find('[')
            end_idx = assistant_reply.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = assistant_reply[start_idx:end_idx]
                try:
                    annotated_results = json.loads(json_str)
                    if isinstance(annotated_results, list):
                        return annotated_results
                except json.JSONDecodeError:
                    pass
            
            # If extraction fails, retry with individual sentences
            print("JSON extraction failed. Retrying individual annotations...")
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

# Process records in batches
for batch_num in range(total_batches):
    start_idx = batch_num * batch_size
    end_idx = min((batch_num + 1) * batch_size, total_records)
    print(f"\nProcessing batch {batch_num + 1}/{total_batches} (records {start_idx + 1} to {end_idx})...")

    sentences_to_annotate = []
    filepaths = []
    batch_records = []

    # Gather records in this batch that have not been processed yet
    for idx, record in enumerate(records[start_idx:end_idx], start=start_idx + 1):
        audio_filepath = record.get('audio_filepath')
        if audio_filepath in annotated_filepaths:
            print(f"Skipping already annotated record {idx}/{total_records}: {audio_filepath}")
            continue
        batch_records.append(record)
        sentences_to_annotate.append(record.get('text', ""))
        filepaths.append(audio_filepath)

    if not sentences_to_annotate:
        continue  # All records in this batch are already annotated

    # Annotate the batch
    annotated_batch = annotate_sentences_with_intent(sentences_to_annotate)

    # If there is a mismatch, reattempt annotation individually
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

    # Update each record with the annotated text
    for i, annotated_sentence in enumerate(annotated_batch):
        audio_filepath = filepaths[i]
        # Update the record's "text" field with the annotated sentence
        batch_records[i]['text'] = annotated_sentence
        # Append the updated record to the annotated records list
        annotated_records.append({
            'audio_filepath': audio_filepath,
            'text': annotated_sentence,
            'duration': batch_records[i].get('duration', "")
        })
        annotated_filepaths.add(audio_filepath)
        # Append immediately to the JSONL output
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



#screen -r 3692230.pts-25.hydra2