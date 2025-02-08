import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def remove_existing_tags(text):
    """
    Remove legacy (NER_) and current (ENTITY_) entity tags while preserving special tags
    like GENDER_*, EMOTION_*, AGE_*, and SPEAKER_CHANGE.
    """
    # Preserve these specific tags as is
    preserved_tags = {
        r'GENDER_[A-Z]+': lambda m: m.group(0),
        r'EMOTION_[A-Z]+': lambda m: m.group(0),
        r'AGE_[0-9_]+': lambda m: m.group(0),
        r'SPEAKER_CHANGE': lambda m: m.group(0),
    }
    preserved_text = text
    for pattern, replacement in preserved_tags.items():
        preserved_text = re.sub(pattern, replacement, preserved_text)
    
    # Remove both legacy and current entity tags (NER_ and ENTITY_)
    cleaned_text = re.sub(r'(?:NER|ENTITY)_\w+\s?', '', preserved_text)
    # Remove stray END tokens
    cleaned_text = re.sub(r'\s?END', '', cleaned_text)
    
    return cleaned_text

def fix_end_tags(text):
    """
    Ensure that every ENTITY_ annotation is followed by a space and 'END'.
    This function first adds a space before any "END" that is stuck to a word,
    then enforces that each annotation is terminated by " END".
    """
    # Add a space before any END that immediately follows a word character
    text = re.sub(r'(\w)END', r'\1 END', text)
    # For each occurrence of an ENTITY_ tag followed by some non-space characters,
    # ensure that it is terminated with " END" (if not already present)
    pattern = r'(ENTITY_[A-Z0-9_]+\s+[^\s.,?!]+)(?!\s+END)'
    text = re.sub(pattern, r'\1 END', text)
    return text

def annotate_sentences(sentences):
    prompt = f'''
You are given a list of sentences that may contain existing tags like GENDER_FEMALE, EMOTION_NEU, AGE_45_60, and SPEAKER_CHANGE.

Your task is to:
1. Remove any legacy entity tags such as NER_PERSON, NER_NORP, etc.
2. Preserve these existing tags exactly as they appear.
3. Add entity annotations for all other important entities in the text.
4. Insert entity tags in the format ENTITY_<TYPE> before each entity and append " END" (with a space) after it.
5. Focus on identifying and tagging ALL entities in the text, not just names and organizations.
6. Return the output as a JSON array of annotated sentences.

IMPORTANT:
- Do NOT modify or add tags for gender, emotion, age, or speaker changes.
- Remove any legacy tags (e.g., NER_PERSON) and annotate entities using **only** the ENTITY_<TYPE> format.
- Every annotation must be in the form "ENTITY_<TYPE> entity_text END" (notice the space before END).

Entities to annotate (exclude gender, emotion, age, and speaker tags):
[
    "PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE", 
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

Example input: "hi folks, i'm karen botic with ion sun valley. GENDER_FEMALE today we have a meeting at 2pm."
Example output: "hi folks, i'm ENTITY_PERSON_NAME karen botic END with ENTITY_ORGANIZATION ion sun valley END. GENDER_FEMALE today we have a ENTITY_MEETING meeting END at ENTITY_TIME 2pm END."

Sentences to Annotate:
{json.dumps(sentences, ensure_ascii=False)}
'''
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        assistant_reply = response.text.strip()

        # Remove markdown code fences if present
        if assistant_reply.startswith("```"):
            lines = assistant_reply.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            assistant_reply = "\n".join(lines).strip()

        try:
            annotated_sentences = json.loads(assistant_reply)
            if isinstance(annotated_sentences, list):
                # Enforce proper END tags for each annotated sentence
                return [fix_end_tags(sentence) for sentence in annotated_sentences]
            else:
                print("Assistant did not return a list. Fallback to raw reply.")
                return [fix_end_tags(assistant_reply)] * len(sentences)
        except json.JSONDecodeError:
            print("JSON decoding failed. Fallback to raw replies.")
            return [fix_end_tags(assistant_reply)] * len(sentences)
    except Exception as e:
        print(f"Error annotating batch: {e}")
        return sentences

def process_jsonl_file(input_path, output_path, batch_size=10):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read all lines from the input file
    with open(input_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    total_records = len(records)
    print(f"Total records to process: {total_records}")

    processed_records = []
    total_batches = (total_records + batch_size - 1) // batch_size
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_records)
        batch_records = records[start_idx:end_idx]
        print(f"\nProcessing batch {batch_num + 1}/{total_batches} (records {start_idx + 1} to {end_idx})...")

        # Prepare sentences for annotation by removing existing entity tags
        original_texts = []
        for record in batch_records:
            text = record['text']
            cleaned_text = remove_existing_tags(text)
            original_texts.append(cleaned_text)

        # Get new annotations
        annotated_texts = annotate_sentences(original_texts)

        # Combine annotations with preserved tags
        for record, annotated_text in zip(batch_records, annotated_texts):
            record['text'] = annotated_text
            processed_records.append(record)

        # Write batch to output file
        with open(output_path, 'a', encoding='utf-8') as f:
            for record in processed_records[-batch_size:]:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("\nProcessing complete.")
    print(f"Processed records saved to: {output_path}")

if __name__ == "__main__":
    input_jsonl_path = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/temo.jsonl"
    output_jsonl_path = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/genai_decoded_extra.jsonl"
    
    # Clear output file if it exists
    if os.path.exists(output_jsonl_path):
        os.remove(output_jsonl_path)
    
    process_jsonl_file(input_jsonl_path, output_jsonl_path)