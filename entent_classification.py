import json
import re
from typing import List, Dict, Tuple
import vertexai
import os
from vertexai.generative_models import GenerativeModel
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()
def process_jsonl_annotations(input_file: str, output_file: str, api_key: str):
   
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    ENTITIES = [
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
        "AGE", "GENDER", "NATIONALITY", "RELIGION", "MARITAL_STATUS", "OCCUPATION", "EDUCATION_LEVEL", "DEGREE",
        "SKILL", "EXPERIENCE", "YEARS_OF_EXPERIENCE", "CERTIFICATION",
        "MEASUREMENT", "DISTANCE", "WEIGHT", "HEIGHT", "VOLUME", "TEMPERATURE", "SPEED", "CAPACITY", "DIMENSION", "AREA",
        "SHAPE", "COLOR", "MATERIAL", "TEXTURE", "PATTERN", "STYLE",
        "WEATHER_CONDITION", "TEMPERATURE_SETTING", "HUMIDITY_LEVEL", "WIND_SPEED", "RAIN_INTENSITY", "AIR_QUALITY",
        "POLLUTION_LEVEL", "UV_INDEX",
        "QUESTION_TYPE", "REQUEST_TYPE", "SUGGESTION_TYPE", "ALERT_TYPE", "REMINDER_TYPE", "STATUS", "ACTION", "COMMAND"
    ]

    def preserve_special_tags(text: str) -> Tuple[str, Dict[str, List[str]]]:
        """Extract and preserve special tags (AGE, GENDER, EMOTION) for later restoration"""
        preserved_tags = {
            'age': re.findall(r'AGE_[^\s]+', text),
            'gender': re.findall(r'GENDER_[^\s]+', text),
            'emotion': re.findall(r'EMOTION_[^\s]+', text),
            'speaker': re.findall(r'SPEAKER_[^\s]+', text)
        }
        
        # Remove the special tags temporarily
        text = re.sub(r'(AGE_[^\s]+|GENDER_[^\s]+|EMOTION_[^\s]+|SPEAKER_[^\s]+)', ' PLACEHOLDER ', text)
        return text, preserved_tags

    def restore_special_tags(text: str, preserved_tags: Dict[str, List[str]]) -> str:
        """Restore the preserved special tags back into the text"""
        for tag_type in preserved_tags:
            for tag in preserved_tags[tag_type]:
                text = text.replace('PLACEHOLDER', tag, 1)
        return text

    def annotate_text(text: str) -> str:
        """
        Annotate the text using Vertex AI while preserving special tags.
        """
        # First preserve special tags
        text_without_tags, preserved_tags = preserve_special_tags(text)
        
        # Remove existing NER tags
        text_without_ner = re.sub(r'NER_\w+\s|END\s?', '', text_without_tags)
        
        try:
            vertexai.init(project="stream2action", location="us-central1")
            model = GenerativeModel("gemini-1.5-flash-002")
            
            prompt = f'''
            Annotate the following text with entity tags from the provided list. Preserve any existing AGE, GENDER, EMOTION, or SPEAKER tags.

            Rules:
            1. Use format: ENTITY_<type> for entities and END to close each entity
            2. Only use entity types from the provided list
            3. Annotate all relevant entities in the text
            4. Preserve any existing AGE_, GENDER_, EMOTION_, or SPEAKER_ tags
            5. Do not add any additional text or explanations
            6. Ensure proper nesting and closing of tags

            Available Entity Types:
            {json.dumps(ENTITIES, indent=2)}

            Text to annotate:
            {text_without_ner}
            '''
            
            response = model.generate_content(prompt)
            annotated_text = response.text.strip()
            
            # Restore the preserved tags
            final_text = restore_special_tags(annotated_text, preserved_tags)
            
            # Validate the annotations
            for entity in ENTITIES:
                # Check if opening tags match closing tags
                opens = len(re.findall(f'ENTITY_{entity}', final_text))
                closes = len(re.findall('END', final_text))
                if opens != closes:
                    print(f"Warning: Mismatched tags for {entity} in text")
            
            return final_text
            
        except Exception as e:
            print(f"Error during annotation: {str(e)}")
            return text

    # Process the JSONL file
    processed_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_number, line in enumerate(f_in, 1):
            try:
                data = json.loads(line)
                if 'text' in data:
                    data['text'] = annotate_text(data['text'])
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                    # Print progress every 100 lines
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} lines...")
                        
            except json.JSONDecodeError as e:
                print(f"Error processing line {line_number}: {str(e)}")
                error_count += 1
                continue
            
            except Exception as e:
                print(f"Unexpected error at line {line_number}: {str(e)}")
                error_count += 1
                continue

    print(f"\nProcessing complete:")
    print(f"Total lines processed: {processed_count}")
    print(f"Errors encountered: {error_count}")

# Usage example
if __name__ == "__main__":
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable")
        
    process_jsonl_annotations(
        input_file="/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/decoded_extra.jsonl",
        output_file="/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/temp.jsonl",
        api_key=api_key
    )