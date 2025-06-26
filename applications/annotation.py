# applications/annotation.py
import json
import asyncio
from typing import Tuple, Optional, List
from config import logger, ENTITY_TYPES, INTENT_TYPES # Relative import
from models import GEMINI_AVAILABLE # Changed from GEMINI_CONFIGURED and relative import
from session_store import get_user_api_key # Added for user-specific keys
import google.generativeai as genai
import re


# changes to the input string will be reflected in the `Sentences to Annotate Now` section below. 
def get_annotation_prompt(texts_to_annotate: List[str]) -> str:
    all_entity_types_str = ", ".join(ENTITY_TYPES)
    all_intent_types_str = ", ".join(INTENT_TYPES)
    return f'''You are an expert linguistic annotator for English text.
You will receive a list of English sentences, which may include multiple sentences in a single string. Each input is a raw lowercase transcription.

Your task is crucial and requires precision. For each input string, you must:
1.  **TOKENIZE:** Split the input into individual words and punctuation (tokens), preserving all elements (e.g., words, commas, periods).
2.  **ASSIGN BIO TAGS:** For each token, assign exactly one BIO tag:
    *   **ENTITY TAGS (Priority):** Identify entities using the provided `ENTITY_TYPES` list.
        *   `B-<ENTITY_TYPE>` for the *beginning* of an entity phrase (e.g., `B-PERSON_NAME`).
        *   `I-<ENTITY_TYPE>` for *inside* an entity phrase (e.g., `I-PERSON_NAME`).
    *   **UTTERANCE INTENT TAGS (Default/Fallback):** If a token is *not* part of any specific entity, tag it to reflect the overall intent.
        *   The first non-entity token of the input should be `B-<UTTERANCE_INTENT>`.
        *   Subsequent non-entity tokens should be `I-<UTTERANCE_INTENT>`.
        *   The `<UTTERANCE_INTENT>` should be from `INTENT_TYPES`.
    *   **CRITICAL:** Every token, including punctuation, must have a tag. Use `O` (Outside) if no entity or intent applies.
3.  **EXTRACT INTENT:** Determine and provide the single overall `intent` of the entire input string from `INTENT_TYPES`.
4.  **OUTPUT FORMAT (CRITICAL):** Return a JSON array of objects. Each object must contain:
    *   `text`: The original lowercase input string (for verification).
    *   `tokens`: A JSON array of all tokenized words and punctuation.
    *   `tags`: A JSON array of BIO tags, exactly matching the `tokens` array in length.
    *   `intent`: A single string representing the overall intent.

**ENTITY TYPES LIST (USE ONLY THESE FOR ENTITY TAGS):**
{json.dumps(ENTITY_TYPES, ensure_ascii=False, indent=2)}

**INTENT TYPES LIST (USE ONE FOR UTTERANCE INTENT AND FOR DEFAULT TAGS):**
{json.dumps(INTENT_TYPES, ensure_ascii=False, indent=2)}

**Example Input String 1 (with entities):**
"then if he becomes a champion, he's entitled to more money after that and champion end"

**CORRECT Example Output 1 (assuming intent is INFORM and "champion" is a PROJECT_NAME):**
```json
[
  {{
    "text": "then if he becomes a champion, he's entitled to more money after that and champion end",
    "tokens": ["then", "if", "he", "becomes", "a", "champion", ",", "he's", "entitled", "to", "more", "money", "after", "that", "and", "champion", "end"],
    "tags": ["B-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "B-PROJECT_NAME", "O", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "B-PROJECT_NAME", "O"],
    "intent": "INFORM"
  }}
]
Sentences to Annotate Now:
{json.dumps(texts_to_annotate, ensure_ascii=False, indent=2)}
'''

async def annotate_text_structured_with_gemini(text_to_annotate: str, custom_prompt: Optional[str], user_id: str) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[str], Optional[str]]:
    if not GEMINI_AVAILABLE:
        return None, None, None, "Gemini (google.generativeai) library is not available."

    gemini_api_key = get_user_api_key(user_id, "gemini")
    if not gemini_api_key:
        return None, None, None, "Gemini API key not found or session expired for user."

    try:
        # Configure Gemini with the user-specific key for this request
        # Note: genai.configure is a global setting. This approach has limitations in highly concurrent scenarios
        # if the SDK doesn't support per-client/per-request API keys directly.
        genai.configure(api_key=gemini_api_key)
    except Exception as e:
        logger.error(f"Failed to configure Gemini with user API key for user {user_id}: {e}")
        return None, None, None, "Failed to configure Gemini API with user key."

    if not text_to_annotate or text_to_annotate.isspace():
        return [], [], "NO_SPEECH_INPUT", None
    # prompt = get_annotation_prompt([text_to_annotate.lower()])
    if custom_prompt:
        prompt = f"{custom_prompt}\n Sentences to Annotate Now: {json.dumps([text_to_annotate.lower()], ensure_ascii=False, indent=2)}"
    else:
        prompt = get_annotation_prompt([text_to_annotate.lower()])
    # logger.info(f"Using prompt for Gemini annotation in annotation func: {prompt[:100]}...")  # Log first 100 chars to avoid clutter
    logger.info(f"Using prompt for Gemini annotation in annotation func: {prompt}")  # Log first 100 chars to avoid clutter
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json"),
            safety_settings=safety_settings,
            request_options={'timeout': 120}
        )
        if response.candidates:
            raw_json_output = response.text.strip()
            logger.debug(f"Gemini raw JSON output for BIO: {raw_json_output}")
            logger.info(f"Gemini raw JSON output for BIO: {raw_json_output}")
            try:
                parsed_data_list = json.loads(raw_json_output)
                if not isinstance(parsed_data_list, list) or not parsed_data_list:
                    logger.error(f"Gemini BIO annotation did not return a list or returned an empty list: {raw_json_output}")
                    return None, None, None, "Gemini BIO: Invalid or empty list format"
                annotation_object = parsed_data_list[0]
                tokens = annotation_object.get("tokens")
                tags = annotation_object.get("tags")
                intent = annotation_object.get("intent")
                if not (isinstance(tokens, list) and isinstance(tags, list) and isinstance(intent, str)):
                    logger.error(f"Gemini BIO: Invalid types for tokens, tags, or intent. Tokens: {type(tokens)}, Tags: {type(tags)}, Intent: {type(intent)}. Tokens: {tokens}, Tags: {tags}")
                    return None, None, None, "Gemini BIO: Type mismatch in parsed data"
                # if len(tokens) != len(tags):
                #     logger.error(f"Gemini BIO: Mismatch between token ({len(tokens)}) and tag ({len(tags)}) counts. Tokens: {tokens}, Tags: {tags}")
                #     return None, None, None, "Gemini BIO: Token/Tag count mismatch"
                if len(tokens) != len(tags):
                    logger.warning(f"Gemini BIO: Mismatch between token ({len(tokens)}) and tag ({len(tags)}) counts. Attempting to pad tags.")
                    # Pad tags if there are fewer tags than tokens
                    if len(tags) < len(tokens):
                        tags += ["O"] * (len(tokens) - len(tags))
                    # Truncate tags if there are more tags than tokens (less common, but handled just in case)
                    elif len(tags) > len(tokens):
                        tags = tags[:len(tokens)]
                    logger.info(f"After padding, Tokens: {tokens}, Tags: {tags}")

                return tokens, tags, intent.upper(), None
            except json.JSONDecodeError as json_e:
                logger.error(f"Gemini BIO annotation JSON decoding failed: {json_e}. Response: {raw_json_output}")
                return None, None, None, f"Gemini BIO: JSONDecodeError - {json_e}"
            except Exception as e:
                logger.error(f"Error parsing Gemini BIO annotation response: {e}. Response: {raw_json_output}")
                return None, None, None, f"Gemini BIO: Parsing error - {e}"
        else:
            error_message = f"No candidates from Gemini BIO annotation for text: {text_to_annotate[:100]}..."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                feedback = getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback))
                error_message += f" Feedback: {feedback}"
            logger.error(error_message)
            return None, None, None, error_message
    except Exception as e:
        logger.error(f"Gemini API/SDK error during BIO annotation: {type(e).__name__}: {e}", exc_info=True)
        return None, None, None, f"Gemini API/SDK error: {type(e).__name__}"













# each sentence in the input text, annotating each sentence separately and combining results.
# async def annotate_text_structured_with_gemini(text_to_annotate: str, custom_prompt: Optional[str], user_id: str) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[str], Optional[str]]:
#     if not GEMINI_AVAILABLE:
#         return None, None, None, "Gemini (google.generativeai) library is not available."

#     gemini_api_key = get_user_api_key(user_id, "gemini")
#     if not gemini_api_key:
#         return None, None, None, "Gemini API key not found or session expired for user."

#     try:
#         genai.configure(api_key=gemini_api_key)
#     except Exception as e:
#         logger.error(f"Failed to configure Gemini with user API key for user {user_id}: {e}")
#         return None, None, None, "Failed to configure Gemini API with user key."

#     if not text_to_annotate or text_to_annotate.isspace():
#         return [], [], "NO_SPEECH_INPUT", None

#     # Normalize newlines: replace multiple newlines with a single space
#     text_to_annotate = re.sub(r'\n+', ' ', text_to_annotate).strip()

#     # Split into sentences by . ! ?
#     sentences = [s.strip() for s in re.split(r'[.!?]', text_to_annotate) if s.strip()]
#     all_tokens, all_tags, final_intent = [], [], None

#     for i, sentence in enumerate(sentences):
#         # Prepare prompt for single sentence
#         prompt = f"{custom_prompt}\n Sentences to Annotate Now: {json.dumps([sentence.lower()], ensure_ascii=False, indent=2)}" or get_annotation_prompt([sentence.lower()])
#         logger.info(f"Annotating sentence: {sentence[:100]}...")

#         # Call Gemini for each sentence
#         try:
#             model = genai.GenerativeModel("models/gemini-1.5-flash")
#             safety_settings = [
#                 {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
#                 {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
#                 {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
#                 {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
#             ]
#             response = await asyncio.to_thread(
#                 model.generate_content,
#                 prompt,
#                 generation_config=genai.types.GenerationConfig(
#                     response_mime_type="application/json",
#                     max_output_tokens=2000
#                 ),
#                 safety_settings=safety_settings,
#                 request_options={'timeout': 300}
#             )
#             if response.candidates:
#                 raw_json_output = response.text.strip()
#                 logger.debug(f"Gemini raw JSON output for sentence: {raw_json_output}")
#                 try:
#                     parsed_data_list = json.loads(raw_json_output)
#                     if not isinstance(parsed_data_list, list) or not parsed_data_list:
#                         logger.error(f"Gemini returned invalid or empty list: {raw_json_output}")
#                         return None, None, None, "Gemini BIO: Invalid or empty list format"
#                     annotation_object = parsed_data_list[0]
#                     tokens = annotation_object.get("tokens")
#                     tags = annotation_object.get("tags")
#                     intent = annotation_object.get("intent")
#                     if not (isinstance(tokens, list) and isinstance(tags, list) and isinstance(intent, str)):
#                         logger.error(f"Invalid types. Tokens: {type(tokens)}, Tags: {type(tags)}, Intent: {type(intent)}")
#                         return None, None, None, "Gemini BIO: Type mismatch in parsed data"
#                     if len(tokens) != len(tags):
#                         logger.warning(f"Token/Tag mismatch for sentence: {sentence}. Tokens: {len(tokens)}, Tags: {len(tags)}. Padding with 'O' tags.")
#                         tags.extend(['O'] * (len(tokens) - len(tags)))  # Pad missing tags
#                     all_tokens.extend(tokens)
#                     all_tags.extend(tags)
#                     final_intent = intent.upper()
#                 except json.JSONDecodeError as json_e:
#                     logger.error(f"JSON decoding failed: {json_e}. Response: {raw_json_output}")
#                     return None, None, None, f"Gemini BIO: JSONDecodeError - {json_e}"
#                 except Exception as e:
#                     logger.error(f"Error parsing response: {e}. Response: {raw_json_output}")
#                     return None, None, None, f"Gemini BIO: Parsing error - {e}"
#             else:
#                 error_message = f"No candidates from Gemini for sentence: {sentence[:100]}..."
#                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
#                     feedback = getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback))
#                     error_message += f" Feedback: {feedback}"
#                 logger.error(error_message)
#                 return None, None, None, error_message
#         except Exception as e:
#             logger.error(f"Gemini API error for sentence: {sentence}. Error: {type(e).__name__}: {e}")
#             return None, None, None, f"Gemini API error: {type(e).__name__}"

#         # Add sentence-ending punctuation if not the last sentence
#         if i < len(sentences) - 1:
#             # Determine the punctuation used in the original split
#             match = re.search(r'([.!?])', text_to_annotate[text_to_annotate.find(sentence) + len(sentence):])
#             punct = match.group(1) if match else '.'
#             all_tokens.append(punct)
#             all_tags.append('O')

#     return all_tokens, all_tags, final_intent, None