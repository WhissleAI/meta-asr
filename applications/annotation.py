# applications/annotation.py
import json
import asyncio
from typing import Tuple, Optional, List
from config import logger, ENTITY_TYPES, INTENT_TYPES
from models import GEMINI_CONFIGURED
import google.generativeai as genai

# def get_annotation_prompt(texts_to_annotate: List[str]) -> str:
#     all_entity_types_str = ", ".join(ENTITY_TYPES)
#     all_intent_types_str = ", ".join(INTENT_TYPES)
#     return f'''You are an expert linguistic annotator for English text.
# You will receive a list of English sentences. Each sentence is a raw lowercase transcription.

# Your task is crucial and requires precision. For each sentence, you must:
# 1.  **TOKENIZE:** Split the sentence into individual words (tokens).
# 2.  **ASSIGN BIO TAGS:** For each token, assign exactly one BIO tag according to the following rules:
#     *   **ENTITY TAGS (Priority):** Identify entities using the provided `ENTITY_TYPES` list.
#         *   `B-<ENTITY_TYPE>` for the *beginning* of an entity phrase (e.g., `B-PERSON_NAME`).
#         *   `I-<ENTITY_TYPE>` for *inside* an entity phrase (e.g., `I-PERSON_NAME`).
#     *   **UTTERANCE INTENT TAGS (Default/Fallback):** If a token is *not* part of any specific entity, it should be tagged to reflect the overall intent of the utterance.
#         *   The first token of the sentence (if not an entity) should be `B-<UTTERANCE_INTENT>`.
#         *   Subsequent non-entity tokens should be `I-<UTTERANCE_INTENT>`.
#         *   The `<UTTERANCE_INTENT>` should be chosen from the `INTENT_TYPES` list.
#     *   **IMPORTANT:** Ensure every token has a tag. If no specific entity or clear intent can be assigned, use `O` (Outside) for tokens.
# 3.  **EXTRACT INTENT:** In addition to tagging, determine and provide the single overall `intent` of the utterance as a separate field. This `intent` should be one of the `INTENT_TYPES`.
# 4.  **OUTPUT FORMAT (CRITICAL):** Return a JSON array of objects. Each object in the array must contain:
#     *   `text`: The original lowercase input sentence (for verification purposes).
#     *   `tokens`: A JSON array of the tokenized words.
#     *   `tags`: A JSON array of the BIO tags, corresponding one-to-one with the `tokens` array.
#     *   `intent`: A single string representing the overall utterance intent.

# **ENTITY TYPES LIST (USE ONLY THESE FOR ENTITY TAGS):**
# {json.dumps(ENTITY_TYPES, ensure_ascii=False, indent=2)}

# **INTENT TYPES LIST (USE ONE FOR UTTERANCE INTENT AND FOR DEFAULT TAGS):**
# {json.dumps(INTENT_TYPES, ensure_ascii=False, indent=2)}

# **Example Input String 1 (with entities):**
# "then if he becomes a champion, he's entitled to more money after that and champion end"

# **CORRECT Example Output 1 (assuming intent is INFORM and "champion" is a PROJECT_NAME):**
# ```json
# [
#   {{
#     "text": "then if he becomes a champion, he's entitled to more money after that and champion end",
#     "tokens": ["then", "if", "he", "becomes", "a", "champion", ",", "he's", "entitled", "to", "more", "money", "after", "that", "and", "champion", "end"],
#     "tags": ["B-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "B-PROJECT_NAME", "O", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "B-PROJECT_NAME", "O"],
#     "intent": "INFORM"
#   }}
# ]

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

async def annotate_text_structured_with_gemini(text_to_annotate: str) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[str], Optional[str]]:
    if not GEMINI_CONFIGURED:
        return None, None, None, "Gemini API is not configured."
    if not text_to_annotate or text_to_annotate.isspace():
        return [], [], "NO_SPEECH_INPUT", None
    prompt = get_annotation_prompt([text_to_annotate.lower()])
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
                    logger.error(f"Gemini BIO: Invalid types for tokens, tags, or intent. Tokens: {type(tokens)}, Tags: {type(tags)}, Intent: {type(intent)}")
                    return None, None, None, "Gemini BIO: Type mismatch in parsed data"
                if len(tokens) != len(tags):
                    logger.error(f"Gemini BIO: Mismatch between token ({len(tokens)}) and tag ({len(tags)}) counts.")
                    return None, None, None, "Gemini BIO: Token/Tag count mismatch"
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