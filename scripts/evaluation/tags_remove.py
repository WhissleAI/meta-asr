import json
import re

# Pattern to match standalone tokens:
#  - AGE_<digits>_<digits>        (e.g. AGE_45_60)
#  - GER_<LETTERS>                (e.g. GER_MALE)
#  - EMOTION_<LETTERS>            (e.g. EMOTION_SAD)
#  - INTENT_<LETTERS>             (e.g. INTENT_COMMAND)
#  - ENTITY_<LETTERS_OR_DIGITS>   (e.g. ENTITY_PERSON_NAME)
TAG_PATTERN = re.compile(
    r'\b(?:AGE_[0-9]+_[0-9]+|GER_[A-Z]+|EMOTION_[A-Z]+|INTENT_[A-Z]+|ENTITY_[A-Z0-9_]+)\b',
    flags=re.IGNORECASE
)

def strip_inline_tags(text: str) -> str:
    """
    Remove any standalone inline tokens matching our tag patterns,
    then collapse extra whitespace.
    """
    no_tags = TAG_PATTERN.sub("", text)
    # collapse multiple spaces into one, strip leading/trailing
    return re.sub(r'\s{2,}', ' ', no_tags).strip()

def clean_text_field_jsonl(input_path: str, output_path: str):
    """
    Reads a JSONL file from input_path, strips inline tags out of the "text" field,
    and writes cleaned records to output_path.
    """
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            if isinstance(rec.get("text"), str):
                rec["text"] = strip_inline_tags(rec["text"])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # specify your input and output files here
    INPUT_FILE = "/external1/datasets/manifest_nemo/vils/valid_withpredictions_base_model.jsonl"
    OUTPUT_FILE = "/external1/datasets/manifest_nemo/vils/valid_withpredictions_base_model_cleaned.jsonl"

    clean_text_field_jsonl(INPUT_FILE, OUTPUT_FILE)
    print(f"✅ Cleaned '{INPUT_FILE}' → '{OUTPUT_FILE}'")







