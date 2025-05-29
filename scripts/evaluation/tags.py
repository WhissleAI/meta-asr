import json

def extract_labels(input_path: str, output_path: str):
    # Prepare containers
    labels = {
        "NER": set(),
        "EMOTION": set(),
        "INTENT": set(),
        "END": set(),
    }

    # Read each JSONL line and split out tokens
    with open(input_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            record = json.loads(line)
            # assume each record has a "text" field
            for tok in record.get("predicted_text", "").split():
                if tok == "END":
                    labels["END"].add(tok)
                elif tok.startswith("ENTITY_"):
                    labels["NER"].add(tok)
                elif tok.startswith("EMOTION_"):
                    labels["EMOTION"].add(tok)
                elif tok.startswith("INTENT_"):
                    labels["INTENT"].add(tok)
                # else: ignore other tokens (e.g., AGE_, GER_, words, punctuation)

    # Convert each set to a sorted list
    out = {cat: sorted(vals) for cat, vals in labels.items()}

    # Write to output JSON
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(out, f_out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # change these paths as needed
    INPUT_JSONL = "/external1/datasets/manifest_nemo/vils/valid_withpredictions.jsonl"
    OUTPUT_JSON = "/external1/datasets/manifest_nemo/vils/tags_predict.json"
    extract_labels(INPUT_JSONL, OUTPUT_JSON)
    print(f"Wrote label summary to {OUTPUT_JSON}")
