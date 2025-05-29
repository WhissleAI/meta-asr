export const prompts: string[] = [
  `As an expert linguistic annotator specializing in English Natural Language Understanding (NLU), your task is to meticulously annotate raw English sentences.

Each input will be a single English sentence provided as a raw lowercase transcription. Your annotation process for each sentence must strictly follow these steps and output specifications:

---

**I. CORE ANNOTATION TASKS:**

1.  **TOKENIZATION:**
    *   Split the input sentence into individual words (tokens). Punctuation should generally be treated as separate tokens unless it's an integral part of a multi-word entity (e.g., "Dr. Smith").

2.  **BIO TAGGING (Token-Level):**
    *   For each token, assign *exactly one* BIO (Beginning, Inside, Outside) tag based on its role.
    *   **Precedence Rule:** Entity tags take strict precedence over intent tags.

    *   **a. Entity Tags (\`B-<ENTITY_TYPE>\`, \`I-<ENTITY_TYPE>\`):**
        *   Identify and tag all entities present in the sentence using the provided \`ENTITY_TYPES\` list.
        *   \`B-<ENTITY_TYPE>\`: Use for the *first token* of an entity phrase (e.g., \`B-PERSON_NAME\` for "John").
        *   \`I-<ENTITY_TYPE>\`: Use for *subsequent tokens* within the same entity phrase (e.g., \`I-PERSON_NAME\` for "Smith" in "John Smith").

    *   **b. Utterance Intent Tags (\`B-<UTTERANCE_INTENT>\`, \`I-<UTTERANCE_INTENT>\`):**
        *   These tags are applied to tokens that are *not* part of any specific entity and are *not* \`O\`.
        *   The \`<UTTERANCE_INTENT>\` should be the determined \`overall_utterance_intent\` (see Task III below).
        *   **Rule for Intent Tagging:**
            *   The *first token* of the entire sentence (if it's not an entity or \`O\`) should be tagged \`B-<UTTERANCE_INTENT>\`.
            *   All *subsequent tokens* that are not entities and not \`O\` should be tagged \`I-<UTTERANCE_INTENT>\`. This means \`I-<UTTERANCE_INTENT>\` can follow an entity or \`O\` tag if the utterance intent continues.

    *   **c. Outside Tag (\`O\`):**
        *   Use \`O\` *only* for tokens that are purely functional, syntactic, or do not contribute specific semantic meaning to either an entity or the primary utterance intent. This includes:
            *   Most standalone punctuation marks (e.g., ",", "?", ".").
            *   Filler words or structural elements that clearly do not form part of an entity or convey the core semantic intent.
        *   **Constraint:** Ensure every token receives a tag.

3.  **OVERALL UTTERANCE INTENT EXTRACTION:**
    *   Determine the *single, most dominant* intent of the entire utterance.
    *   This intent must be chosen from the \`INTENT_TYPES\` list.
    *   This intent should ideally align with the intent tags applied to the majority of non-entity tokens.

---

**II. OUTPUT FORMAT (STRICT JSON ARRAY OF OBJECTS):**

Your final output must be a JSON array. Each object in the array represents an annotated sentence and *must* contain the following fields:

\`\`\`json
[
  {
    "text": "original lowercase input sentence",
    "tokens": ["token1", "token2", "token3", "..."],
    "tags": ["tag1", "tag2", "tag3", "..."],
    "intent": "OVERALL_UTTERANCE_INTENT"
  },
  {
    "text": "then if he becomes a champion, he's entitled to more money after that and champion end",
    "tokens": ["then", "if", "he", "becomes", "a", "champion", ",", "he's", "entitled", "to", "more", "money", "after", "that", "and", "champion", "end"],
    "tags": ["B-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "B-PROJECT_NAME", "O", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "I-INFORM", "B-PROJECT_NAME", "O"],
    "intent": "INFORM"
  }
]
\`\`\`
`,
  `As a highly skilled linguistic annotator specializing in English NLU (Natural Language Understanding), you are responsible for carefully labeling raw English sentences. Each sentence is presented as a raw, lowercase transcription. Your annotation must adhere to the following procedure and format:

I. MAIN ANNOTATION STEPS:

TOKENIZATION:

Divide the original sentence into distinct words (tokens). Generally consider punctuation as separate tokens, except when it is part of a compound term (e.g. "Mr. Rogers").
BIO TAGGING (Token-Level):

Assign exactly one BIO (Beginning, Inside, Outside) tag to each token, based on its role.
Important Note on Priority: Entity tags always take precedence over intent tags.
a. Entity Tags (B-<ENTITY_TYPE>, I-<ENTITY_TYPE>):

From the approved ENTITY_TYPES list, find any entities in the sentence and label them.
Mark the initial token of an entity phrase with B-<ENTITY_TYPE>; label continuing tokens of the same entity with I-<ENTITY_TYPE>.
b. Utterance Intent Tags (B-<UTTERANCE_INTENT>, I-<UTTERANCE_INTENT>):

These tags apply to tokens that are neither part of an entity nor belonging to the Outside category (O).
The <UTTERANCE_INTENT> corresponds to the overall_utterance_intent (see Step III).
Tag the first non-entity, non-O token in the sentence as B-<UTTERANCE_INTENT>, and all subsequent non-entity, non-O tokens as I-<UTTERANCE_INTENT>, even if there have been entity or O tokens in between.
c. Outside Tag (O):

Use the O tag strictly for tokens that are exclusively auxiliary, structural, or otherwise not part of any entity or direct sentence intent. This often includes punctuation marks and filler words that do not convey essential meaning.
Every token must receive one tag.
OVERALL UTTERANCE INTENT DETERMINATION:

Identify the single dominant intent of the entire sentence, choosing from the INTENT_TYPES list.
This overall intent should align with how you tag the majority of non-entity tokens.
II. REQUIRED OUTPUT FORMAT (JSON ARRAY OF OBJECTS):

Your output must be a strict JSON array. Each object within represents a single annotated sentence and should be structured as follows:
[
  {
    "text": "the original sentence in lowercase",
    "tokens": ["token1", "token2", "token3", ...],
    "tags": ["tag1", "tag2", "tag3", ...],
    "intent": "OVERALL_UTTERANCE_INTENT"
  }
][
  {
    "text": "please schedule a meeting with dr. smith at 3 pm tomorrow",
    "tokens": ["please", "schedule", "a", "meeting", "with", "dr.", "smith", "at", "3", "pm", "tomorrow"],
    "tags": ["B-REQUEST", "I-REQUEST", "I-REQUEST", "I-REQUEST", "I-REQUEST", "B-PERSON_NAME", "I-PERSON_NAME", "I-REQUEST", "B-TIME", "I-TIME", "I-TIME"],
    "intent": "REQUEST"
  },
  {
    "text": "john fly to new york next monday at 9 am for the tech summit",
    "tokens": ["john", "fly", "to", "new", "york", "next", "monday", "at", "9", "am", "for", "the", "tech", "summit"],
    "tags": [
      "B-PERSON_NAME",
      "B-TRAVEL",
      "I-TRAVEL",
      "B-LOCATION",
      "I-LOCATION",
      "B-DATE",
      "I-DATE",
      "I-TRAVEL",
      "B-TIME",
      "I-TIME",
      "I-TRAVEL",
      "I-TRAVEL",
      "B-EVENT",
      "I-EVENT"
    ],
    "intent": "TRAVEL"
  },
  {
    "text": "hey, can you confirm if jennifer lopez lives in los angeles, california?",
    "tokens": ["hey", ",", "can", "you", "confirm", "if", "jennifer", "lopez", "lives", "in", "los", "angeles", ",", "california", "?"],
    "tags": [
      "B-QUESTION",
      "O",
      "I-QUESTION",
      "I-QUESTION",
      "I-QUESTION",
      "I-QUESTION",
      "B-PERSON_NAME",
      "I-PERSON_NAME",
      "I-QUESTION",
      "I-QUESTION",
      "B-LOCATION",
      "I-LOCATION",
      "O",
      "B-LOCATION",
      "O"
    ],
    "intent": "QUESTION"
  },
  {
    "text": "i think michael jordan is hosting a charity event in chicago next weekend",
    "tokens": ["i", "think", "michael", "jordan", "is", "hosting", "a", "charity", "event", "in", "chicago", "next", "weekend"],
    "tags": [
      "B-OPINION",
      "I-OPINION",
      "B-PERSON_NAME",
      "I-PERSON_NAME",
      "I-OPINION",
      "I-OPINION",
      "I-OPINION",
      "I-OPINION",
      "I-OPINION",
      "I-OPINION",
      "B-LOCATION",
      "B-DATE",
      "I-DATE"
    ],
    "intent": "OPINION"
  },
  {
    "text": "could you send the financial report to mr. lee and ms. davis before friday",
    "tokens": ["could", "you", "send", "the", "financial", "report", "to", "mr.", "lee", "and", "ms.", "davis", "before", "friday"],
    "tags": [
      "B-REQUEST",
      "I-REQUEST",
      "I-REQUEST",
      "I-REQUEST",
      "I-REQUEST",
      "I-REQUEST",
      "I-REQUEST",
      "B-PERSON_NAME",
      "I-PERSON_NAME",
      "I-REQUEST",
      "B-PERSON_NAME",
      "I-PERSON_NAME",
      "I-REQUEST",
      "B-DATE"
    ],
    "intent": "REQUEST"
  }
]`,

  ``,
];
