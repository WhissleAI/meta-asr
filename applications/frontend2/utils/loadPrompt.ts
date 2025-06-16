const PROMPT_FILE_NAMES = [
  "call-center.txt",
  "educational-interaction.txt",
  "healthcare.txt",
  "soccer-commentry.txt",
  "default.txt",
  "debate-analysis.txt",
  "news_interview.txt",
  "fitness_expert.txt",
  "universal_prompt.txt",

]; // Add more as needed

export async function loadPrompts() {
  const prompts = await Promise.all(
    PROMPT_FILE_NAMES.map(async (fileName) => {
      const res = await fetch(`/prompts/${fileName}`);
      const text = await res.text();
      return {
        name: fileName,
        content: text,
      };
    })
  );

  return prompts;
}
