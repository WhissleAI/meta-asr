const PROMPT_FILE_NAMES = [
  { file: "call-center.txt", displayName: "Call Center" },
  {
    file: "educational-interaction.txt",
    displayName: "Educational Interaction",
  },
  { file: "healthcare.txt", displayName: "Healthcare" },
  { file: "soccer-commentry.txt", displayName: "Soccer Commentary" },
  { file: "default.txt", displayName: "Default" },
  { file: "debate-analysis.txt", displayName: "Debate Analysis" },
  { file: "news_interview.txt", displayName: "News Interview" },
  { file: "fitness_expert.txt", displayName: "Fitness Expert" },
  { file: "universal_prompt.txt", displayName: "Universal Prompt" },
  { file: "football_new.txt", displayName: "Football (New)" },
];

export async function loadPrompts() {
  const prompts = await Promise.all(
    PROMPT_FILE_NAMES.map(async ({ file, displayName }) => {
      const res = await fetch(`/prompts/${file}`);
      const text = await res.text();
      return {
        name: file,
        displayName,
        content: text,
      };
    })
  );

  return prompts;
}
