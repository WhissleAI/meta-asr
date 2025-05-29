const PROMPT_FILE_NAMES = [
  "2.5_pro_prompt.txt",
  "o1_prompt.txt",
  "o3_mini_prompt.txt",
  "sonnet_4_prompt.txt",
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
