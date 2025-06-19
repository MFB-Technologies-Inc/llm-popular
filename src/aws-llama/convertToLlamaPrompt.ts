export type Llama33 = {
  prompt: string
  /** @property {number} [max_gen_len=512] - The maximum number of tokens for the generated response.
   * The response is truncated once it exceeds this value.
   * Defaults to `512`. Minimum: `1`, Maximum: `2048`.
   */
  max_gen_len?: number
  /**
   * Controls the randomness of the response.
   * A lower value decreases randomness.
   * Defaults to `0.5`. Minimum: `0`, Maximum: `1`.
   */
  temperature?: number
  /**
   * Filters out less probable options.
   * Use `0` or `1.0` to disable.
   * Defaults to `0.9`. Minimum: `0`, Maximum: `1`.
   */
  top_p?: number
}

export function convertToLlamaPrompt(
  input: { role: "user" | "assistant"; text: string }[],
  instructions?: string,
  version: "3" | "4" = "3"
): Llama33 {
  let llamaPrompt = `<|begin_of_text|>`

  if (version === "4") {
    // Llama 4 format
    if (instructions) {
      llamaPrompt += `<|header_start|>system<|header_end|>\n\n${instructions}<|eot|>`
    }

    for (const message of input) {
      llamaPrompt += `<|header_start|>${message.role}<|header_end|>\n\n${message.text}<|eot|>`
    }

    // End the prompt for Llama 4
    llamaPrompt += "<|header_start|>assistant<|header_end|>\n"
  } else {
    // Llama 3 format (default)
    if (instructions) {
      llamaPrompt += `<|start_header_id|>system<|end_header_id|>${instructions}<|eot_id|>`
    }

    for (const message of input) {
      llamaPrompt += `<|start_header_id|>${message.role}<|end_header_id|>${message.text}<|eot_id|>`
    }

    // End the prompt for Llama 3
    llamaPrompt += "<|start_header_id|>assistant<|end_header_id|>\n"
  }

  // Return the final prompt string
  return {
    prompt: llamaPrompt,
    max_gen_len: 2048,
    temperature: 0.5,
    top_p: 0.9
  }
}
