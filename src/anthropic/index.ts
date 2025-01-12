import Anthropic from "@anthropic-ai/sdk"
import { RawMessageStreamEvent } from "@anthropic-ai/sdk/resources/messages.mjs"
import { ChatPrompt, ModelApi } from "@mfbtech/llm-api-types"

type AnthropicModel = "claude-3-5-sonnet-20241022" | "claude-3-5-haiku-20241022"

export function buildAnthrophicLlm(
  model: AnthropicModel,
  anthrophicApiKey: string
): ModelApi {
  const client = new Anthropic({
    apiKey: anthrophicApiKey
  })

  const toAnthropicPrompt = (
    prompt: string | ChatPrompt
  ): { role: "user" | "assistant"; content: string }[] =>
    typeof prompt === "string"
      ? [
          {
            role: "user",
            content: prompt
          }
        ]
      : prompt.map(p => ({
          role: p.role,
          content: p.prompt
        }))

  return {
    getText: async (prompt: string | ChatPrompt, instructions?: string) => {
      const result = await client.messages.create({
        max_tokens: 8192,
        model: model,
        system: instructions ? `${instructions}` : "",
        messages: toAnthropicPrompt(prompt)
      })
      const response = {
        uuid: result.id,
        text: result.content.map(c => (c.type === "text" ? c.text : "")).join(),
        inputTokens: result.usage.input_tokens,
        outputTokens: result.usage.output_tokens
      }
      return response
    },
    getStream: async (prompt: string | ChatPrompt, instructions?: string) => {
      const stream = client.messages.stream({
        max_tokens: 8192,
        model: model,
        system: instructions ? `${instructions}` : "",
        messages: toAnthropicPrompt(prompt)
      })
      const result = {
        stream: mapIterable(stream, (s: RawMessageStreamEvent) => {
          if (
            s.type === "content_block_delta" &&
            s.delta.type === "text_delta"
          ) {
            return s.delta.text
          } else {
            return ""
          }
        }),
        getFinalResponse: async () => {
          const claudeFinal = await stream.finalMessage()
          const response = {
            uuid: claudeFinal.id,
            text: claudeFinal.content
              .map(c => (c.type === "text" ? c.text : ""))
              .join(),
            inputTokens: claudeFinal.usage.input_tokens,
            outputTokens: claudeFinal.usage.output_tokens
          }
          return response
        }
      }
      return result
    }
  }
}

async function* mapIterable<S, T>(
  asyncIterable: AsyncIterable<S>,
  f: (source: S) => T
): AsyncIterable<T> {
  for await (const item of asyncIterable) {
    yield f(item)
  }
}
