import Anthropic from "@anthropic-ai/sdk"
import type {
  RawMessageStreamEvent,
  Model
} from "@anthropic-ai/sdk/resources/messages.mjs"
import type { ChatPrompt, ModelApi } from "@mfbtech/llm-api-types"

type AnthropicModel = Extract<
  Model,
  | "claude-4-sonnet-20250514"
  | "claude-4-opus-20250514"
  | "claude-3-7-sonnet-latest"
  | "claude-3-5-haiku-latest"
>

export function buildAnthrophicLlm(
  model: AnthropicModel,
  anthrophicApiKey: string,
  dangerouslyAllowBrowser?: boolean
): ModelApi {
  const client = new Anthropic({
    apiKey: anthrophicApiKey,
    dangerouslyAllowBrowser
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
