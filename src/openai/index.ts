import { ModelApi, ChatPrompt } from "@mfbtech/llm-api-types"
import OpenAI from "openai"
import { ChatCompletionChunk } from "openai/resources/index.mjs"

type OpenAiModel = "gpt-4o" | "gpt-4o-mini"

export function buildOpenAiLlm(
  model: OpenAiModel,
  openAiKey: string
): ModelApi {
  const client = new OpenAI({
    apiKey: openAiKey
  })

  const defaultInstructions =
    "ensure that the last character of the response is a newline character."

  const toOpenAiPrompt = (
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
      const result = await client.beta.chat.completions.parse({
        model: model,
        messages: [
          {
            role: "system",
            content: instructions
              ? `${instructions} In addition, ${defaultInstructions}`
              : defaultInstructions
          },
          ...toOpenAiPrompt(prompt)
        ],
        n: 1
      })
      const response = {
        uuid: result.id,
        text: result.choices[0]?.message.content ?? "",
        inputTokens: result.usage?.prompt_tokens,
        outputTokens: result.usage?.completion_tokens
      }
      return response
    },
    getStream: async (prompt: string | ChatPrompt, instructions?: string) => {
      const stream = client.beta.chat.completions.stream({
        model: model,
        messages: [
          {
            role: "system",
            content: instructions
              ? `${instructions} In addition, ${defaultInstructions}`
              : defaultInstructions
          },
          ...toOpenAiPrompt(prompt)
        ],
        n: 1,
        stream: true,
        stream_options: {
          include_usage: true
        }
      })

      const result = {
        stream: mapIterable(stream, (s: ChatCompletionChunk): string => {
          //  console.log(s)
          return s.choices[0]?.delta.content ?? ""
        }),
        getFinalResponse: async () => {
          const streamResponse = await stream.finalChatCompletion()
          const response = {
            uuid: streamResponse.id,
            text: streamResponse.choices[0]?.message.content ?? "",
            inputTokens: streamResponse.usage?.prompt_tokens,
            outputTokens: streamResponse.usage?.completion_tokens
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
