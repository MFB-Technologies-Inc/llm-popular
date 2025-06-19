import {
  BedrockRuntimeClient,
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand
} from "@aws-sdk/client-bedrock-runtime"
import { ChatPrompt, ModelApi } from "@mfbtech/llm-api-types"
import { convertToLlamaPrompt } from "./convertToLlamaPrompt.js"
type Meta4Model =
  | "us.meta.llama4-maverick-17b-instruct-v1:0"
  | "us.meta.llama4-scout-17b-instruct-v1:0"
type Meta3Model =
  | "us.meta.llama3-3-70b-instruct-v1:0"
  | "us.meta.llama3-2-1b-instruct-v1:0"
  | "us.meta.llama3-2-3b-instruct-v1:0"

/**
 * The response returned by Llama 2 Chat, Llama 2, and Llama 3 Instruct models
 * for a text completion inference call.
 */
export type TextCompletionResponse = {
  /**
   * The generated text.
   */
  "generation": string

  /**
   * The number of tokens in the prompt.
   */
  "prompt_token_count": number

  /**
   * The number of tokens in the generated text.
   */
  "generation_token_count": number

  /**
   * The reason why the response stopped generating text.
   *
   * Possible values:
   * - `"stop"`: The model has finished generating text for the input prompt.
   * - `"length"`: The generated text exceeds the value of `max_gen_len`
   *   in the call to `InvokeModel`. The response is truncated to `max_gen_len` tokens.
   *   Consider increasing the value of `max_gen_len` and trying again.
   */
  "stop_reason": "stop" | "length" | null
  "amazon-bedrock-invocationMetrics"?: {
    inputTokenCount: number
    outputTokenCount: number
    invocationLatency: number
    firstByteLatency: number
  }
}

export function buildLlamaLlm(
  model: Meta4Model | Meta3Model,
  awsCredentials: {
    awsAccessKey: string
    awsSecret: string
    region: string
  }
): ModelApi {
  const client = new BedrockRuntimeClient({
    region: awsCredentials.region,
    credentials: {
      accessKeyId: awsCredentials.awsAccessKey,
      secretAccessKey: awsCredentials.awsSecret
    }
  })

  return {
    getText: async (prompt: string | ChatPrompt, instructions?: string) => {
      const invoke = new InvokeModelCommand({
        modelId: model,
        contentType: "application/json",
        body: JSON.stringify(
          typeof prompt === "string"
            ? convertToLlamaPrompt(
                [{ role: "user", text: prompt }],
                instructions
              )
            : convertToLlamaPrompt(
                prompt.map(p => ({ role: p.role, text: p.prompt })),
                instructions
              )
        )
      })
      const result = await client.send(invoke)
      if (!result.body) {
        throw new Error("Unexpectedly did not receive response stream")
      }
      // Decode and return the response(s)
      const decodedResponseBody = new TextDecoder().decode(result.body)
      const rawResponse = JSON.parse(decodedResponseBody) as {
        generation: string
        prompt_token_count: number
        generation_token_count: number
        stop_reason: "stop" | "length"
      }

      const response = {
        uuid: result.$metadata.requestId ?? "",
        text: rawResponse.generation,
        inputTokens: rawResponse.prompt_token_count,
        outputTokens: rawResponse.generation_token_count
      }
      return response
    },
    getStream: async (prompt: string | ChatPrompt, instructions?: string) => {
      const finalResponse = {
        uuid: "",
        text: "",
        inputTokens: undefined as number | undefined,
        outputTokens: undefined as number | undefined
      }

      const invoke = new InvokeModelWithResponseStreamCommand({
        modelId: model,
        contentType: "application/json",
        body: JSON.stringify(
          typeof prompt === "string"
            ? convertToLlamaPrompt(
                [{ role: "user", text: prompt }],
                instructions
              )
            : convertToLlamaPrompt(
                prompt.map(p => ({ role: p.role, text: p.prompt })),
                instructions
              )
        )
      })
      const stream = await client.send(invoke)
      if (!stream.body) {
        throw new Error("Unexpectedly did not receive response stream")
      }

      const result = {
        stream: mapIterable(stream.body, s => {
          const chunk: TextCompletionResponse = JSON.parse(
            new TextDecoder().decode(s.chunk?.bytes)
          )
          finalResponse.text += chunk.generation
          if (chunk.stop_reason === "stop") {
            finalResponse.inputTokens =
              chunk["amazon-bedrock-invocationMetrics"]?.inputTokenCount
            finalResponse.outputTokens =
              chunk["amazon-bedrock-invocationMetrics"]?.outputTokenCount
          }
          return chunk.generation
        }),
        getFinalResponse: async () => {
          const response = {
            ...finalResponse,
            uuid: stream.$metadata.requestId ?? ""
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
