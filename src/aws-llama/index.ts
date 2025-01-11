import {
  BedrockRuntimeClient,
  InvokeModelWithResponseStreamCommand
} from "@aws-sdk/client-bedrock-runtime"
import { ModelApi, PromptFinalResponse, PromptResult } from "@mfb/llm-types"
type Meta33Model = "us.meta.llama3-3-70b-instruct-v1:0"
type Meta32Model =
  | "us.meta.llama3-2-1b-instruct-v1:0"
  | "us.meta.llama3-2-3b-instruct-v1:0"

type Llama33 = {
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

function convertToLlamaPrompt(
  input: { role: "user" | "assistant"; text: string }[],
  instructions?: string
): Llama33 {
  let llamaPrompt = `<|begin_of_text|>`

  if (instructions) {
    llamaPrompt += `<|start_header_id|>system<|end_header_id|>${instructions}<|eot_id|>`
  }

  for (const message of input) {
    llamaPrompt += `<|start_header_id|>${message.role}<|end_header_id|>${message.text}<|eot_id|>`
  }

  // End the prompt -- ending the prompt this way ensure the next thing it generates in it's
  // completion is the start of the answer. Otherwise it will generate headers or newlines.
  llamaPrompt += "<|start_header_id|>assistant<|end_header_id|>\n"

  // Return the final prompt string
  return {
    prompt: llamaPrompt,
    max_gen_len: 2048,
    temperature: 0.5,
    top_p: 0.9
  }
}

export function buildLlamaLlm(
  model: Meta33Model | Meta32Model,
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
    textPrompt: async (prompt: string, instructions?: string) => {
      const finalResponse: PromptFinalResponse = {
        uuid: "",
        text: ""
      }

      const invoke = new InvokeModelWithResponseStreamCommand({
        modelId: model,
        contentType: "application/json",
        body: JSON.stringify(
          convertToLlamaPrompt([{ role: "user", text: prompt }], instructions)
        )
      })
      const stream = await client.send(invoke)
      if (!stream.body) {
        throw new Error("Unexpectedly did not receive response stream")
      }

      const result: PromptResult = {
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
          const response: PromptFinalResponse = {
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
