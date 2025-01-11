import {
  EnhancedGenerateContentResponse,
  GoogleGenerativeAI
} from "@google/generative-ai"
import {
  Model,
  ModelApi,
  PromptFinalResponse,
  PromptResult
} from "@mfb/llm-types"
import { nanoid } from "nanoid"

type GeminiModel = Extract<Model, "gemini-1.5-pro" | "gemini-1.5-flash">

export function buildGeminiLlm(
  model: GeminiModel,
  geminiApiKey: string
): ModelApi {
  const generativeModel = new GoogleGenerativeAI(
    geminiApiKey
  ).getGenerativeModel({
    model: model
  })

  return {
    textPrompt: async (prompt: string, instructions?: string) => {
      const internalUuid = nanoid()
      const geminiResult = await generativeModel.generateContentStream({
        contents: [
          {
            role: "user",
            parts: [{ text: prompt }]
          }
        ],
        systemInstruction: instructions
      })

      // for some reason the geminiResult.response doesn't seem to accumulate
      // the text, so doing it manually
      //
      // this needs to be at this scope in case the user has already started to
      // exhaust the async iterable -- (i think)
      let accumulatedText = ""

      const result: PromptResult = {
        stream: mapIterable(
          geminiResult.stream,
          (s: EnhancedGenerateContentResponse): string => {
            const text = s.text()
            accumulatedText += text
            return text
          }
        ),
        getFinalResponse: async () => {
          for await (const chunk of geminiResult.stream) {
            accumulatedText += chunk.text()
          }
          const geminiResponse = await geminiResult.response
          const response: PromptFinalResponse = {
            uuid: internalUuid,
            text: accumulatedText,
            inputTokens: geminiResponse.usageMetadata?.promptTokenCount,
            outputTokens: geminiResponse.usageMetadata?.candidatesTokenCount
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
