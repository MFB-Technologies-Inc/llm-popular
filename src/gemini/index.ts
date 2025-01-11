import {
  EnhancedGenerateContentResponse,
  GoogleGenerativeAI
} from "@google/generative-ai"
import { nanoid } from "nanoid"
import { ChatPrompt, ModelApi } from "../types/index.js"

type GeminiModel = "gemini-1.5-pro" | "gemini-1.5-flash"

export function buildGeminiLlm(
  model: GeminiModel,
  geminiApiKey: string
): ModelApi {
  const generativeModel = new GoogleGenerativeAI(
    geminiApiKey
  ).getGenerativeModel({
    model: model
  })

  const toGeminiPrompt = (
    prompt: string | ChatPrompt
  ): { role: "user" | "model"; parts: { text: string }[] }[] =>
    typeof prompt === "string"
      ? [
          {
            role: "user",
            parts: [{ text: prompt }]
          }
        ]
      : prompt.map(p => ({
          role: p.role === "user" ? "user" : "model",
          parts: [{ text: p.prompt }]
        }))

  return {
    getText: async (prompt: string | ChatPrompt, instructions?: string) => {
      const internalUuid = nanoid()
      const result = (
        await generativeModel.generateContent({
          contents: toGeminiPrompt(prompt),
          systemInstruction: instructions
        })
      ).response

      const response = {
        uuid: internalUuid,
        text: result.text(),
        inputTokens: result.usageMetadata?.promptTokenCount,
        outputTokens: result.usageMetadata?.candidatesTokenCount
      }
      return response
    },
    getStream: async (prompt: string | ChatPrompt, instructions?: string) => {
      const internalUuid = nanoid()
      const geminiResult = await generativeModel.generateContentStream({
        contents: toGeminiPrompt(prompt),
        systemInstruction: instructions
      })

      // for some reason the geminiResult.response doesn't seem to accumulate
      // the text, so doing it manually
      //
      // this needs to be at this scope in case the user has already started to
      // exhaust the async iterable -- (i think)
      let accumulatedText = ""

      const result = {
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
          const response = {
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
