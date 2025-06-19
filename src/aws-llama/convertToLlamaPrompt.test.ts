import { describe, it, expect } from "vitest"
import { convertToLlamaPrompt } from "./convertToLlamaPrompt.js"

describe("convertToLlamaPrompt", () => {
  describe("Llama 3 format (default)", () => {
    it("should convert a simple user message to Llama 3 prompt format", () => {
      const input = [{ role: "user" as const, text: "Hello, how are you?" }]
      const result = convertToLlamaPrompt(input)

      expect(result.prompt).toBe(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Hello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
      )
      expect(result.max_gen_len).toBe(2048)
      expect(result.temperature).toBe(0.5)
      expect(result.top_p).toBe(0.9)
    })

    it("should handle multiple messages in conversation", () => {
      const input = [
        { role: "user" as const, text: "What's 2+2?" },
        { role: "assistant" as const, text: "2+2 equals 4." },
        { role: "user" as const, text: "What about 3+3?" }
      ]
      const result = convertToLlamaPrompt(input)

      expect(result.prompt).toBe(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>What's 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>2+2 equals 4.<|eot_id|><|start_header_id|>user<|end_header_id|>What about 3+3?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
      )
    })

    it("should include system instructions when provided", () => {
      const input = [{ role: "user" as const, text: "Tell me a joke" }]
      const instructions = "You are a helpful assistant who tells funny jokes."
      const result = convertToLlamaPrompt(input, instructions)

      expect(result.prompt).toBe(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant who tells funny jokes.<|eot_id|><|start_header_id|>user<|end_header_id|>Tell me a joke<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
      )
    })

    it("should handle empty input array", () => {
      const input: { role: "user" | "assistant"; text: string }[] = []
      const result = convertToLlamaPrompt(input)

      expect(result.prompt).toBe(
        "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n"
      )
    })
  })

  describe("Llama 4 format", () => {
    it("should convert a simple user message to Llama 4 prompt format", () => {
      const input = [{ role: "user" as const, text: "Hello, how are you?" }]
      const result = convertToLlamaPrompt(input, undefined, "4")

      expect(result.prompt).toBe(
        "<|begin_of_text|><|header_start|>user<|header_end|>\n\nHello, how are you?<|eot|><|header_start|>assistant<|header_end|>\n"
      )
    })

    it("should handle multiple messages in conversation with Llama 4 format", () => {
      const input = [
        { role: "user" as const, text: "What's 2+2?" },
        { role: "assistant" as const, text: "2+2 equals 4." },
        { role: "user" as const, text: "What about 3+3?" }
      ]
      const result = convertToLlamaPrompt(input, undefined, "4")

      expect(result.prompt).toBe(
        "<|begin_of_text|><|header_start|>user<|header_end|>\n\nWhat's 2+2?<|eot|><|header_start|>assistant<|header_end|>\n\n2+2 equals 4.<|eot|><|header_start|>user<|header_end|>\n\nWhat about 3+3?<|eot|><|header_start|>assistant<|header_end|>\n"
      )
    })

    it("should include system instructions with Llama 4 format", () => {
      const input = [{ role: "user" as const, text: "Tell me a joke" }]
      const instructions = "You are a helpful assistant who tells funny jokes."
      const result = convertToLlamaPrompt(input, instructions, "4")

      expect(result.prompt).toBe(
        "<|begin_of_text|><|header_start|>system<|header_end|>\n\nYou are a helpful assistant who tells funny jokes.<|eot|><|header_start|>user<|header_end|>\n\nTell me a joke<|eot|><|header_start|>assistant<|header_end|>\n"
      )
    })

    it("should handle empty input array with Llama 4 format", () => {
      const input: { role: "user" | "assistant"; text: string }[] = []
      const result = convertToLlamaPrompt(input, undefined, "4")

      expect(result.prompt).toBe(
        "<|begin_of_text|><|header_start|>assistant<|header_end|>\n"
      )
    })

    it("should handle multi-line messages in Llama 4 format", () => {
      const input = [
        {
          role: "user" as const,
          text: "Line 1\nLine 2\nLine 3"
        }
      ]
      const result = convertToLlamaPrompt(input, undefined, "4")

      expect(result.prompt).toBe(
        "<|begin_of_text|><|header_start|>user<|header_end|>\n\nLine 1\nLine 2\nLine 3<|eot|><|header_start|>assistant<|header_end|>\n"
      )
    })
  })

  describe("Common behavior", () => {
    it("should handle special characters in messages", () => {
      const input = [
        {
          role: "user" as const,
          text: "Can you explain <|start_header_id|> and <|eot_id|>?"
        }
      ]
      const result = convertToLlamaPrompt(input)

      expect(result.prompt).toBe(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Can you explain <|start_header_id|> and <|eot_id|>?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
      )
    })

    it("should return consistent default parameters", () => {
      const input = [{ role: "user" as const, text: "Test" }]
      const result = convertToLlamaPrompt(input)

      expect(result).toHaveProperty("prompt")
      expect(result).toHaveProperty("max_gen_len", 2048)
      expect(result).toHaveProperty("temperature", 0.5)
      expect(result).toHaveProperty("top_p", 0.9)
    })
  })
})
