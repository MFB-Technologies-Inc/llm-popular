import { buildOpenAiLlm } from "../src/index.js"
import dotenv from "dotenv"

dotenv.config()

const model = buildOpenAiLlm("gpt-4.1-mini", varOrThrow("OPENAI_KEY"))

// const model = buildGeminiLlm("gemini-2.0-flash", varOrThrow("GEMINI_KEY"))

const instructions =
  "Your response must be no longer than a single sentence that is vigorous and concise."

const prompt = "Confirm you are up and running"

console.log(prompt + "\n")

const response = await model.getText(prompt, instructions)

console.log(response.text)

const secondPrompt = "Now say it backwards"

console.log(secondPrompt + "\n")

const secondResponse = await model.getText(
  [
    { role: "user", prompt: prompt },
    { role: "assistant", prompt: response.text },
    { role: "user", prompt: secondPrompt }
  ],
  instructions
)

console.log(secondResponse.text)

function varOrThrow(key: string): string {
  if (!process.env[key]) {
    throw new Error(`Env variable ${key} not defined`)
  }
  return process.env[key]
}
