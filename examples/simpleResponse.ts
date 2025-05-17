import { buildGeminiLlm, buildOpenAiLlm } from "../src/index.js"
import dotenv from "dotenv"

dotenv.config()

// const model = buildLlamaLlm("us.meta.llama3-3-70b-instruct-v1:0", {
//   awsAccessKey: varOrThrow("AWS_ACCESS_KEY"),
//   awsSecret: varOrThrow("AWS_SECRET"),
//   region: "us-east-1"
// })

// const model = buildAnthrophicLlm(
//   "claude-3-7-sonnet-20250219",
//   varOrThrow("ANTHROPIC_KEY")
// )

// const model = buildGeminiLlm(
//   "gemini-2.5-flash-preview-04-17",
//   varOrThrow("GEMINI_KEY")
// )
//

const model = buildOpenAiLlm("o4-mini", varOrThrow("OPENAI_KEY"))

const instructions = "Your response must be in the form of a rhyming couplet"

const prompt = "Confirm you are up and running"

const response = await model.getText(prompt, instructions)

console.log(response)

function varOrThrow(key: string): string {
  if (!process.env[key]) {
    throw new Error(`Env variable ${key} not defined`)
  }
  return process.env[key]
}
