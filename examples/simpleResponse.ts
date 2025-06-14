import {
  buildGeminiLlm,
  buildOpenAiLlm,
  buildAnthrophicLlm,
  buildLlamaLlm
} from "../src/index.js"
import dotenv from "dotenv"
import * as readline from "readline/promises"

dotenv.config()

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})

console.log("Select a provider:")
console.log("1. AWS Llama")
console.log("2. Anthropic")
console.log("3. Gemini")
console.log("4. OpenAI")

const choice = await rl.question("\nEnter your choice (1-4): ")
rl.close()

let model

switch (choice) {
  case "1":
    model = buildLlamaLlm("us.meta.llama4-scout-17b-instruct-v1:0", {
      awsAccessKey: varOrThrow("AWS_ACCESS_KEY"),
      awsSecret: varOrThrow("AWS_SECRET"),
      region: "us-east-1"
    })
    break
  case "2":
    model = buildAnthrophicLlm(
      "claude-4-opus-20250514",
      varOrThrow("ANTHROPIC_KEY")
    )
    break
  case "3":
    model = buildGeminiLlm(
      "gemini-2.5-flash-preview-05-20",
      varOrThrow("GEMINI_KEY")
    )
    break
  case "4":
    model = buildOpenAiLlm("o3", varOrThrow("OPENAI_KEY"))
    break
  default:
    console.error("Invalid choice. Please run again and select 1-4.")
    process.exit(1)
}

const instructions = "Your response must be in the form of a rhyming couplet"

const prompt = "Identify yourself and confirm you are up and running"

const response = await model.getText(prompt, instructions)

console.log(response)

function varOrThrow(key: string): string {
  if (!process.env[key]) {
    throw new Error(`Env variable ${key} not defined`)
  }
  return process.env[key]
}
