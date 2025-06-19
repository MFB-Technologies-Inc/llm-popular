import { buildLlamaLlm } from "../src/index.js"
import * as dotenv from "dotenv"

dotenv.config()

// Helper function to get environment variable or throw error
function varOrThrow(key: string): string {
  if (!process.env[key]) {
    throw new Error(`Env variable ${key} not defined`)
  }
  return process.env[key]
}

async function main() {
  console.log("🦙 Llama 4 Multiple Runs Example")
  console.log("================================\n")

  // Build a Llama 4 model instance
  const model = buildLlamaLlm("us.meta.llama4-maverick-17b-instruct-v1:0", {
    awsAccessKey: varOrThrow("AWS_ACCESS_KEY"),
    awsSecret: varOrThrow("AWS_SECRET"),
    region: "us-east-1"
  })

  // Define the prompt and instructions
  const instructions = "You are a creative AI assistant. Keep your responses brief and engaging."
  const prompt = "Generate a unique, creative name for a fictional planet and describe one interesting feature about it in a single sentence."

  console.log("Instructions:", instructions)
  console.log("Prompt:", prompt)
  console.log("\nRunning 10 iterations...\n")

  // Run the prompt 10 times
  const results: string[] = []
  
  for (let i = 1; i <= 10; i++) {
    try {
      console.log(`🌍 Iteration ${i}:`)
      const startTime = Date.now()
      
      const response = await model.getText(prompt, instructions)
      const endTime = Date.now()
      
      console.log(`Response: ${response.text}`)
      console.log(`Time: ${endTime - startTime}ms`)
      console.log(`Tokens - Input: ${response.inputTokens}, Output: ${response.outputTokens}`)
      console.log("---")
      
      results.push(response.text)
    } catch (error) {
      console.error(`Error on iteration ${i}:`, error)
    }
  }

  // Summary
  console.log("\n📊 Summary")
  console.log("==========")
  console.log(`Total successful runs: ${results.length}/10`)
  console.log("\nAll generated planet names:")
  results.forEach((result, index) => {
    console.log(`${index + 1}. ${result}`)
  })
}

// Run the main function
main().catch(error => {
  console.error("Fatal error:", error)
  process.exit(1)
})