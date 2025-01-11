export type ModelApi = {
  getText: (
    prompt: string | ChatPrompt,
    instructions?: string
  ) => Promise<Response>
  getStream: (
    prompt: string | ChatPrompt,
    instructions?: string
  ) => Promise<StreamResponse>
}

export type ChatPrompt = {
  role: "user" | "assistant"
  prompt: string
}[]

export type StreamResponse = {
  stream: AsyncIterable<string>
  getFinalResponse: () => Promise<Response>
}

export type Response = {
  uuid: string
  text: string
  inputTokens?: number
  outputTokens?: number
}
