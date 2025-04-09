import { Buffer } from "node:buffer";

export default {
  async fetch (request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const errHandler = (err) => {
      console.error(err);
      return new Response(err.message, fixCors({ status: err.status ?? 500 }));
    };
    try {
      const auth = request.headers.get("Authorization");
      const apiKey = auth?.split(" ")[1];
      const assert = (success) => {
        if (!success) {
          throw new HttpError("The specified HTTP method is not allowed for the requested resource", 400);
        }
      };
      const { pathname } = new URL(request.url);
      switch (true) {
        case pathname.endsWith("/chat/completions"):
          assert(request.method === "POST");
          return handleCompletions(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/embeddings"):
          assert(request.method === "POST");
          return handleEmbeddings(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/models"):
          assert(request.method === "GET");
          return handleModels(apiKey)
            .catch(errHandler);
        default:
          throw new HttpError("404 Not Found", 404);
      }
    } catch (err) {
      return errHandler(err);
    }
  }
};

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  return { headers, status, statusText };
};

const handleOPTIONS = async () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*",
      "Access-Control-Allow-Headers": "*",
    }
  });
};

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";

// https://github.com/google-gemini/generative-ai-js/blob/cf223ff4a1ee5a2d944c53cddb8976136382bee6/src/requests/request.ts#L71
const API_CLIENT = "genai-js/0.21.0"; // npm view @google/generative-ai version
const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more
});

async function handleModels (apiKey) {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  let { body } = response;
  if (response.ok) {
    const { models } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: models.map(({ name }) => ({
        id: name.replace("models/", ""),
        object: "model",
        created: 0,
        owned_by: "",
      })),
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
async function handleEmbeddings (req, apiKey) {
  if (typeof req.model !== "string") {
    throw new HttpError("model is not specified", 400);
  }
  if (!Array.isArray(req.input)) {
    req.input = [ req.input ];
  }
  let model;
  if (req.model.startsWith("models/")) {
    model = req.model;
  } else {
    req.model = DEFAULT_EMBEDDINGS_MODEL;
    model = "models/" + req.model;
  }
  const response = await fetch(`${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({
      "requests": req.input.map(text => ({
        model,
        content: { parts: { text } },
        outputDimensionality: req.dimensions,
      }))
    })
  });
  let { body } = response;
  if (response.ok) {
    const { embeddings } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: embeddings.map(({ values }, index) => ({
        object: "embedding",
        index,
        embedding: values,
      })),
      model: req.model,
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_MODEL = "gemini-2.0-flash";
async function handleCompletions (req, apiKey) {
  let model = DEFAULT_MODEL;
  switch(true) {
    case typeof req.model !== "string":
      break;
    case req.model.startsWith("models/"):
      model = req.model.substring(7);
      break;
    case req.model.startsWith("gemini-"):
    case req.model.startsWith("gemma-"):
    case req.model.startsWith("learnlm-"):
      model = req.model;
  }
  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) { url += "?alt=sse"; }
  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(await transformRequest(req)), // try
  });

  let body = response.body;
  if (response.ok) {
    let id = generateChatcmplId(); //"chatcmpl-8pMMaqXMK68B3nyDBrapTDrhkHBQK";
    if (req.stream) {
      body = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new TransformStream({
          transform: parseStream,
          flush: parseStreamFlush,
          buffer: "",
        }))
        .pipeThrough(new TransformStream({
          transform: toOpenAiStream,
          flush: toOpenAiStreamFlush,
          streamIncludeUsage: req.stream_options?.include_usage,
          model, id,
          last: [],
          lastReceivedData: null
        }))
        .pipeThrough(new TextEncoderStream());
    } else {
      body = await response.text();
      body = processCompletionsResponse(JSON.parse(body), model, id);
    }
  }
  return new Response(body, fixCors(response));
}

const harmCategory = [
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_CIVIC_INTEGRITY",
];
const safetySettings = harmCategory.map(category => ({
  category,
  threshold: "BLOCK_NONE",
}));
const fieldsMap = {
  stop: "stopSequences",
  n: "candidateCount", // not for streaming
  max_tokens: "maxOutputTokens",
  max_completion_tokens: "maxOutputTokens",
  temperature: "temperature",
  top_p: "topP",
  top_k: "topK", // non-standard
  frequency_penalty: "frequencyPenalty",
  presence_penalty: "presencePenalty",
};
const transformConfig = (req) => {
  let cfg = {};
  //if (typeof req.stop === "string") { req.stop = [req.stop]; } // no need
  for (let key in req) {
    const matchedKey = fieldsMap[key];
    if (matchedKey) {
      cfg[matchedKey] = req[key];
    }
  }
  if (req.response_format) {
    switch(req.response_format.type) {
      case "json_schema":
        cfg.responseSchema = req.response_format.json_schema?.schema;
        if (cfg.responseSchema && "enum" in cfg.responseSchema) {
          cfg.responseMimeType = "text/x.enum";
          break;
        }
        // eslint-disable-next-line no-fallthrough
      case "json_object":
        cfg.responseMimeType = "application/json";
        break;
      case "text":
        cfg.responseMimeType = "text/plain";
        break;
      default:
        throw new HttpError("Unsupported response_format.type", 400);
    }
  }
  return cfg;
};

const parseImg = async (url) => {
  let mimeType, data;
  if (url.startsWith("http://") || url.startsWith("https://")) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText} (${url})`);
      }
      mimeType = response.headers.get("content-type");
      data = Buffer.from(await response.arrayBuffer()).toString("base64");
    } catch (err) {
      throw new Error("Error fetching image: " + err.toString());
    }
  } else {
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match) {
      throw new Error("Invalid image data: " + url);
    }
    ({ mimeType, data } = match.groups);
  }
  return {
    inlineData: {
      mimeType,
      data,
    },
  };
};

const transformMsg = async ({ role, content }) => {
  const parts = [];
  if (!Array.isArray(content)) {
    // system, user: string
    // assistant: string or null (Required unless tool_calls is specified.)
    parts.push({ text: content });
    return { role, parts };
  }
  // user:
  // An array of content parts with a defined type.
  // Supported options differ based on the model being used to generate the response.
  // Can contain text, image, or audio inputs.
  for (const item of content) {
    switch (item.type) {
      case "text":
        parts.push({ text: item.text });
        break;
      case "image_url":
        parts.push(await parseImg(item.image_url.url));
        break;
      case "input_audio":
        parts.push({
          inlineData: {
            mimeType: "audio/" + item.input_audio.format,
            data: item.input_audio.data,
          }
        });
        break;
      default:
        throw new TypeError(`Unknown "content" item type: "${item.type}"`);
    }
  }
  if (content.every(item => item.type === "image_url")) {
    parts.push({ text: "" }); // to avoid "Unable to submit request because it must have a text parameter"
  }
  return { role, parts };
};

const transformMessages = async (messages) => {
  if (!messages) { return; }
  const contents = [];
  let system_instruction;
  for (const item of messages) {
    if (item.role === "system") {
      delete item.role;
      system_instruction = await transformMsg(item);
    } else {
      item.role = item.role === "assistant" ? "model" : "user";
      contents.push(await transformMsg(item));
    }
  }
  if (system_instruction && contents.length === 0) {
    contents.push({ role: "model", parts: { text: " " } });
  }
  //console.info(JSON.stringify(contents, 2));
  return { system_instruction, contents };
};

const transformRequest = async (req) => ({
  ...await transformMessages(req.messages),
  safetySettings,
  generationConfig: transformConfig(req),
});

const generateChatcmplId = () => {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
  return "chatcmpl-" + Array.from({ length: 29 }, randomChar).join("");
};

const reasonsMap = { //https://ai.google.dev/api/rest/v1/GenerateContentResponse#finishreason
  //"FINISH_REASON_UNSPECIFIED": // Default value. This value is unused.
  "STOP": "stop",
  "MAX_TOKENS": "length",
  "SAFETY": "content_filter",
  "RECITATION": "content_filter",
  //"OTHER": "OTHER",
  // :"function_call",
};
const SEP = "\n\n|>";
const transformCandidates = (key, cand) => ({
  index: cand.index || 0, // 0-index is absent in new -002 models response
  [key]: {
    role: "assistant",
    content: cand.content?.parts.map(p => p.text).join(SEP) },
  logprobs: null,
  finish_reason: reasonsMap[cand.finishReason] || cand.finishReason,
});
const transformCandidatesMessage = transformCandidates.bind(null, "message");
const transformCandidatesDelta = transformCandidates.bind(null, "delta");

const transformUsage = (data) => ({
  completion_tokens: data.candidatesTokenCount,
  prompt_tokens: data.promptTokenCount,
  total_tokens: data.totalTokenCount
});

const processCompletionsResponse = (data, model, id) => {
  // Default empty choices and null usage
  let choices = [];
  let usage = null;

  // Safely process candidates if the array exists
  if (data && Array.isArray(data.candidates)) {
    choices = data.candidates.map(transformCandidatesMessage);
  } else {
    console.warn("processCompletionsResponse: Received non-streaming response without a valid 'candidates' array.", data);
    // Decide how to represent this: maybe an empty choices array is sufficient,
    // or perhaps add an error object to the response if appropriate.
    // For a "ping" that might just return feedback, empty choices is likely okay.
  }

  // Safely process usageMetadata if it exists
  if (data && data.usageMetadata) {
    try {
        usage = transformUsage(data.usageMetadata);
    } catch (usageError) {
        console.error("Error transforming usage metadata:", usageError, data.usageMetadata);
        // Keep usage as null if transformation fails
    }
  } else {
      console.warn("processCompletionsResponse: Received non-streaming response without 'usageMetadata'.", data);
  }

  return JSON.stringify({
    id,
    choices, // Use the safely processed choices
    created: Math.floor(Date.now()/1000),
    model,
    object: "chat.completion",
    usage, // Use the safely processed usage
  });
};

const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;
async function parseStream (chunk, controller) {
  chunk = await chunk;
  if (!chunk) { return; }
  this.buffer += chunk;
  do {
    const match = this.buffer.match(responseLineRE);
    if (!match) { break; }
    controller.enqueue(match[1]);
    this.buffer = this.buffer.substring(match[0].length);
  } while (true); // eslint-disable-line no-constant-condition
}
async function parseStreamFlush (controller) {
  if (this.buffer) {
    console.error("Invalid data:", this.buffer);
    controller.enqueue(this.buffer);
  }
}

function transformResponseStream (data, stop, first) {
  if (!data || !data.candidates || !Array.isArray(data.candidates) || data.candidates.length === 0) {
      console.error("transformResponseStream called with invalid data:", data);
      return null;
  }

  const cand = data.candidates[0];
  const item = transformCandidatesDelta(cand);

  if (stop) {
      item.delta = {};
      item.finish_reason = reasonsMap[cand.finishReason] || cand.finishReason || "stop";
  } else {
      item.finish_reason = null;
  }

  if (first) {
      item.delta = item.delta || {};
      item.delta.role = "assistant";
      item.delta.content = "";
  } else {
      if (item.delta) {
          delete item.delta.role;
      }
  }

  const choiceItem = {
      index: item.index || 0,
      delta: item.delta || {},
      logprobs: item.logprobs,
      finish_reason: item.finish_reason,
  };

  const output = {
    id: this.id,
    choices: [choiceItem],
    created: Math.floor(Date.now()/1000),
    model: this.model,
    object: "chat.completion.chunk",
  };

  return "data: " + JSON.stringify(output) + delimiter;
}
const delimiter = "\n\n";
async function toOpenAiStream (chunk, controller) {
  const transform = transformResponseStream.bind(this);
  const line = await chunk;
  if (!line) { return; }
  let data;
  try {
    data = JSON.parse(line);
    this.lastReceivedData = data;
  } catch (err) {
    console.error("Error parsing SSE line:", line, err);
    const errorOutput = {
      id: this.id,
      choices: [{
        index: 0,
        delta: { role: "assistant", content: `

[Error parsing stream data: ${err.message}]` },
        logprobs: null,
        finish_reason: "error"
      }],
      created: Math.floor(Date.now()/1000),
      model: this.model,
      object: "chat.completion.chunk",
    };
    controller.enqueue("data: " + JSON.stringify(errorOutput) + delimiter);
    return;
  }

  if (data.promptFeedback) {
    console.warn("Received promptFeedback:", JSON.stringify(data.promptFeedback));
  }

  if (data.candidates && Array.isArray(data.candidates) && data.candidates.length > 0) {
    const cand = data.candidates[0];
    cand.index = cand.index || 0;

    if (!this.last[cand.index]) {
       const firstChunkStr = transform(data, false, "first");
       if (firstChunkStr) controller.enqueue(firstChunkStr);
    }

    this.last[cand.index] = data;

    if (cand.content && cand.content.parts && cand.content.parts.length > 0) {
        const textContent = cand.content.parts.map(p => p.text).filter(Boolean).join("");
        if (textContent) {
           const contentChunkStr = transform(data);
           if (contentChunkStr) controller.enqueue(contentChunkStr);
        } else {
           console.log("Received chunk with empty content parts for index:", cand.index);
        }
    } else if (cand.finishReason) {
        console.log("Received chunk with finishReason for index:", cand.index, cand.finishReason);
    } else {
        console.log("Received chunk with candidate but no content/finishReason for index:", cand.index);
    }

  } else {
    console.log("Received SSE chunk with empty or missing candidates array:", line);
  }
}
async function toOpenAiStreamFlush (controller) {
  const transform = transformResponseStream.bind(this);
  let finalUsage = null;
  let sentFinalChunk = false;

  if (this.lastReceivedData?.usageMetadata && this.streamIncludeUsage) {
      finalUsage = transformUsage(this.lastReceivedData.usageMetadata);
      console.log("Captured final usage metadata from the last received chunk:", JSON.stringify(finalUsage));
  }

  if (this.last.length > 0) {
    for (let i = 0; i < this.last.length; i++) {
        const data = this.last[i];
        if (data) {
            const finalChunkStrBase = transform(data, "stop");

            if (finalChunkStrBase) {
                let finalChunkStrToEnqueue = finalChunkStrBase;
                if (i === this.last.length - 1 && finalUsage) {
                    try {
                        const outputJson = JSON.parse(finalChunkStrBase.substring(6));
                        outputJson.usage = finalUsage;
                        finalChunkStrToEnqueue = "data: " + JSON.stringify(outputJson) + delimiter;
                        console.log("Injected final usage into the last candidate's final chunk.");
                    } catch (e) {
                        console.error("Error injecting final usage into last chunk:", e);
                    }
                }
                controller.enqueue(finalChunkStrToEnqueue);
                sentFinalChunk = true;
            } else {
                 console.log(`Failed to transform final data for candidate index ${i}, skipping.`);
            }
        } else {
          console.log(`No final data stored for candidate index ${i}, skipping.`);
        }
    }
  }

  if (!sentFinalChunk && this.lastReceivedData) {
      console.warn("Stream ended, but no candidate data was ever processed. Constructing final chunk from last received data.");

      let finishReason = "unknown";
      let blockReasonDetails = null;

      if (this.lastReceivedData.promptFeedback?.blockReason) {
          finishReason = "content_filter";
          blockReasonDetails = this.lastReceivedData.promptFeedback.blockReason;
          console.warn("Detected blockReason in promptFeedback:", blockReasonDetails);
      } else {
          console.warn("Could not determine specific finish reason when no candidates were received.");
          finishReason = "error";
      }

      const finalChoice = {
          index: 0,
          delta: {},
          logprobs: null,
          finish_reason: finishReason,
      };

      const finalOutput = {
          id: this.id,
          choices: [finalChoice],
          created: Math.floor(Date.now()/1000),
          model: this.model,
          object: "chat.completion.chunk",
          usage: finalUsage,
      };
      controller.enqueue("data: " + JSON.stringify(finalOutput) + delimiter);
      sentFinalChunk = true;
  }

  controller.enqueue("data: [DONE]" + delimiter);
}
