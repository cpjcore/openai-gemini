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
  
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
      body: JSON.stringify(await transformRequest(req)),
    });

    let body = response.body;
    let id = generateChatcmplId();
    
    if (!response.ok) {
      console.error("[ERROR] API请求失败:", response.status, response.statusText);
      
      if (req.stream) {
        // 对于流请求，返回一个错误流，但保持SSE连接格式
        const errorStream = new ReadableStream({
          start(controller) {
            // 发送错误开始
            controller.enqueue(`data: ${JSON.stringify({
              id,
              choices: [{
                index: 0,
                delta: {
                  role: "assistant",
                  content: ""
                },
                finish_reason: null
              }],
              created: Math.floor(Date.now()/1000),
              model,
              object: "chat.completion.chunk"
            })}\n\n`);
            
            // 结束流
            controller.enqueue(`data: ${JSON.stringify({
              id,
              choices: [{
                index: 0,
                delta: {},
                finish_reason: "error"
              }],
              created: Math.floor(Date.now()/1000),
              model,
              object: "chat.completion.chunk"
            })}\n\n`);
            
            controller.enqueue("data: [DONE]\n\n");
            controller.close();
          }
        });
        
        return new Response(errorStream, fixCors({
          status: 200, // 返回200以保持流连接
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
          }
        }));
      } else {
        // 对于非流请求，返回标准错误响应
        return new Response(JSON.stringify({
          error: {
            message: `API请求失败: ${response.status} ${response.statusText}`,
            type: "api_error",
            code: response.status
          }
        }), fixCors({ status: 200 })); // 使用200而不是错误状态码
      }
    }

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
          model, id, last: [],
        }))
        .pipeThrough(new TextEncoderStream());
        
      return new Response(body, fixCors({
        status: 200,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive"
        }
      }));
    } else {
      try {
        const text = await response.text();
        let jsonData;
        
        try {
          jsonData = JSON.parse(text);
        } catch (jsonErr) {
          console.error("[ERROR] 无法解析API响应JSON:", jsonErr);
          console.error("[DEBUG] 原始响应:", text);
          
          return new Response(JSON.stringify({
            id,
            choices: [{
              index: 0,
              message: {
                role: "assistant",
                content: "服务返回了无效的JSON格式"
              },
              logprobs: null,
              finish_reason: "error"
            }],
            created: Math.floor(Date.now()/1000),
            model,
            object: "chat.completion",
            usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
          }), fixCors({ status: 200 }));
        }
        
        body = processCompletionsResponse(jsonData, model, id);
        return new Response(body, fixCors({ status: 200 }));
      } catch (err) {
        console.error("[ERROR] 处理响应文本时出错:", err);
        
        return new Response(JSON.stringify({
          id,
          choices: [{
            index: 0,
            message: {
              role: "assistant",
              content: "处理响应时出错"
            },
            logprobs: null,
            finish_reason: "error"
          }],
          created: Math.floor(Date.now()/1000),
          model,
          object: "chat.completion",
          usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
        }), fixCors({ status: 200 }));
      }
    }
  } catch (err) {
    console.error("[ERROR] 处理请求时发生异常:", err);
    
    // 即使发生异常，也确保返回有效响应
    const id = generateChatcmplId();
    
    if (req.stream) {
      // 对于流请求，返回一个错误流，但保持SSE连接格式
      const errorStream = new ReadableStream({
        start(controller) {
          // 发送错误开始
          controller.enqueue(`data: ${JSON.stringify({
            id,
            choices: [{
              index: 0,
              delta: {
                role: "assistant",
                content: ""
              },
              finish_reason: null
            }],
            created: Math.floor(Date.now()/1000),
            model,
            object: "chat.completion.chunk"
          })}\n\n`);
          
          // 结束流
          controller.enqueue(`data: ${JSON.stringify({
            id,
            choices: [{
              index: 0,
              delta: {},
              finish_reason: "error"
            }],
            created: Math.floor(Date.now()/1000),
            model,
            object: "chat.completion.chunk"
          })}\n\n`);
          
          controller.enqueue("data: [DONE]\n\n");
          controller.close();
        }
      });
      
      return new Response(errorStream, fixCors({
        status: 200,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive"
        }
      }));
    } else {
      return new Response(JSON.stringify({
        id,
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            content: "处理请求时发生错误"
          },
          logprobs: null,
          finish_reason: "error"
        }],
        created: Math.floor(Date.now()/1000),
        model,
        object: "chat.completion",
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      }), fixCors({ status: 200 }));
    }
  }
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
const transformCandidates = (key, cand) => {
  try {
    // 安全检查
    if (!cand) {
      console.error("[ERROR] 候选项为空");
      return {
        index: 0,
        [key]: {
          role: "assistant",
          content: "处理候选项时出错"
        },
        logprobs: null,
        finish_reason: "error"
      };
    }

    // 检查content和parts是否存在
    let contentText = "";
    if (cand.content && Array.isArray(cand.content.parts)) {
      try {
        contentText = cand.content.parts.map(p => p?.text || "").join(SEP);
      } catch (err) {
        console.error("[ERROR] 处理content.parts时出错:", err);
        console.error("[DEBUG] content.parts内容:", JSON.stringify(cand.content.parts));
        contentText = "内容处理出错";
      }
    } else if (cand.content) {
      // content存在但parts不是数组
      console.error("[ERROR] content.parts不是数组或不存在");
      console.error("[DEBUG] content内容:", JSON.stringify(cand.content));
      contentText = "API返回了不完整的内容";
    } else {
      // content不存在
      console.error("[ERROR] 候选项缺少content字段");
      console.error("[DEBUG] 候选项内容:", JSON.stringify(cand));
      contentText = "无法获取内容";
    }

    return {
      index: cand.index || 0, // 0-index is absent in new -002 models response
      [key]: {
        role: "assistant",
        content: contentText
      },
      logprobs: null,
      finish_reason: reasonsMap[cand.finishReason] || cand.finishReason || "error",
    };
  } catch (err) {
    console.error("[ERROR] 转换候选项时发生错误:", err);
    console.error("[DEBUG] 候选项内容:", JSON.stringify(cand));
    return {
      index: 0,
      [key]: {
        role: "assistant",
        content: "转换候选项时出错"
      },
      logprobs: null,
      finish_reason: "error"
    };
  }
};
const transformCandidatesMessage = transformCandidates.bind(null, "message");
const transformCandidatesDelta = transformCandidates.bind(null, "delta");

const transformUsage = (data) => {
  try {
    if (!data) {
      console.error("[ERROR] usageMetadata为空");
      return {
        completion_tokens: 0,
        prompt_tokens: 0,
        total_tokens: 0
      };
    }
    
    return {
      completion_tokens: data.candidatesTokenCount || 0,
      prompt_tokens: data.promptTokenCount || 0,
      total_tokens: data.totalTokenCount || 0
    };
  } catch (err) {
    console.error("[ERROR] 处理usageMetadata时出错:", err);
    console.error("[DEBUG] usageMetadata内容:", JSON.stringify(data));
    return {
      completion_tokens: 0,
      prompt_tokens: 0,
      total_tokens: 0
    };
  }
};

const processCompletionsResponse = (data, model, id) => {
  try {
    // 对数据完整性进行详细检查
    if (!data) {
      console.error("[ERROR] Gemini API返回空数据");
      console.error("[DEBUG] 完整响应:", JSON.stringify(data));
      // 返回一个合理的空响应
      return JSON.stringify({
        id,
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            content: "抱歉，服务暂时无法提供回答。"
          },
          logprobs: null,
          finish_reason: "error"
        }],
        created: Math.floor(Date.now()/1000),
        model,
        object: "chat.completion",
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
    }
    
    // 检查是否为ping消息
    // ping消息通常只有usageMetadata和modelVersion，没有candidates，且promptTokenCount很低
    if (!data.candidates && data.usageMetadata && 
        data.modelVersion && 
        data.usageMetadata.promptTokenCount <= 5) {
      console.log("[INFO] 检测到ping消息");
      console.log("[DEBUG] Ping消息内容:", JSON.stringify(data));
      
      // 为ping消息创建一个特殊响应
      return JSON.stringify({
        id,
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            content: "pong"
          },
          logprobs: null,
          finish_reason: "stop"
        }],
        created: Math.floor(Date.now()/1000),
        model,
        object: "chat.completion",
        usage: transformUsage(data.usageMetadata)
      });
    }
    
    // 检查candidates是否存在
    if (!data.candidates) {
      console.error("[ERROR] Gemini API返回的数据中缺少candidates字段");
      console.error("[DEBUG] 完整响应:", JSON.stringify(data));
      // 返回包含错误信息的响应
      return JSON.stringify({
        id,
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            content: "API服务返回了异常数据结构。"
          },
          logprobs: null,
          finish_reason: "error"
        }],
        created: Math.floor(Date.now()/1000),
        model,
        object: "chat.completion",
        usage: data.usageMetadata ? transformUsage(data.usageMetadata) : { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
    }
    
    // 检查candidates是否为数组
    if (!Array.isArray(data.candidates)) {
      console.error("[ERROR] Gemini API返回的candidates不是数组");
      console.error("[DEBUG] candidates类型:", typeof data.candidates);
      console.error("[DEBUG] 完整响应:", JSON.stringify(data));
      
      // 尝试转换非数组的candidates为数组
      const candidatesArray = [].concat(data.candidates).filter(Boolean);
      
      if (candidatesArray.length > 0) {
        // 如果成功转换，继续使用转换后的数组
        data.candidates = candidatesArray;
      } else {
        // 返回一个包含错误信息的响应
        return JSON.stringify({
          id,
          choices: [{
            index: 0,
            message: {
              role: "assistant",
              content: "API服务返回了意外的数据格式。"
            },
            logprobs: null,
            finish_reason: "error"
          }],
          created: Math.floor(Date.now()/1000),
          model,
          object: "chat.completion",
          usage: data.usageMetadata ? transformUsage(data.usageMetadata) : { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
        });
      }
    }
    
    // 检查candidates是否为空数组
    if (data.candidates.length === 0) {
      console.error("[ERROR] Gemini API返回的candidates是空数组");
      console.error("[DEBUG] 完整响应:", JSON.stringify(data));
      // 返回一个空的但有效的响应
      return JSON.stringify({
        id,
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            content: "API服务无法生成回答。"
          },
          logprobs: null,
          finish_reason: "error"
        }],
        created: Math.floor(Date.now()/1000),
        model,
        object: "chat.completion",
        usage: data.usageMetadata ? transformUsage(data.usageMetadata) : { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
    }
    
    // 正常处理有效响应
    return JSON.stringify({
      id,
      choices: data.candidates.map((cand, index) => {
        try {
          return transformCandidatesMessage(cand);
        } catch (err) {
          console.error(`[ERROR] 转换候选项 #${index} 时出错:`, err.message);
          console.error("[DEBUG] 候选项内容:", JSON.stringify(cand));
          // 返回一个安全的替代项
          return {
            index,
            message: {
              role: "assistant",
              content: "内容处理出错，无法显示完整回答。"
            },
            logprobs: null,
            finish_reason: "error"
          };
        }
      }),
      created: Math.floor(Date.now()/1000),
      model,
      object: "chat.completion",
      usage: data.usageMetadata ? transformUsage(data.usageMetadata) : { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    });
  } catch (error) {
    console.error("[ERROR] 处理完成响应时发生错误:", error);
    console.error("[DEBUG] 尝试处理的数据:", JSON.stringify(data));
    // 即使在处理函数发生错误时也能返回有效响应
    return JSON.stringify({
      id,
      choices: [{
        index: 0,
        message: {
          role: "assistant",
          content: "处理响应时发生内部错误。"
        },
        logprobs: null,
        finish_reason: "error"
      }],
      created: Math.floor(Date.now()/1000),
      model,
      object: "chat.completion",
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    });
  }
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
  try {
    const item = transformCandidatesDelta(data.candidates[0]);
    
    // 检查是否为错误消息，且是否应该忽略
    if (item.finish_reason === "error" && !stop && !first) {
      // 对于中间出现的错误，不将其添加到流中
      console.error("[WARN] 忽略流中间出现的错误消息，避免污染内容");
      return null; // 返回null表示不输出这个消息
    }
    
    if (stop) { item.delta = {}; } else { item.finish_reason = null; }
    if (first) { item.delta.content = ""; } else { delete item.delta.role; }
    
    const output = {
      id: this.id,
      choices: [item],
      created: Math.floor(Date.now()/1000),
      model: this.model,
      //system_fingerprint: "fp_69829325d0",
      object: "chat.completion.chunk",
    };
    
    if (data.usageMetadata && this.streamIncludeUsage) {
      output.usage = stop ? transformUsage(data.usageMetadata) : null;
    }
    
    return "data: " + JSON.stringify(output) + delimiter;
  } catch (err) {
    console.error("[ERROR] 转换流响应时出错:", err);
    // 对于转换错误，返回null而不是错误消息，避免污染内容
    return null;
  }
}
const delimiter = "\n\n";
async function toOpenAiStream (chunk, controller) {
  const transform = transformResponseStream.bind(this);
  const line = await chunk;
  if (!line) { return; }
  
  let data;
  let shouldProcess = true; // 标记是否处理当前数据
  let isError = false;      // 标记当前数据是否为错误
  
  try {
    data = JSON.parse(line);
  } catch (err) {
    console.error("[ERROR] 解析流数据行失败:", err);
    console.error("[DEBUG] 原始行数据:", line);
    
    // 判断当前是否已经有内容在输出
    if (this.last.length > 0) {
      // 如果已经有内容在输出，则不插入错误消息，避免破坏内容
      console.error("[WARN] 已有内容输出，跳过错误消息但保持连接");
      shouldProcess = false; // 不处理这个错误数据，但不中断连接
    } else {
      // 创建一个表示错误的候选项，但使用空内容避免污染输出
      const candidates = [{
        finishReason: "error",
        content: { parts: [{ text: "" }] },
        index: 0,
      }];
      data = { candidates };
      isError = true;
    }
  }
  
  // 如果不应处理，直接返回但不中断连接
  if (!shouldProcess) {
    return;
  }
  
  // 检查是否为ping消息
  if (!data.candidates && data.usageMetadata && 
      data.modelVersion && 
      data.usageMetadata.promptTokenCount <= 5) {
    console.log("[INFO] 检测到流式ping消息");
    
    // 为ping消息创建一个特殊候选项
    data.candidates = [{
      finishReason: "stop",
      content: { parts: [{ text: "pong" }] },
      index: 0,
    }];
  }
  // 检查candidates是否存在且为数组
  else if (!data.candidates || !Array.isArray(data.candidates)) {
    console.error("[ERROR] 流响应中缺少candidates数组或格式不正确");
    console.error("[DEBUG] 响应数据:", JSON.stringify(data));
    
    // 判断当前是否已经有内容在输出
    if (this.last.length > 0) {
      // 如果已经有内容在输出，则不插入错误消息，避免破坏内容
      console.error("[WARN] 已有内容输出，跳过错误消息但保持连接");
      shouldProcess = false; // 不处理这个错误数据，但不中断连接
    } else {
      // 创建一个有效的candidates数组，但使用空内容避免污染输出
      data.candidates = [{
        finishReason: "error",
        content: { parts: [{ text: "" }] },
        index: 0,
      }];
      isError = true;
    }
  }
  
  // 如果不应处理，直接返回但不中断连接
  if (!shouldProcess) {
    return;
  }
  
  // 检查candidates是否为空数组
  if (data.candidates.length === 0) {
    console.error("[ERROR] 流响应中candidates是空数组");
    console.error("[DEBUG] 响应数据:", JSON.stringify(data));
    
    // 判断当前是否已经有内容在输出
    if (this.last.length > 0) {
      // 如果已经有内容在输出，则不插入错误消息，避免破坏内容
      console.error("[WARN] 已有内容输出，跳过错误消息但保持连接");
      shouldProcess = false; // 不处理这个错误数据，但不中断连接
    } else {
      // 添加一个默认候选项，但使用空内容避免污染输出
      data.candidates = [{
        finishReason: "error",
        content: { parts: [{ text: "" }] },
        index: 0,
      }];
      isError = true;
    }
  }
  
  // 如果不应处理，直接返回但不中断连接
  if (!shouldProcess) {
    return;
  }
  
  try {
    const cand = data.candidates[0];
    
    // 确保index存在
    cand.index = cand.index || 0; // absent in new -002 models response
    
    // 如果是错误且已经有输出，跳过处理
    if (isError && this.last.length > 0) {
      console.error("[WARN] 跳过错误处理以避免中断已有输出");
      return;
    }
    
    if (!this.last[cand.index]) {
      const result = transform(data, false, "first");
      if (result) { // 只有在transform返回有效内容时才发送
        controller.enqueue(result);
      }
    }
    
    this.last[cand.index] = data;
    
    // 检查content是否存在
    if (cand.content) {
      const result = transform(data);
      if (result) { // 只有在transform返回有效内容时才发送
        controller.enqueue(result);
      }
    } else if (!isError) { // 只有在非错误情况下才记录
      console.error("[ERROR] 候选项缺少content字段");
      console.error("[DEBUG] 候选项内容:", JSON.stringify(cand));
      // 不发送空内容，但保留在last中以便在flush时正确处理
    }
  } catch (err) {
    console.error("[ERROR] 处理流数据时出错:", err);
    console.error("[DEBUG] 尝试处理的数据:", JSON.stringify(data));
    
    // 判断当前是否已经有内容在输出
    if (this.last.length > 0) {
      // 如果已经有内容在输出，则不插入错误消息，避免破坏内容
      console.error("[WARN] 已有内容输出，跳过错误消息但保持连接");
      // 不处理此错误，但不中断连接
      return;
    }
    
    // 替换为错误候选项，但使用空内容避免污染输出
    const errorData = {
      candidates: [{
        finishReason: "error",
        content: { parts: [{ text: "" }] },
        index: 0,
      }]
    };
    this.last = [errorData];
    const result = transform(errorData, false, "first");
    if (result) { // 只有在transform返回有效内容时才发送
      controller.enqueue(result);
    }
  }
}
async function toOpenAiStreamFlush (controller) {
  const transform = transformResponseStream.bind(this);
  if (this.last.length > 0) {
    for (const data of this.last) {
      const result = transform(data, "stop");
      if (result) { // 只有在transform返回有效内容时才发送
        controller.enqueue(result);
      }
    }
    controller.enqueue("data: [DONE]" + delimiter);
  }
}
