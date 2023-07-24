// src/chatgpt-api.ts
import Keyv from "keyv";
import pTimeout from "p-timeout";
import QuickLRU from "quick-lru";
import { v4 as uuidv4 } from "uuid";

// src/tokenizer.ts
import GPT3TokenizerImport from "gpt3-tokenizer";
var GPT3Tokenizer = typeof GPT3TokenizerImport === "function" ? GPT3TokenizerImport : GPT3TokenizerImport.default;
var tokenizer = new GPT3Tokenizer({ type: "gpt3" });
function encode(input) {
  return tokenizer.encode(input).bpe;
}

// src/types.ts
var ChatGPTError = class extends Error {
};

// src/fetch.ts
var fetch = globalThis.fetch;

// src/fetch-sse.ts
import { createParser } from "eventsource-parser";

// src/stream-async-iterable.ts
async function* streamAsyncIterable(stream) {
  const reader = stream.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        return;
      }
      yield value;
    }
  } finally {
    reader.releaseLock();
  }
}

// src/fetch-sse.ts
async function fetchSSE(url, options, fetch2 = fetch) {
  const { onMessage, ...fetchOptions } = options;
  const res = await fetch2(url, fetchOptions);
  if (!res.ok) {
    const reason = await res.text();
    const msg = `ChatGPT error ${res.status || res.statusText}: ${reason}`;
    const error = new ChatGPTError(msg, { cause: reason });
    error.statusCode = res.status;
    error.statusText = res.statusText;
    throw error;
  }
  const parser = createParser((event) => {
    if (event.type === "event") {
      onMessage(event.data);
    }
  });
  if (!res.body.getReader) {
    const body = res.body;
    if (!body.on || !body.read) {
      throw new ChatGPTError('unsupported "fetch" implementation');
    }
    body.on("readable", () => {
      let chunk;
      while (null !== (chunk = body.read())) {
        parser.feed(chunk.toString());
      }
    });
  } else {
    for await (const chunk of streamAsyncIterable(res.body)) {
      const str = new TextDecoder().decode(chunk);
      parser.feed(str);
    }
  }
}

// src/chatgpt-api.ts
var CHATGPT_MODEL = "text-davinci-003";
var USER_LABEL_DEFAULT = "User";
var ASSISTANT_LABEL_DEFAULT = "ChatGPT";

var ChatGPTAPI = class {
  constructor(opts) {
    const {
      apiBaseUrl = "https://chatbot.theb.ai",
      organization,
      debug = false,
      messageStore,
      completionParams,
      maxModelTokens = 4096,
      maxResponseTokens = 1000,
      userLabel = USER_LABEL_DEFAULT,
      assistantLabel = ASSISTANT_LABEL_DEFAULT,
      getMessageById = this._defaultGetMessageById,
      upsertMessage = this._defaultUpsertMessage,
      fetch: fetch2 = fetch
    } = opts;

    this._apiBaseUrl = apiBaseUrl;
    this._organization = organization;
    this._debug = !!debug;
    this._fetch = fetch2;
    this._completionParams = {
      model: CHATGPT_MODEL,
      temperature: 0.8,
      top_p: 1,
      presence_penalty: 1,
      ...completionParams
    };
    this._endToken = "";
    this._sepToken = "";

    if (this._isChatGPTModel) {
      this._endToken = "";
      this._sepToken = "";
      if (!this._completionParams.stop) {
        this._completionParams.stop = [this._endToken, this._sepToken];
      }
    } else if (this._isCodexModel) {
      this._endToken = "</code>";
      this._sepToken = this._endToken;
      if (!this._completionParams.stop) {
        this._completionParams.stop = [this._endToken];
      }
    } else {
      this._endToken = "";
      this._sepToken = this._endToken;
      if (!this._completionParams.stop) {
        this._completionParams.stop = [this._endToken];
      }
    }

    this._maxModelTokens = maxModelTokens;
    this._maxResponseTokens = maxResponseTokens;
    this._userLabel = userLabel;
    this._assistantLabel = assistantLabel;
    this._getMessageById = getMessageById;
    this._upsertMessage = upsertMessage;
    this._messageStore = messageStore || new Keyv({ store: new QuickLRU({ maxSize: 1e4 }) });

    if (!this._fetch) {
      throw new Error("Invalid environment; fetch is not defined");
    }
    if (typeof this._fetch !== "function") {
      throw new Error('Invalid "fetch" is not a function');
    }
  }

  /**
   * Sends a message to ChatGPT, waits for the response to resolve, and returns
   * the response.
   *
   * If you want your response to have historical context, you must provide a valid `parentMessageId`.
   *
   * If you want to receive a stream of partial responses, use `opts.onProgress`.
   * If you want to receive the full response, including message and conversation IDs,
   * you can use `opts.onConversationResponse` or use the `ChatGPTAPI.getConversation`
   * helper.
   *
   * Set `debug: true` in the `ChatGPTAPI` constructor to log more info on the full prompt sent to the OpenAI completions API. You can override the `promptPrefix` and `promptSuffix` in `opts` to customize the prompt.
   *
   * @param message - The prompt message to send
   * @param opts.conversationId - Optional ID of a conversation to continue (defaults to a random UUID)
   * @param opts.parentMessageId - Optional ID of the previous message in the conversation (defaults to `undefined`)
   * @param opts.messageId - Optional ID of the message to send (defaults to a random UUID)
   * @param opts.promptPrefix - Optional override for the prompt prefix to send to the OpenAI completions endpoint
   * @param opts.promptSuffix - Optional override for the prompt suffix to send to the OpenAI completions endpoint
   * @param opts.timeoutMs - Optional timeout in milliseconds (defaults to no timeout)
   * @param opts.onProgress - Optional callback which will be invoked every time the partial response is updated
   * @param opts.abortSignal - Optional callback used to abort the underlying `fetch` call using an [AbortController](https://developer.mozilla.org/en-US/docs/Web/API/AbortController)
   *
   * @returns The response from ChatGPT
   */ async sendMessage(text, opts = {}) {
    const {
      conversationId = uuidv4(),
      parentMessageId,
      messageId = uuidv4(),
      timeoutMs,
      onProgress,
      stream = onProgress ? true : false
    } = opts;
    let { abortSignal } = opts;
    let abortController = null;
    if (timeoutMs && !abortSignal) {
      abortController = new AbortController();
      abortSignal = abortController.signal;
    }
    const message = {
      role: "user",
      id: messageId,
      parentMessageId,
      conversationId,
      text
    };
    await this._upsertMessage(message);
    let prompt = text;
    let maxTokens = 0;
    if (!this._isCodexModel) {
      const builtPrompt = await this._buildPrompt(text, opts);
      prompt = builtPrompt.prompt;
      maxTokens = builtPrompt.maxTokens;
    }
    const result = {
      role: "assistant",
      id: uuidv4(),
      parentMessageId: messageId,
      conversationId,
      text: ""
    };
    const responseP = new Promise(
      async (resolve, reject) => {
        var _a, _b;
        const url = `${this._apiBaseUrl}/api/chat-process`;
        const headers = {
          'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0',
          'Accept': 'application/json, text/plain, */*',
          'Accept-Language': 'en-US,en;q=0.5',
          'Content-Type': 'application/json',
          'Connection': 'keep-alive',
          'Referer': 'http://aiassist.love/',
        };

        if (this._organization) {
          headers["OpenAI-Organization"] = this._organization;
        }
        const body = {
          max_tokens: maxTokens,
          ...this._completionParams,
          prompt,
          stream
        };
        if (this._debug) {
          const numTokens = await this._getTokenCount(body.prompt);
          console.log(`sendMessage (${numTokens} tokens)`, body);
        }
        if (stream) {
          fetchSSE(
            url,
            {
              method: "POST",
              headers,
              body: JSON.stringify(body),
              signal: abortSignal,
              onMessage: (data) => {
                var _a2;
                if (data === "[DONE]") {
                  result.text = result.text.trim();
                  return resolve(result);
                }
                try {
                  const response = JSON.parse(data);
                  if (response.id) {
                    result.id = response.id;
                  }
                  if ((_a2 = response == null ? void 0 : response.choices) == null ? void 0 : _a2.length) {
                    result.text += response.choices[0].text;
                    result.detail = response;
                    onProgress == null ? void 0 : onProgress(result);
                  }
                } catch (err) {
                  console.warn("ChatGPT stream SEE event unexpected error", err);
                  return reject(err);
                }
              }
            },
            this._fetch
          ).catch(reject);
        } else {
          try {
            const res = await this._fetch(url, {
              method: "POST",
              headers,
              body: JSON.stringify(body),
              signal: abortSignal
            });
            if (!res.ok) {
              const reason = await res.text();
              const msg = `ChatGPT error ${res.status || res.statusText}: ${reason}`;
              const error = new ChatGPTError(msg, { cause: res });
              error.statusCode = res.status;
              error.statusText = res.statusText;
              return reject(error);
            }
            const response = await res.json();
            if (this._debug) {
              console.log(response);
            }
            if (response == null ? void 0 : response.id) {
              result.id = response.id;
            }
            if ((_a = response == null ? void 0 : response.choices) == null ? void 0 : _a.length) {
              result.text = response.choices[0].text.trim();
            } else {
              const res2 = response;
              return reject(
                new Error(
                  `ChatGPT error: ${((_b = res2 == null ? void 0 : res2.detail) == null ? void 0 : _b.message) || (res2 == null ? void 0 : res2.detail) || "unknown"}`
                )
              );
            }
            result.detail = response;
            return resolve(result);
          } catch (err) {
            return reject(err);
          }
        }
      }
    ).then((message2) => {
      return this._upsertMessage(message2).then(() => message2);
    });
    if (timeoutMs) {
      if (abortController) {
        ;
        responseP.cancel = () => {
          abortController.abort();
        };
      }
      return pTimeout(responseP, {
        milliseconds: timeoutMs,
        message: "ChatGPT timed out waiting for response"
      });
    } else {
      return responseP;
    }
  }
  async _buildPrompt(message, opts) {
    const currentDate = (/* @__PURE__ */ new Date()).toISOString().split("T")[0];
    const promptPrefix = opts.promptPrefix || `Instructions:
You are ${this._assistantLabel}, a large language model trained by OpenAI.
Current date: ${currentDate}${this._sepToken}

`;
    const promptSuffix = opts.promptSuffix || `

${this._assistantLabel}:
`;
    const maxNumTokens = this._maxModelTokens - this._maxResponseTokens;
    let { parentMessageId } = opts;
    let nextPromptBody = `${this._userLabel}:

${message}${this._endToken}`;
    let promptBody = "";
    let prompt;
    let numTokens;
    do {
      const nextPrompt = `${promptPrefix}${nextPromptBody}${promptSuffix}`;
      const nextNumTokens = await this._getTokenCount(nextPrompt);
      const isValidPrompt = nextNumTokens <= maxNumTokens;
      if (prompt && !isValidPrompt) {
        break;
      }
      promptBody = nextPromptBody;
      prompt = nextPrompt;
      numTokens = nextNumTokens;
      if (!isValidPrompt) {
        break;
      }
      if (!parentMessageId) {
        break;
      }
      const parentMessage = await this._getMessageById(parentMessageId);
      if (!parentMessage) {
        break;
      }
      const parentMessageRole = parentMessage.role || "user";
      const parentMessageRoleDesc = parentMessageRole === "user" ? this._userLabel : this._assistantLabel;
      const parentMessageString = `${parentMessageRoleDesc}:

${parentMessage.text}${this._endToken}

`;
      nextPromptBody = `${parentMessageString}${promptBody}`;
      parentMessageId = parentMessage.parentMessageId;
    } while (true);
    const maxTokens = Math.max(
      1,
      Math.min(this._maxModelTokens - numTokens, this._maxResponseTokens)
    );
    return { prompt, maxTokens };
  }
  async _getTokenCount(text) {
    if (this._isChatGPTModel) {
      text = text.replace(/<\|im_end\|>/g, "<|endoftext|>");
      text = text.replace(/<\|im_sep\|>/g, "<|endoftext|>");
    }
    return encode(text).length;
  }
  get _isChatGPTModel() {
    return this._completionParams.model.startsWith("text-chat") || this._completionParams.model.startsWith("text-davinci-002-render");
  }
  get _isCodexModel() {
    return this._completionParams.model.startsWith("code-");
  }
  async _defaultGetMessageById(id) {
    const res = await this._messageStore.get(id);
    if (this._debug) {
      console.log("getMessageById", id, res);
    }
    return res;
  }
  async _defaultUpsertMessage(message) {
    if (this._debug) {
      console.log("upsertMessage", message.id, message);
    }
    await this._messageStore.set(message.id, message);
  }
};

export {
  ChatGPTAPI,
  ChatGPTError
};
