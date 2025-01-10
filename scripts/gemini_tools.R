escape_quotes = function(txt, double_quotes=TRUE, single_quotes=TRUE) {
  if (single_quotes) {
    txt = gsub("'","\\'",txt, fixed=TRUE)
  }
  if (double_quotes) {
    txt = gsub('"','\\"',txt, fixed=TRUE)
  }
  txt
}

example = function() {
  cat(gemini_curl_cmd("Tell a 'joke'","my_key",json_mode = TRUE))
  run_gemini("Tell a joke.","my_key",json_mode = TRUE)
}

example = function() {
  res = readRDS("C:/libraries/gpt/gemini/result.Rds")
  df = gemini_result_to_df(res, artid="myart")
}

gemini_result_to_df = function(res, ...) {
  if (!is.null(res$error)) {
    li = c(
      list(...),
      res[c("model","json_mode","temperature")],
      list(
        error = res$error$message,
        finishReason = "error",
        content = NA
      )
    )
  } else {
    li = c(
      list(...),
      res[c("model","json_mode","temperature")],
      list(
        error = "",
        finishReason = res$candidates$finishReason[1],
        content = paste0(unlist(res$candidates$content,use.names = FALSE), collapse="")
      )
    )
  }
  return(as.data.frame(li))
}

run_gemini = function(prompt, api_key, model="gemini-1.5-flash", json_mode=FALSE, temperature=0.1, add_prompt=FALSE, verbose=TRUE) {
  library(httr)
  library(jsonlite)
  
  # 生成設定を最適化
  generationConfig = list(
    temperature = temperature,
    maxOutputTokens = 30000,    # 長い応答に対応
    topK = 40,
    topP = 1,                   # より確実な出力のために1に設定
    stopSequences = list()      # 途中で停止しないように
  )
  
  if (json_mode) {
    generationConfig$response_mime_type = "application/json"
  }
  
  # HTTPリクエストの設定を強化
  response <- POST(
    url = paste0("https://generativelanguage.googleapis.com/v1beta/models/", model,":generateContent"),
    query = list(key = api_key),
    content_type_json(),
    encode = "json",
    body = list(
      contents = list(
        parts = list(
          list(text = prompt)
        )
      ),
      generationConfig = generationConfig
    ),
    config = httr::config(
      timeout = 300,          # タイムアウトを5分に設定
      followlocation = TRUE,
      maxredirs = 10,
      encoding = "utf-8"
    )
  )
  
  status_code = status_code(response)
  json = content(response, "text")
  if (verbose) {
    cat("\n\nResult:\n",nchar(json), " characters:\n\n",json)
  }
  
  res = try(fromJSON(json),silent = TRUE)
  if (is(res, "try-error")) {
    res = list(status_code = status_code,parse_error=TRUE, json=json)
    return(res)
  }
  res$status_code = status_code
  res$parse_error = FALSE
  if (add_prompt) {
    res$prompt = prompt
  }
  res$model = model
  res$json_mode = json_mode
  res$temperature = temperature
  res
}

run_gemini_embedding = function(text, api_key, model="gemini-1.5-flash", add_text=FALSE, verbose=TRUE) {
  library(httr)
  library(jsonlite)
  cat("\nCreate embedding:\n")
  response <- POST(
    url = paste0("https://generativelanguage.googleapis.com/v1beta/models/", model,":generateEmbeddings"),
    query = list(key = api_key),
    content_type_json(),
    encode = "json",
    body = list(
      contents = list(
        text = list(text=text)
      )
    ),
    config = httr::config(
      timeout = 300,          # タイムアウトを5分に設定
      followlocation = TRUE,
      maxredirs = 10,
      encoding = "utf-8"
    )
  )
  
  status_code = status_code(response)
  json = content(response, "text")
  if (verbose) {
    cat("\n\nResult:\n",nchar(json), " characters:\n\n",json)
  }
  
  res = try(fromJSON(json),silent = TRUE)
  if (is(res, "try-error")) {
    res = list(status_code = status_code,parse_error=TRUE, json=json)
    return(res)
  }
  res$status_code = status_code
  res$parse_error = FALSE
  if (add_text) {
    res$text = text
  }
  res$model = model
  res
}
