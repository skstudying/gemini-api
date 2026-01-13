package gemini

import (
	"encoding/json"
	"io"
	"net/http"
	"one-api/common"
	"one-api/dto"
	"one-api/logger"
	relaycommon "one-api/relay/common"
	"one-api/relay/helper"
	"one-api/service"
	"one-api/types"
	"strings"

	"github.com/pkg/errors"

	"github.com/gin-gonic/gin"
)

func GeminiTextGenerationHandler(c *gin.Context, info *relaycommon.RelayInfo, resp *http.Response) (*dto.Usage, *types.NewAPIError) {
	defer service.CloseResponseBodyGracefully(resp)

	// 读取响应体
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, types.NewOpenAIError(err, types.ErrorCodeBadResponseBody, http.StatusInternalServerError)
	}

	if common.DebugEnabled {
		println(string(responseBody))
	}

	// 解析为 Gemini 原生响应格式
	var geminiResponse dto.GeminiChatResponse
	err = common.Unmarshal(responseBody, &geminiResponse)
	if err != nil {
		return nil, types.NewOpenAIError(err, types.ErrorCodeBadResponseBody, http.StatusInternalServerError)
	}

	// 检查是否被内容审核拦截
	if geminiResponse.PromptFeedback.IsBlocked() {
		logger.LogWarn(c, "Gemini content blocked: "+geminiResponse.PromptFeedback.BlockReason)
		return nil, types.NewErrorWithStatusCode(
			errors.New("content blocked by Gemini safety filter: "+geminiResponse.PromptFeedback.BlockReason),
			types.ErrorCodeContentFiltered,
			http.StatusBadRequest,
			types.ErrOptionWithSkipRetry(),
		)
	}

	// 计算使用量（基于 UsageMetadata）
	usage := dto.Usage{
		PromptTokens:     geminiResponse.UsageMetadata.PromptTokenCount,
		CompletionTokens: geminiResponse.UsageMetadata.CandidatesTokenCount + geminiResponse.UsageMetadata.ThoughtsTokenCount,
		TotalTokens:      geminiResponse.UsageMetadata.TotalTokenCount,
	}

	usage.CompletionTokenDetails.ReasoningTokens = geminiResponse.UsageMetadata.ThoughtsTokenCount

	for _, detail := range geminiResponse.UsageMetadata.PromptTokensDetails {
		if detail.Modality == "AUDIO" {
			usage.PromptTokensDetails.AudioTokens = detail.TokenCount
		} else if detail.Modality == "TEXT" {
			usage.PromptTokensDetails.TextTokens = detail.TokenCount
		}
	}

	// 如果响应体中未包含 usageMetadata.candidatesTokenCount 字段，说明没有有效响应，返回错误
	// 注意：使用 map 解析以判断字段是否存在（与值为 0 区分开）
	var raw map[string]interface{}
	if err := json.Unmarshal(responseBody, &raw); err == nil {
		if um, ok := raw["usageMetadata"].(map[string]interface{}); ok {
			if _, has := um["candidatesTokenCount"]; !has {
				logger.LogWarn(c, "Gemini response missing candidatesTokenCount, treating as empty response")
				return nil, types.NewErrorWithStatusCode(
					errors.New("no response from Gemini API"),
					types.ErrorCodeEmptyResponse,
					http.StatusBadRequest,
					types.ErrOptionWithSkipRetry(),
				)
			}
		}
	}

	service.IOCopyBytesGracefully(c, resp, responseBody)

	return &usage, nil
}

func NativeGeminiEmbeddingHandler(c *gin.Context, resp *http.Response, info *relaycommon.RelayInfo) (*dto.Usage, *types.NewAPIError) {
	defer service.CloseResponseBodyGracefully(resp)

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, types.NewOpenAIError(err, types.ErrorCodeBadResponseBody, http.StatusInternalServerError)
	}

	if common.DebugEnabled {
		println(string(responseBody))
	}

	usage := &dto.Usage{
		PromptTokens: info.PromptTokens,
		TotalTokens:  info.PromptTokens,
	}

	if info.IsGeminiBatchEmbedding {
		var geminiResponse dto.GeminiBatchEmbeddingResponse
		err = common.Unmarshal(responseBody, &geminiResponse)
		if err != nil {
			return nil, types.NewOpenAIError(err, types.ErrorCodeBadResponseBody, http.StatusInternalServerError)
		}
	} else {
		var geminiResponse dto.GeminiEmbeddingResponse
		err = common.Unmarshal(responseBody, &geminiResponse)
		if err != nil {
			return nil, types.NewOpenAIError(err, types.ErrorCodeBadResponseBody, http.StatusInternalServerError)
		}
	}

	service.IOCopyBytesGracefully(c, resp, responseBody)

	return usage, nil
}

func GeminiTextGenerationStreamHandler(c *gin.Context, info *relaycommon.RelayInfo, resp *http.Response) (*dto.Usage, *types.NewAPIError) {
	var usage = &dto.Usage{}
	var imageCount int

	helper.SetEventStreamHeaders(c)

	responseText := strings.Builder{}

	// 用于延迟发送的状态
	var pendingData string          // 暂存的第一个消息块
	var isPendingEmptyThought bool  // 第一个消息块是否是空思考
	var contentBlocked bool         // 是否被内容拦截
	var blockReason string          // 拦截原因

	helper.StreamScannerHandler(c, resp, info, func(data string) bool {
		var geminiResponse dto.GeminiChatResponse
		err := common.UnmarshalJsonStr(data, &geminiResponse)
		if err != nil {
			logger.LogError(c, "error unmarshalling stream response: "+err.Error())
			return false
		}

		// 检查是否被内容审核拦截
		if geminiResponse.PromptFeedback.IsBlocked() {
			// 如果之前有暂存的空思考消息，说明这是拦截场景
			if isPendingEmptyThought {
				contentBlocked = true
				blockReason = geminiResponse.PromptFeedback.BlockReason
				logger.LogWarn(c, "Gemini content blocked (detected empty thought pattern): "+blockReason)
				return false // 停止处理，不发送任何数据
			}
			// 如果没有暂存的消息，说明流已经开始了，只记录日志
			logger.LogWarn(c, "Gemini content blocked (stream already started): "+geminiResponse.PromptFeedback.BlockReason)
		}

		// 第一个消息块的特殊处理
		if info.SendResponseCount == 0 && pendingData == "" {
			// 检查是否是"空思考"模式
			if dto.IsEmptyThoughtResponse(&geminiResponse) {
				// 暂存第一个消息块，等待第二个消息块确认
				pendingData = data
				isPendingEmptyThought = true
				logger.LogDebug(c, "Detected potential empty thought pattern, pending first chunk")
				return true // 继续接收下一个消息块
			}
		}

		// 如果有暂存的消息，先发送暂存的
		if pendingData != "" {
			err = helper.StringData(c, pendingData)
			if err != nil {
				logger.LogError(c, err.Error())
			}
			info.SendResponseCount++
			pendingData = ""
			isPendingEmptyThought = false
		}

		// 统计图片数量
		for _, candidate := range geminiResponse.Candidates {
			for _, part := range candidate.Content.Parts {
				if part.InlineData != nil && part.InlineData.MimeType != "" {
					imageCount++
				}
				if part.Text != "" {
					responseText.WriteString(part.Text)
				}
			}
		}

		// 更新使用量统计
		if geminiResponse.UsageMetadata.TotalTokenCount != 0 {
			usage.PromptTokens = geminiResponse.UsageMetadata.PromptTokenCount
			usage.CompletionTokens = geminiResponse.UsageMetadata.CandidatesTokenCount + geminiResponse.UsageMetadata.ThoughtsTokenCount
			usage.TotalTokens = geminiResponse.UsageMetadata.TotalTokenCount
			usage.CompletionTokenDetails.ReasoningTokens = geminiResponse.UsageMetadata.ThoughtsTokenCount
			for _, detail := range geminiResponse.UsageMetadata.PromptTokensDetails {
				if detail.Modality == "AUDIO" {
					usage.PromptTokensDetails.AudioTokens = detail.TokenCount
				} else if detail.Modality == "TEXT" {
					usage.PromptTokensDetails.TextTokens = detail.TokenCount
				}
			}
		}

		// 发送当前 GeminiChatResponse 响应
		err = helper.StringData(c, data)
		if err != nil {
			logger.LogError(c, err.Error())
		}
		info.SendResponseCount++
		return true
	})

	// 如果内容被拦截（在发送任何数据之前），返回 400 错误
	if contentBlocked {
		return nil, types.NewErrorWithStatusCode(
			errors.New("content blocked by Gemini safety filter: "+blockReason),
			types.ErrorCodeContentFiltered,
			http.StatusBadRequest,
			types.ErrOptionWithSkipRetry(),
		)
	}

	// 如果还有暂存的消息没发送（流意外结束），发送它
	if pendingData != "" {
		err := helper.StringData(c, pendingData)
		if err != nil {
			logger.LogError(c, err.Error())
		}
		info.SendResponseCount++
	}

	if info.SendResponseCount == 0 {
		return nil, types.NewOpenAIError(errors.New("no response received from Gemini API"), types.ErrorCodeEmptyResponse, http.StatusInternalServerError)
	}

	if imageCount != 0 {
		if usage.CompletionTokens == 0 {
			usage.CompletionTokens = imageCount * 258
		}
	}

	// 如果usage.CompletionTokens为0，则使用本地统计的completion tokens
	if usage.CompletionTokens == 0 {
		str := responseText.String()
		if len(str) > 0 {
			usage = service.ResponseText2Usage(responseText.String(), info.UpstreamModelName, info.PromptTokens)
		} else {
			// 空补全，不需要使用量
			usage = &dto.Usage{}
		}
	}

	// 移除流式响应结尾的[Done]，因为Gemini API没有发送Done的行为
	//helper.Done(c)

	return usage, nil
}
