package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

var (
	punctRe = regexp.MustCompile(`[^\p{L}\p{N}\s]+`)
	spaceRe = regexp.MustCompile(`\s{2,}`)
)

const currentModel = "DeepSeek-R1-Distill-Qwen-32B-AWQ"
const logFileName = "benchmark_" + currentModel + ".log"

const errorLogFileName = "benchmark_" + currentModel + "errors.log"
const apiBaseURL = "http://213.172.7.235:8000/v1"
const maxWorkers = 5

var prompt = `
You are an expert in data extraction from tabular business documents (TORG-12, UPD, etc.). Your task is to ANALYZE THE ATTACHED DOCUMENT, EXTRACT ALL COMMODITY POSITIONS, and RETURN THEM IN A STRICTLY DEFINED JSON FORMAT.
Your main goal is to extract items correctly, without any errors in names and numbers.

STRICT JSON FORMATTING RULES:
- Do not truncate the list! You MUST output the full data array. If there are 150 items in the document, there must be exactly 150 objects in the JSON.
- Escaping: Replace any double quotes (") in item names with single quotes (').

Determine the table structure. Possible types: ТОРГ-12, УПД, UNKNOWN.

1) TYPE: ТОРГ-12
- "NAME" = Column 2
- "QUANTITY" = Column 10
- "PRICE" = Column 11
- "VAT" = Column 13
- "CHECK SUM" = Column 12

2) TYPE: УПД
- "NAME" = Column 1
- "QUANTITY" = Column 3
- "PRICE" = Column 4
- "VAT" = Column 7
- "CHECK SUM" = Column 5

MANDATORY REASONING ALGORITHM:
Step 1. Determine document type and attach indices.
Step 2. COUNT THE EXACT NUMBER of item rows.
Step 3. SELECT EXACTLY 5 ITEMS for mathematical verification (PRICE * QUANTITY = SUM).
Step 4. If step 3 succeeds - proceed to step 5. If step 3 fails - try selecting different columns and repeat the math check.
Step 5. Proceed to JSON generation

THE NUMBER OF ITEMS IN JSON MUST MATCH THE REAL NUMBER OF ITEMS IN THE DOCUMENT.

Output format:
{
  "incoming_date": "",
  "incoming_number": "",
  "positions": [
    {
      "n": "Exact item name in Russian. Cutting/modifying is strictly prohibited. Do not ignore English prefixes and brands. Whatever is in the name MUST remain in the name.",
      "q": 2.0,  // put a special attention to this field. Most of your errors are about this field. Be carefull
      "p": 1150.00,
      "v": 20
    }
  ]
}

DEATH PENALTY FOR VIOLATING THESE RULES:
1) Distorting/shortening item names, translating them, rearranging words, adding or skipping words. Even if the name contains a country or an article number - leave it as is.
2) Truncating the response, not outputting the full array, breaking the JSON structure. This is a separate rule. Violation = firing squad.
3) Trying to determine the table structure more than 10 times. If you fail - return an empty json.
4) Checking EVERY item mathematically. You only need to check exactly 5 items.

Pay special attention to mathematical accuracy. Do not make mistakes in numbers.
`

type TabbyRequest struct {
	Model             string         `json:"model"`
	Messages          []TabbyMessage `json:"messages"`
	Temperature       float32        `json:"temperature"`
	MaxTokens         int            `json:"max_tokens"`
	TopP              float32        `json:"top_p"`
	RepetitionPenalty float32        `json:"repetition_penalty"`
}

type TabbyMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type TabbyResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage map[string]interface{} `json:"usage"`
}

type GroundTruthPosition struct {
	Name     string  `json:"name"`
	Quantity float64 `json:"quantity"`
	Vat      float64 `json:"vat"`
	Price    float64 `json:"price"`
}

type LLMPosition struct {
	Name     string  `json:"n"`
	Quantity float64 `json:"q"`
	Price    float64 `json:"p"`
	Tax      float64 `json:"v"`
}

type LLMDocument struct {
	Positions      []LLMPosition `json:"positions"`
	IncomingDate   string        `json:"incoming_date"`
	IncomingNumber string        `json:"incoming_number"`
}

type GlobalStats struct {
	TotalGTItems          int
	TotalErrors           int
	JsonCrashes           int
	TotalSpeed            float64
	ValidDocs             int
	TotalModelTimeSeconds float64
}

func floatsEqual(a, b float64, tolerance float64) bool {
	return math.Abs(a-b) <= tolerance
}

func cleanJSON(raw string) string {
	start := strings.Index(raw, "{")
	end := strings.LastIndex(raw, "}")
	if start != -1 && end != -1 && end >= start {
		return raw[start : end+1]
	}
	return raw
}

func prepareString(s string) string {
	s = strings.ToLower(s)
	s = punctRe.ReplaceAllString(s, "")
	s = spaceRe.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}

func levenshteinRatio(s1, s2 string) float64 {
	r1, r2 := []rune(s1), []rune(s2)
	len1, len2 := len(r1), len(r2)
	if len1 == 0 && len2 == 0 {
		return 1.0
	}
	if len1 == 0 || len2 == 0 {
		return 0.0
	}

	d := make([][]int, len1+1)
	for i := range d {
		d[i] = make([]int, len2+1)
		d[i][0] = i
	}
	for j := 0; j <= len2; j++ {
		d[0][j] = j
	}

	for i := 1; i <= len1; i++ {
		for j := 1; j <= len2; j++ {
			cost := 1
			if r1[i-1] == r2[j-1] {
				cost = 0
			}
			min := d[i-1][j] + 1
			if d[i][j-1]+1 < min {
				min = d[i][j-1] + 1
			}
			if d[i-1][j-1]+cost < min {
				min = d[i-1][j-1] + cost
			}
			d[i][j] = min
		}
	}
	maxLen := len1
	if len2 > maxLen {
		maxLen = len2
	}
	return 1.0 - float64(d[len1][len2])/float64(maxLen)
}

func main() {
	f, err := os.OpenFile(logFileName, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0666)
	if err != nil {
		log.Fatalf("Ошибка создания лога: %v", err)
	}
	defer f.Close()

	errFile, err := os.OpenFile(errorLogFileName, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0666)
	if err != nil {
		log.Fatalf("Ошибка создания лога ошибок: %v", err)
	}
	defer errFile.Close()

	mw := io.MultiWriter(os.Stdout, f)
	mainLogger := log.New(mw, "", 0)

	dataset := "../../dataset"
	entries, err := os.ReadDir(dataset)
	if err != nil {
		mainLogger.Fatalf("Ошибка чтения директории: %v", err)
	}

	client := &http.Client{
		Timeout:   30 * time.Minute,
		Transport: &http.Transport{MaxIdleConnsPerHost: maxWorkers},
	}

	stats := GlobalStats{}
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup
	var mu sync.Mutex

	mainLogger.Printf("🚀 ЗАПУСК БЕНЧМАРКА: %s (PROD ACCURACY & LEVENSHTEIN > 0.95)\n", currentModel)
	mainLogger.Println(strings.Repeat("=", 80))
	globalStart := time.Now()

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dirPath := filepath.Join(dataset, entry.Name())
		pBytes, errP := os.ReadFile(filepath.Join(dirPath, "prompt.txt"))
		gBytes, errG := os.ReadFile(filepath.Join(dirPath, "parsed.json"))
		if errP != nil || errG != nil {
			continue
		}

		wg.Add(1)
		go func(folderName string, promptBytes []byte, groundTruthBytes []byte) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			var gtPositions []GroundTruthPosition
			_ = json.Unmarshal(groundTruthBytes, &gtPositions)
			gtCount := len(gtPositions)

			cleanDoc := string(promptBytes)
			commaRe := regexp.MustCompile(`,{2,}`)
			cleanDoc = commaRe.ReplaceAllString(cleanDoc, ",")

			reqPayload := TabbyRequest{
				Model: currentModel,
				Messages: []TabbyMessage{
					{Role: "system", Content: prompt},
					{Role: "user", Content: cleanDoc},
				},
				Temperature: 0.0, MaxTokens: 16384, TopP: 1.0, RepetitionPenalty: 1.0,
			}

			jsonData, _ := json.Marshal(reqPayload)
			var docLog bytes.Buffer

			docLog.WriteString("\n" + strings.Repeat("=", 80) + "\n")
			docLog.WriteString(fmt.Sprintf("📂 ДОКУМЕНТ: %s (Ожидается товаров: %d)\n", folderName, gtCount))
			docLog.WriteString(strings.Repeat("=", 80) + "\n")

			startReq := time.Now()
			resp, errReq := client.Post(apiBaseURL+"/chat/completions", "application/json", bytes.NewBuffer(jsonData))

			var localErrors, localJsonCrashes int
			var speed, requestDuration float64

			if errReq != nil {
				docLog.WriteString(fmt.Sprintf("  ❌ ОШИБКА СЕТИ: %v\n", errReq))
				localJsonCrashes = 1
				localErrors = gtCount
			} else {
				requestDuration = time.Since(startReq).Seconds()

				body, _ := io.ReadAll(resp.Body)
				resp.Body.Close()

				var tResp TabbyResponse
				_ = json.Unmarshal(body, &tResp)

				if len(tResp.Choices) == 0 {
					docLog.WriteString("  ❌ ПУСТОЙ ОТВЕТ ОТ МОДЕЛИ\n")
					docLog.WriteString(fmt.Sprintf("  RAW BODY: %s\n", string(body)))
					localJsonCrashes = 1
					localErrors = gtCount
				} else {
					if tResp.Usage != nil {
						if compTokens, ok := tResp.Usage["completion_tokens"].(float64); ok && requestDuration > 0 {
							speed = compTokens / requestDuration
						}
					}

					raw := tResp.Choices[0].Message.Content

					docLog.WriteString(fmt.Sprintf("⏱️ RTT (Общее время): %.2f сек | Speed (расчетная): %.1f T/s\n", requestDuration, speed))
					docLog.WriteString(fmt.Sprintf("📊 USAGE ОТ СЕРВЕРА: %+v\n", tResp.Usage))
					docLog.WriteString(fmt.Sprintf("\n📝 RAW RESPONSE (ОТВЕТ LLM):\n%s\n", raw))
					docLog.WriteString(strings.Repeat("-", 60) + "\n")

					jsonBody := cleanJSON(raw)
					var llmDoc LLMDocument
					if err := json.Unmarshal([]byte(jsonBody), &llmDoc); err != nil {
						docLog.WriteString(fmt.Sprintf("  💥 КРАШ JSON ПАРСЕРА: %v\n", err))
						localJsonCrashes = 1
						localErrors = gtCount
					} else {
						llmCount := len(llmDoc.Positions)
						matches := 0
						llmMatched := make([]bool, llmCount)
						gtMatched := make([]bool, gtCount)

						// Проверка на совпадения
						for i, gt := range gtPositions {
							for j, lp := range llmDoc.Positions {
								if !llmMatched[j] && !gtMatched[i] {
									gtNorm := prepareString(gt.Name)
									lpNorm := prepareString(lp.Name)

									ratio := levenshteinRatio(gtNorm, lpNorm)
									isNameMatch := ratio > 0.95
									isMathMatch := floatsEqual(gt.Quantity, lp.Quantity, 0.001) &&
										floatsEqual(gt.Price, lp.Price, 0.01)

									if isNameMatch && isMathMatch {
										gtMatched[i], llmMatched[j] = true, true
										matches++
										docLog.WriteString(fmt.Sprintf("  ✅ ПРАВИЛЬНО: %s (Схожесть: %.2f)\n", gt.Name, ratio))
										break
									}
								}
							}
						}

						// Вывод ошибок для товаров, которые не нашли пару
						hasUnmatched := false
						for i, gt := range gtPositions {
							if !gtMatched[i] {
								if !hasUnmatched {
									docLog.WriteString("\n  --- ❌ ОШИБКИ И РАСХОЖДЕНИЯ ---\n")
									hasUnmatched = true
								}
								docLog.WriteString(fmt.Sprintf("  ❌ УПУЩЕНО В ЭТАЛОНЕ: %s | Qty: %.2f | Price: %.2f\n", gt.Name, gt.Quantity, gt.Price))
							}
						}
						for j, lp := range llmDoc.Positions {
							if !llmMatched[j] {
								if !hasUnmatched {
									docLog.WriteString("\n  --- ❌ ОШИБКИ И РАСХОЖДЕНИЯ ---\n")
									hasUnmatched = true
								}
								docLog.WriteString(fmt.Sprintf("  ⚠️ ЛИШНЕЕ/ИКАЖЕНО (LLM): %s | Qty: %.2f | Price: %.2f\n", lp.Name, lp.Quantity, lp.Price))
							}
						}

						countDiff := int(math.Abs(float64(gtCount - llmCount)))
						unmatchedLLM := llmCount - matches
						localErrors = countDiff + unmatchedLLM

						docLog.WriteString(strings.Repeat("-", 60) + "\n")
						docLog.WriteString(fmt.Sprintf("📊 Найдено: %d | Совпало: %d | ОШИБОК В ФАЙЛЕ: %d\n", llmCount, matches, localErrors))
					}
				}
			}

			mu.Lock()
			// Пишем в основной лог и консоль всегда
			mainLogger.Print(docLog.String())
			f.Sync()

			// Пишем во второй файл ТОЛЬКО если есть ошибки (несовпадения или краши)
			if localErrors > 0 || localJsonCrashes > 0 {
				_, _ = errFile.WriteString(docLog.String())
				errFile.Sync()
			}

			stats.TotalGTItems += gtCount
			stats.TotalErrors += localErrors
			stats.JsonCrashes += localJsonCrashes
			if speed > 0 {
				stats.TotalSpeed += speed
				stats.ValidDocs++
			}
			stats.TotalModelTimeSeconds += requestDuration
			mu.Unlock()

		}(entry.Name(), pBytes, gBytes)
	}

	wg.Wait()
	globalElapsed := time.Since(globalStart)

	if stats.TotalGTItems > 0 {
		accuracy := 100.0 - (float64(stats.TotalErrors) / float64(stats.TotalGTItems) * 100.0)
		if accuracy < 0 {
			accuracy = 0
		}

		// Формируем финальную плашку
		var finalLog bytes.Buffer
		finalLog.WriteString("\n" + strings.Repeat("=", 80) + "\n")
		finalLog.WriteString("🏁 ИТОГИ PROD БЕНЧМАРКА\n")
		finalLog.WriteString(fmt.Sprintf("🎯 ACCURACY (Точность) : %.2f%%\n", accuracy))
		finalLog.WriteString(fmt.Sprintf("📦 Всего эталонных поз.: %d\n", stats.TotalGTItems))
		finalLog.WriteString(fmt.Sprintf("❌ Суммарно ошибок     : %d\n", stats.TotalErrors))
		finalLog.WriteString(fmt.Sprintf("💥 Крашей JSON         : %d\n", stats.JsonCrashes))

		if stats.ValidDocs > 0 {
			avgSpeed := stats.TotalSpeed / float64(stats.ValidDocs)
			avgCleanTime := stats.TotalModelTimeSeconds / float64(stats.ValidDocs)
			finalLog.WriteString(fmt.Sprintf("\n⚡ Средняя скорость          : %.1f T/s\n", avgSpeed))
			finalLog.WriteString(fmt.Sprintf("⏱️ Ср. время на файл (RTT)  : %.2f сек\n", avgCleanTime))
		}

		finalLog.WriteString(fmt.Sprintf("\n⏱️ Общее время теста : %v\n", globalElapsed.Round(time.Second)))
		finalLog.WriteString(strings.Repeat("=", 80) + "\n")

		// Пишем итоги в консоль, основной лог и в конец файла с ошибками
		mainLogger.Print(finalLog.String())
		_, _ = errFile.WriteString(finalLog.String())
	}
}
