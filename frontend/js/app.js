/**
 * Intelligent Stock Market Prediction System - Frontend Client Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    let currentTicker = "AAPL";
    let currentPeriod = "1y";
    let currentInterval = "1d";
    let currentModel = "all";
    let chartInstance = null;
    let showSma20 = true;
    let showSma50 = true;

    // DOM Elements
    const tickerInput = document.getElementById("tickerInput");
    const searchBtn = document.getElementById("searchBtn");
    const btnText = searchBtn.querySelector(".btn-text");
    const btnSpinner = searchBtn.querySelector(".spinner");
    const modelSelect = document.getElementById("modelSelect");
    const periodButtons = document.querySelectorAll(".segment-btn");
    const presetChips = document.querySelectorAll(".preset-chip");
    const alertBox = document.getElementById("alertBox");

    // Overview Elements
    const stockNameEl = document.getElementById("stockName");
    const stockSectorEl = document.getElementById("stockSector");
    const currentPriceEl = document.getElementById("currentPrice");
    const priceChangeEl = document.getElementById("priceChange");
    const stockCurrencyEl = document.getElementById("stockCurrency");
    const stockVolumeEl = document.getElementById("stockVolume");

    // Prediction Elements
    const predictedPriceEl = document.getElementById("predictedPrice");
    const predictedChangeEl = document.getElementById("predictedChange");
    const signalBadgeEl = document.getElementById("signalBadge");
    const activeModelLabelEl = document.getElementById("activeModelLabel");
    const techConsensusEl = document.getElementById("techConsensus");
    const newsSentimentMoodEl = document.getElementById("newsSentimentMood");

    // 52-Week Elements
    const fiftyTwoLowEl = document.getElementById("fiftyTwoLow");
    const fiftyTwoHighEl = document.getElementById("fiftyTwoHigh");
    const fiftyTwoRangeFillEl = document.getElementById("fiftyTwoRangeFill");

    // Multi-Timeframe Elements
    const hourlyTrendBadge = document.getElementById("hourlyTrendBadge");
    const hourlyVolVal = document.getElementById("hourlyVolVal");
    const hourlyTableBody = document.getElementById("hourlyTableBody");
    const monthlyGrid = document.getElementById("monthlyGrid");
    const yearlyList = document.getElementById("yearlyList");

    // Grid Containers
    const modelComparisonBody = document.getElementById("modelComparisonBody");
    const featureBarsContainer = document.getElementById("featureBarsContainer");
    const technicalSignalsList = document.getElementById("technicalSignalsList");
    const newsFeedList = document.getElementById("newsFeedList");
    const posPctBadge = document.getElementById("posPctBadge");
    const neuPctBadge = document.getElementById("neuPctBadge");
    const negPctBadge = document.getElementById("negPctBadge");

    const toggleButtons = document.querySelectorAll(".toggle-btn");

    initDashboard();

    function initDashboard() {
        setupEventListeners();
        loadAllData(currentTicker, currentPeriod, currentInterval, currentModel);
    }

    function setupEventListeners() {
        searchBtn.addEventListener("click", () => {
            const val = tickerInput.value.trim().toUpperCase();
            if (val) {
                currentTicker = val;
                updatePresetHighlight(val);
                loadAllData(currentTicker, currentPeriod, currentInterval, currentModel);
            }
        });

        tickerInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") {
                searchBtn.click();
            }
        });

        presetChips.forEach((chip) => {
            chip.addEventListener("click", () => {
                const ticker = chip.getAttribute("data-ticker");
                currentTicker = ticker;
                tickerInput.value = ticker;
                updatePresetHighlight(ticker);
                loadAllData(currentTicker, currentPeriod, currentInterval, currentModel);
            });
        });

        periodButtons.forEach((btn) => {
            btn.addEventListener("click", () => {
                periodButtons.forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                currentPeriod = btn.getAttribute("data-period");
                currentInterval = btn.getAttribute("data-interval") || "1d";
                loadAllData(currentTicker, currentPeriod, currentInterval, currentModel);
            });
        });

        modelSelect.addEventListener("change", (e) => {
            currentModel = e.target.value;
            loadAllData(currentTicker, currentPeriod, currentInterval, currentModel);
        });

        toggleButtons.forEach((btn) => {
            btn.addEventListener("click", () => {
                const target = btn.getAttribute("data-dataset");
                btn.classList.toggle("active");
                if (target === "sma20") {
                    showSma20 = btn.classList.contains("active");
                } else if (target === "sma50") {
                    showSma50 = btn.classList.contains("active");
                }
                if (chartInstance) {
                    chartInstance.data.datasets.forEach((ds) => {
                        if (ds.id === "sma20") ds.hidden = !showSma20;
                        if (ds.id === "sma50") ds.hidden = !showSma50;
                    });
                    chartInstance.update();
                }
            });
        });
    }

    function updatePresetHighlight(ticker) {
        presetChips.forEach((chip) => {
            if (chip.getAttribute("data-ticker") === ticker) {
                chip.classList.add("active");
            } else {
                chip.classList.remove("active");
            }
        });
    }

    function showAlert(msg, isError = true) {
        alertBox.textContent = msg;
        alertBox.className = isError ? "alert-banner error" : "alert-banner";
        alertBox.style.display = "block";
    }

    function hideAlert() {
        alertBox.style.display = "none";
    }

    function setLoading(isLoading) {
        if (isLoading) {
            btnText.style.display = "none";
            btnSpinner.style.display = "inline-block";
            searchBtn.disabled = true;
        } else {
            btnText.style.display = "inline-block";
            btnSpinner.style.display = "none";
            searchBtn.disabled = false;
        }
    }

    function getCurrencySymbol(curr) {
        if (!curr) return "$";
        if (curr === "INR" || curr.includes("INR")) return "₹";
        if (curr === "EUR") return "€";
        if (curr === "GBP") return "£";
        return "$";
    }

    async function safeFetchJson(url, options = {}) {
        try {
            const response = await fetch(url, options);
            const text = await response.text();
            let data;
            try {
                data = JSON.parse(text);
            } catch (err) {
                if (!response.ok) {
                    return { status: "error", message: `Server error (${response.status}): ${response.statusText}` };
                }
                return { status: "error", message: "Invalid server response format." };
            }
            return data;
        } catch (e) {
            return { status: "error", message: e.message || "Failed to communicate with server." };
        }
    }

    async function loadAllData(rawTicker, period, interval, model) {
        setLoading(true);
        hideAlert();

        const ticker = String(rawTicker).replace(/\s+/g, "").toUpperCase();

        try {
            // Phase 1: Rapid metadata and chart history load
            const [infoRes, histRes] = await Promise.all([
                safeFetchJson(`/api/stock/info?ticker=${encodeURIComponent(ticker)}`),
                safeFetchJson(`/api/stock/history?ticker=${encodeURIComponent(ticker)}&period=${period}&interval=${interval}`),
            ]);

            if (infoRes.status === "error") throw new Error(infoRes.message);
            if (histRes.status === "error") throw new Error(histRes.message);

            const currSymbol = getCurrencySymbol(infoRes.data.currency);
            renderOverview(infoRes.data, currSymbol);
            renderChart(histRes.data, null, currSymbol, interval);

            // Phase 2: Sequential model prediction, sentiment & multi-timeframe analytics
            const predRes = await safeFetchJson(`/api/stock/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ticker, period, model_type: model }),
            });

            if (predRes && predRes.status === "success") {
                renderPredictionCard(predRes.data, currSymbol);
                renderChart(histRes.data, predRes.data, currSymbol, interval);
                renderComparisonTable(predRes.data, currSymbol);
                renderExplainability(predRes.data);

                if (predRes.data.technical_indicators) {
                    renderTechnicalSignals(predRes.data.technical_indicators);
                }
            }

            // Phase 3: Sentiment and timeframe data
            const [sentRes, tfRes] = await Promise.all([
                safeFetchJson(`/api/stock/sentiment?ticker=${encodeURIComponent(ticker)}`),
                safeFetchJson(`/api/stock/timeframe-analysis?ticker=${encodeURIComponent(ticker)}`),
            ]);

            if (sentRes && sentRes.status === "success") {
                renderSentimentFeed(sentRes.data);
            }

            if (tfRes && tfRes.status === "success") {
                renderTimeframeAnalysis(tfRes.data, currSymbol);
            }

        } catch (err) {
            console.error("Dashboard Load Error:", err);
            showAlert(`Failed to load data for '${ticker}': ${err.message}`);
        } finally {
            setLoading(false);
        }
    }

    function renderOverview(data, currSymbol) {
        stockNameEl.textContent = `${data.name} (${data.symbol})`;
        stockSectorEl.textContent = data.sector || "Equity";
        currentPriceEl.textContent = `${currSymbol}${data.current_price.toFixed(2)}`;
        stockCurrencyEl.textContent = data.currency;
        stockVolumeEl.textContent = Number(data.volume).toLocaleString();

        const changeSign = data.change >= 0 ? "+" : "";
        priceChangeEl.textContent = `${changeSign}${data.change.toFixed(2)} (${changeSign}${data.change_percent.toFixed(2)}%)`;
        priceChangeEl.className = data.change >= 0 ? "change-pill positive" : "change-pill negative";

        fiftyTwoLowEl.textContent = `52W L: ${currSymbol}${data.fifty_two_week_low.toFixed(2)}`;
        fiftyTwoHighEl.textContent = `52W H: ${currSymbol}${data.fifty_two_week_high.toFixed(2)}`;

        const rangeSpan = data.fifty_two_week_high - data.fifty_two_week_low;
        if (rangeSpan > 0) {
            const pct = Math.min(100, Math.max(0, ((data.current_price - data.fifty_two_week_low) / rangeSpan) * 100));
            fiftyTwoRangeFillEl.style.width = `${pct}%`;
        }
    }

    function renderPredictionCard(predData, currSymbol) {
        predictedPriceEl.textContent = `${currSymbol}${predData.next_day_predicted_price.toFixed(2)}`;
        const changeSign = predData.predicted_change >= 0 ? "+" : "";
        predictedChangeEl.textContent = `${changeSign}${predData.predicted_change.toFixed(2)} (${changeSign}${predData.predicted_change_percent.toFixed(2)}%)`;
        predictedChangeEl.className = `pred-change-pill ${predData.signal_color}`;

        signalBadgeEl.textContent = predData.signal;
        signalBadgeEl.className = `signal-badge ${predData.signal_color}`;
        activeModelLabelEl.textContent = predData.selected_model_name;

        if (predData.technical_indicators) {
            techConsensusEl.textContent = predData.technical_indicators.consensus;
            techConsensusEl.style.color = predData.technical_indicators.consensus === "BULLISH" 
                ? "var(--color-bullish)" 
                : predData.technical_indicators.consensus === "BEARISH" 
                ? "var(--color-bearish)" 
                : "var(--color-neutral)";
        }
    }

    function renderChart(history, prediction, currSymbol, interval) {
        const ctx = document.getElementById("priceChart").getContext("2d");

        if (chartInstance) {
            chartInstance.destroy();
        }

        const histDates = history.dates;
        const histPrices = history.close;
        const sma20 = history.sma_20;
        const sma50 = history.sma_50;

        const isHourly = interval === "1h";
        const allDates = [...histDates];
        const futureForecastLine = new Array(histDates.length - 1).fill(null);
        futureForecastLine.push(histPrices[histPrices.length - 1]);

        if (prediction && !isHourly) {
            if (prediction.future_dates) {
                prediction.future_dates.forEach((d) => allDates.push(d));
            }
            if (prediction.future_forecast && prediction.future_forecast.length > 0) {
                prediction.future_forecast.forEach((p) => futureForecastLine.push(p));
            } else if (prediction.next_day_predicted_price) {
                futureForecastLine.push(prediction.next_day_predicted_price);
            }
        }

        const priceGradient = ctx.createLinearGradient(0, 0, 0, 400);
        priceGradient.addColorStop(0, "rgba(59, 130, 246, 0.35)");
        priceGradient.addColorStop(1, "rgba(59, 130, 246, 0.0)");

        const datasets = [
            {
                id: "close",
                label: isHourly ? "Intraday Price (1H)" : "Historical Close Price",
                data: histPrices,
                borderColor: "#3b82f6",
                backgroundColor: priceGradient,
                fill: true,
                borderWidth: 2,
                pointRadius: isHourly ? 2 : 0,
                pointHoverRadius: 5,
                tension: 0.2,
            },
            {
                id: "sma20",
                label: "20-Period SMA",
                data: sma20,
                borderColor: "#06b6d4",
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                hidden: !showSma20,
                tension: 0.2,
            },
            {
                id: "sma50",
                label: "50-Period SMA",
                data: sma50,
                borderColor: "#f59e0b",
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                hidden: !showSma50,
                tension: 0.2,
            },
        ];

        if (prediction && !isHourly) {
            datasets.push({
                id: "forecast",
                label: "5-Day Forecast Projection",
                data: futureForecastLine,
                borderColor: "#a855f7",
                borderWidth: 2.5,
                borderDash: [5, 5],
                pointBackgroundColor: "#a855f7",
                pointBorderColor: "#ffffff",
                pointRadius: (ctx) => {
                    const idx = ctx.dataIndex;
                    if (idx === histDates.length) return 6;
                    if (idx > histDates.length) return 4;
                    return 0;
                },
                fill: false,
                tension: 0.2,
            });
        }

        chartInstance = new Chart(ctx, {
            type: "line",
            data: {
                labels: allDates,
                datasets: datasets,
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: "index",
                    intersect: false,
                },
                plugins: {
                    legend: {
                        labels: {
                            color: "#94a3b8",
                            font: { family: "Inter", size: 12 },
                            usePointStyle: true,
                        },
                    },
                    tooltip: {
                        backgroundColor: "rgba(17, 24, 39, 0.95)",
                        titleColor: "#f8fafc",
                        bodyColor: "#cbd5e1",
                        borderColor: "rgba(255, 255, 255, 0.1)",
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label: function (context) {
                                if (context.parsed.y !== null) {
                                    return `${context.dataset.label}: ${currSymbol}${context.parsed.y.toFixed(2)}`;
                                }
                                return null;
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: {
                            color: "#64748b",
                            maxTicksLimit: isHourly ? 12 : 10,
                            font: { family: "Inter", size: 11 },
                        },
                    },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.06)" },
                        ticks: {
                            color: "#64748b",
                            font: { family: "JetBrains Mono", size: 11 },
                            callback: (v) => `${currSymbol}${v}`,
                        },
                    },
                },
            },
        });
    }

    function renderTimeframeAnalysis(data, currSymbol) {
        // 1. Hourly (Bottom Level)
        const hourly = data.hourly || {};
        hourlyTrendBadge.textContent = `Trend: ${hourly.trend || "NEUTRAL"}`;
        hourlyTrendBadge.className = hourly.trend === "BULLISH" ? "tf-badge positive" : hourly.trend === "BEARISH" ? "tf-badge negative" : "tf-badge neutral";
        hourlyVolVal.textContent = `${hourly.volatility_pct || 0.0}%`;

        hourlyTableBody.innerHTML = "";
        const bars = hourly.recent_bars || [];
        if (bars.length === 0) {
            hourlyTableBody.innerHTML = `<tr><td colspan="3" class="text-center">No hourly data available.</td></tr>`;
        } else {
            bars.slice(-6).reverse().forEach((b) => {
                const tr = document.createElement("tr");
                const chgClass = b.change_pct >= 0 ? "positive" : "negative";
                const sign = b.change_pct >= 0 ? "+" : "";
                tr.innerHTML = `
                    <td>${b.timestamp}</td>
                    <td>${currSymbol}${b.close.toFixed(2)}</td>
                    <td class="${chgClass}">${sign}${b.change_pct.toFixed(2)}%</td>
                `;
                hourlyTableBody.appendChild(tr);
            });
        }

        // 2. Monthly (Month-to-Month)
        monthlyGrid.innerHTML = "";
        const months = data.monthly || [];
        if (months.length === 0) {
            monthlyGrid.innerHTML = `<div class="placeholder-text">No monthly return history.</div>`;
        } else {
            months.slice(-9).forEach((m) => {
                const div = document.createElement("div");
                div.className = "month-pill";
                const chgClass = m.return_pct >= 0 ? "positive" : "negative";
                const sign = m.return_pct >= 0 ? "+" : "";
                div.innerHTML = `
                    <span class="month-name">${m.period}</span>
                    <span class="month-ret ${chgClass}">${sign}${m.return_pct.toFixed(1)}%</span>
                `;
                monthlyGrid.appendChild(div);
            });
        }

        // 3. Yearly (Year-to-Year)
        yearlyList.innerHTML = "";
        const years = data.yearly || [];
        if (years.length === 0) {
            yearlyList.innerHTML = `<div class="placeholder-text">No multi-year history.</div>`;
        } else {
            years.slice(-5).forEach((y) => {
                const div = document.createElement("div");
                div.className = "yearly-item";
                const chgClass = y.return_pct >= 0 ? "positive" : "negative";
                const sign = y.return_pct >= 0 ? "+" : "";
                const fillWidth = Math.min(100, Math.max(10, Math.abs(y.return_pct)));
                const fillColor = y.return_pct >= 0 ? "var(--color-bullish)" : "var(--color-bearish)";

                div.innerHTML = `
                    <div class="yoy-top">
                        <span>${y.year} (H: ${currSymbol}${y.high})</span>
                        <span class="${chgClass}">${sign}${y.return_pct.toFixed(2)}%</span>
                    </div>
                    <div class="yoy-bar-track">
                        <div class="yoy-bar-fill" style="width: ${fillWidth}%; background-color: ${fillColor};"></div>
                    </div>
                `;
                yearlyList.appendChild(div);
            });
        }
    }

    function renderComparisonTable(predData, currSymbol) {
        modelComparisonBody.innerHTML = "";

        if (!predData.comparison_table || predData.comparison_table.length === 0) {
            modelComparisonBody.innerHTML = `<tr><td colspan="6" class="text-center">No comparison data available.</td></tr>`;
            return;
        }

        predData.comparison_table.forEach((row) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${row.model}</strong></td>
                <td>${row.rmse.toFixed(2)}</td>
                <td>${row.mae.toFixed(2)}</td>
                <td>${row.mape.toFixed(2)}%</td>
                <td><span class="positive">${row.directional_accuracy.toFixed(1)}%</span></td>
                <td><strong>${currSymbol}${row.next_day_pred.toFixed(2)}</strong></td>
            `;
            modelComparisonBody.appendChild(tr);
        });
    }

    function renderExplainability(predData) {
        featureBarsContainer.innerHTML = "";

        let importances = [];
        if (predData.models && predData.models.rf && predData.models.rf.feature_importances) {
            importances = predData.models.rf.feature_importances;
        } else if (predData.models && predData.models.linear && predData.models.linear.coefficients) {
            importances = predData.models.linear.coefficients.map((c) => ({
                feature: c.feature,
                importance: Math.min(100, Math.abs(c.weight * 10)),
            }));
        }

        if (importances.length === 0) {
            featureBarsContainer.innerHTML = `<div class="placeholder-text">Feature importance metrics available after running Random Forest.</div>`;
            return;
        }

        importances.slice(0, 6).forEach((item) => {
            const row = document.createElement("div");
            row.className = "feat-bar-row";
            row.innerHTML = `
                <div class="feat-bar-header">
                    <span class="feat-name">${formatFeatureName(item.feature)}</span>
                    <span class="feat-val">${item.importance.toFixed(1)}%</span>
                </div>
                <div class="feat-track">
                    <div class="feat-fill" style="width: ${Math.min(100, item.importance * 2.5)}%;"></div>
                </div>
            `;
            featureBarsContainer.appendChild(row);
        });
    }

    function formatFeatureName(name) {
        const map = {
            Close: "Current Close Price",
            Close_Lag_1: "1-Day Lag Price",
            Close_Lag_2: "2-Day Lag Price",
            Close_Lag_3: "3-Day Lag Price",
            SMA_20: "20-Day Moving Avg",
            SMA_50: "50-Day Moving Avg",
            EMA_20: "20-Day Exp Moving Avg",
            RSI_14: "RSI Momentum (14D)",
            MACD: "MACD Trend",
            MACD_Signal: "MACD Signal Line",
            Volatility_20: "20-Day Volatility",
            Volume: "Trading Volume",
            Daily_Return: "Daily Return %",
        };
        return map[name] || name;
    }

    function renderTechnicalSignals(techData) {
        technicalSignalsList.innerHTML = "";

        if (!techData.signals || techData.signals.length === 0) {
            technicalSignalsList.innerHTML = `<div class="placeholder-text">No technical signals available.</div>`;
            return;
        }

        techData.signals.forEach((sig) => {
            const card = document.createElement("div");
            card.className = `signal-card ${sig.type}`;
            card.innerHTML = `
                <div class="signal-top">
                    <span>${sig.indicator}</span>
                    <span class="${sig.type}">${sig.status}</span>
                </div>
                <div class="signal-exp">${sig.explanation}</div>
            `;
            technicalSignalsList.appendChild(card);
        });
    }

    function renderSentimentFeed(sentData) {
        newsFeedList.innerHTML = "";

        posPctBadge.textContent = `Bullish: ${sentData.breakdown_pct.positive}%`;
        neuPctBadge.textContent = `Neutral: ${sentData.breakdown_pct.neutral}%`;
        negPctBadge.textContent = `Bearish: ${sentData.breakdown_pct.negative}%`;

        newsSentimentMoodEl.textContent = `${sentData.overall_sentiment} (${sentData.sentiment_score > 0 ? "+" : ""}${sentData.sentiment_score})`;
        newsSentimentMoodEl.style.color = sentData.overall_sentiment === "BULLISH" 
            ? "var(--color-bullish)" 
            : sentData.overall_sentiment === "BEARISH" 
            ? "var(--color-bearish)" 
            : "var(--color-neutral)";

        if (!sentData.articles || sentData.articles.length === 0) {
            newsFeedList.innerHTML = `<div class="placeholder-text">No recent news articles found for this ticker.</div>`;
            return;
        }

        sentData.articles.forEach((art) => {
            const a = document.createElement("a");
            a.className = "news-item";
            a.href = art.link !== "#" ? art.link : "javascript:void(0)";
            a.target = art.link !== "#" ? "_blank" : "_self";
            a.innerHTML = `
                <div class="news-title">${escapeHtml(art.title)}</div>
                <div class="news-footer">
                    <span>${escapeHtml(art.publisher)} • ${art.date}</span>
                    <span class="${art.tag_class}">${art.label} (${art.score > 0 ? "+" : ""}${art.score.toFixed(2)})</span>
                </div>
            `;
            newsFeedList.appendChild(a);
        });
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
