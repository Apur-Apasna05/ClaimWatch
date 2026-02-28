(function () {
  "use strict";

  var insuranceForm = document.getElementById("insurance-form");
  var insuranceSubmitBtn = document.getElementById("insurance-submit");
  var insuranceCsvForm = document.getElementById("insurance-csv-form");
  var insuranceCsvSubmitBtn = document.getElementById("insurance-csv-submit");
  var jobForm = document.getElementById("job-form");
  var jobSubmitBtn = document.getElementById("job-submit");
  var errorEl = document.getElementById("error-message");
  var resultsSection = document.getElementById("results-section");
  var feedbackCard = document.getElementById("feedback-card");
  var feedbackYes = document.getElementById("feedback-yes");
  var feedbackNo = document.getElementById("feedback-no");
  var feedbackNote = document.getElementById("feedback-note");
  var lastRequestPayload = null;
  var lastPrediction = null;

  // Chart.js instances
  var fraudGaugeChart = null;
  var anomalyBarChart = null;
  var shapBarChart = null;

  function showError(message) {
    errorEl.textContent = message;
    errorEl.hidden = false;
    resultsSection.hidden = true;
  }

  function hideError() {
    errorEl.hidden = true;
  }

  function setLoading(btn, loading, idleText) {
    btn.disabled = loading;
    btn.textContent = loading ? "Evaluating…" : idleText;
  }

  function safeNum(x) {
    var n = Number(x);
    return (typeof n === "number" && !Number.isNaN(n)) ? n : 0;
  }

  function formatFeatureName(name) {
    return name
      .split("_")
      .map(function (w) {
        return w.charAt(0).toUpperCase() + w.slice(1);
      })
      .join(" ");
  }

  function renderResults(data) {
    hideError();
    resultsSection.hidden = false;
    feedbackCard.hidden = false;

    lastPrediction = data;

    var prob = safeNum(data.fraud_probability);
    var fused = data.fused_risk != null && !Number.isNaN(Number(data.fused_risk))
      ? Number(data.fused_risk) : null;
    var primaryRisk = fused != null ? fused : prob;
    var probPct = (primaryRisk * 100).toFixed(1);
    document.getElementById("fraud-probability").textContent = probPct + "%";

    var trust = data.trust_score != null && !Number.isNaN(Number(data.trust_score))
      ? (Number(data.trust_score) * 100).toFixed(1) + "%" : "—";
    var trustEl = document.getElementById("trust-score");
    if (trustEl) trustEl.textContent = trust;

    var riskBadge = document.getElementById("risk-level-badge");
    if (riskBadge) {
      riskBadge.classList.remove("risk-low", "risk-medium", "risk-high", "risk-unknown");
      if (primaryRisk < 0.3) {
        riskBadge.textContent = "Low risk";
        riskBadge.classList.add("risk-low");
      } else if (primaryRisk < 0.7) {
        riskBadge.textContent = "Medium risk";
        riskBadge.classList.add("risk-medium");
      } else {
        riskBadge.textContent = "High risk";
        riskBadge.classList.add("risk-high");
      }
    }

    // Anomaly score is always 0-10 for both insurance and job fraud (aligned with fraud probability).
    var anomaly = (data.anomaly_score != null && !Number.isNaN(Number(data.anomaly_score)))
      ? Number(data.anomaly_score) : 0;
    document.getElementById("anomaly-score").textContent =
      anomaly.toFixed(1) + " / 10";
    var anomalousLabel =
      data.is_anomalous === true ? "Yes" : data.is_anomalous === false ? "No" : "—";
    document.getElementById("is-anomalous").textContent = anomalousLabel;

    // Update charts (use fused risk when present for consistency)
    renderCharts(data, primaryRisk, anomaly, anomalousLabel);

    // Fraud persona badge
    var personaEl = document.getElementById("fraud-persona");
    if (personaEl) {
      var persona = data.fraud_persona || "Unknown";
      personaEl.textContent = persona;

      personaEl.classList.remove("persona-low", "persona-medium", "persona-high", "persona-neutral");

      var lower = persona.toLowerCase();
      if (lower.indexOf("low risk") !== -1 || lower.indexOf("normal") !== -1 || lower.indexOf("normal posting") !== -1) {
        personaEl.classList.add("persona-low");
      } else if (lower.indexOf("high risk") !== -1 || lower.indexOf("opportunistic") !== -1 || lower.indexOf("policy") !== -1 || lower.indexOf("likely fake") !== -1) {
        personaEl.classList.add("persona-high");
      } else if (lower.indexOf("repeat") !== -1 || lower.indexOf("financial") !== -1) {
        personaEl.classList.add("persona-high");
      } else if (lower.indexOf("medium risk") !== -1 || lower.indexOf("needs review") !== -1 || lower.indexOf("needs analyst review") !== -1) {
        personaEl.classList.add("persona-medium");
      } else {
        personaEl.classList.add("persona-neutral");
      }
    }

    // Important keywords (job fraud)
    var keywordsCard = document.getElementById("keywords-card");
    var keywordsList = document.getElementById("keywords-list");
    keywordsList.innerHTML = "";
    if (data.fraud_type === "job_fraud" && (data.important_keywords || []).length > 0) {
      (data.important_keywords || []).forEach(function (kw) {
        var li = document.createElement("li");
        li.textContent = (kw.keyword || "") + " (" + safeNum(kw.score).toFixed(3) + ")";
        keywordsList.appendChild(li);
      });
      keywordsCard.hidden = false;
    } else {
      keywordsCard.hidden = true;
    }

    var tbody = document.querySelector("#features-table tbody");
    tbody.innerHTML = "";
    var rowsSource = (data.top_features || []).slice();

    // For job fraud, fall back to important_keywords for explanation table
    if (data.fraud_type === "job_fraud" && rowsSource.length === 0) {
      rowsSource = (data.important_keywords || []).map(function (kw) {
        return {
          feature: kw.keyword,
          value: 1,
          shap_value: kw.score,
        };
      });
    }

    rowsSource.forEach(function (f) {
      var tr = document.createElement("tr");
      var sv = safeNum(f.shap_value);
      var shapClass = sv > 0 ? "shap-positive" : "shap-negative";
      tr.innerHTML =
        "<td>" +
        formatFeatureName(String(f.feature || "")) +
        "</td><td>" +
        safeNum(f.value).toFixed(2) +
        "</td><td class=\"" +
        shapClass +
        "\">" +
        sv.toFixed(4) +
        "</td>";
      tbody.appendChild(tr);
    });

    document.getElementById("summary-text").textContent = data.summary || "";

    var actionsList = document.getElementById("actions-list");
    actionsList.innerHTML = "";
    (data.recommended_actions || []).forEach(function (action) {
      var li = document.createElement("li");
      li.textContent = action;
      actionsList.appendChild(li);
    });
  }

  // Insurance form submit
  insuranceForm.addEventListener("submit", function (e) {
    e.preventDefault();

    var payload = {
      fraud_type: "insurance",
      claim_amount: Number(insuranceForm.claim_amount.value) || 0,
      policy_tenure_days: Number(insuranceForm.policy_tenure_days.value) || 0,
      num_prior_claims: Number(insuranceForm.num_prior_claims.value) || 0,
      customer_age: Number(insuranceForm.customer_age.value) || 0,
    };

    lastRequestPayload = payload;

    setLoading(insuranceSubmitBtn, true, "Evaluate insurance claim");

    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.text().then(function (t) {
            throw new Error("API error " + res.status + ": " + (t || res.statusText));
          });
        }
        return res.json();
      })
      .then(function (data) {
        renderResults(data);
      })
      .catch(function (err) {
        showError(err.message || "Request failed. Is the backend running on port 8000?");
      })
      .finally(function () {
        setLoading(insuranceSubmitBtn, false, "Evaluate insurance claim");
      });
  });

  // Job form submit (file or text)
  jobForm.addEventListener("submit", function (e) {
    e.preventDefault();

    var fileInput = document.getElementById("job_file");
    var file = fileInput.files[0] || null;
    var textValue = (document.getElementById("job_text").value || "").trim();

    // Prefer file when present
    if (file) {
      var formData = new FormData();
      formData.append("fraud_type", "job_fraud");
      formData.append("file", file);

      lastRequestPayload = { fraud_type: "job_fraud", file_name: file.name };
      setLoading(jobSubmitBtn, true, "Evaluate job posting");

      fetch("/predict-from-file", {
        method: "POST",
        body: formData,
      })
        .then(function (res) {
          if (!res.ok) {
            return res.text().then(function (t) {
              throw new Error("API error " + res.status + ": " + (t || res.statusText));
            });
          }
          return res.json();
        })
        .then(function (data) {
          // FileUploadResponse has .prediction inside
          if (data && data.prediction) {
            renderResults(data.prediction);
          } else {
            showError("Unexpected response format from file-based prediction.");
          }
        })
        .catch(function (err) {
          showError(err.message || "File-based request failed.");
        })
        .finally(function () {
          setLoading(jobSubmitBtn, false, "Evaluate job posting");
        });
      return;
    }

    // Fallback: text-based job fraud prediction
    var payload = {
      fraud_type: "job_fraud",
      job_text: textValue,
    };
    lastRequestPayload = payload;

    setLoading(jobSubmitBtn, true, "Evaluate job posting");

    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.text().then(function (t) {
            throw new Error("API error " + res.status + ": " + (t || res.statusText));
          });
        }
        return res.json();
      })
      .then(function (data) {
        renderResults(data);
      })
      .catch(function (err) {
        showError(err.message || "Request failed. Is the backend running on port 8000?");
      })
      .finally(function () {
        setLoading(jobSubmitBtn, false, "Evaluate job posting");
      });
  });

  // Insurance CSV bulk submit
  insuranceCsvForm.addEventListener("submit", function (e) {
    e.preventDefault();

    var fileInput = document.getElementById("insurance_csv_file");
    var file = fileInput.files[0] || null;
    if (!file) {
      showError("Please select a CSV file first.");
      return;
    }

    var formData = new FormData();
    formData.append("file", file);

    lastRequestPayload = { bulk_csv_file: file.name };
    setLoading(insuranceCsvSubmitBtn, true, "Upload & evaluate CSV");

    fetch("/predict-from-csv", {
      method: "POST",
      body: formData,
    })
      .then(function (res) {
        if (!res.ok) {
          return res.text().then(function (t) {
            throw new Error("API error " + res.status + ": " + (t || res.statusText));
          });
        }
        return res.json();
      })
      .then(function (items) {
        if (!Array.isArray(items) || items.length === 0) {
          showError("No predictions returned for CSV.");
          return;
        }
        // For now, show the last prediction in the standard panel.
        renderResults(items[items.length - 1]);
      })
      .catch(function (err) {
        showError(err.message || "CSV upload request failed.");
      })
      .finally(function () {
        setLoading(insuranceCsvSubmitBtn, false, "Upload & evaluate CSV");
      });
  });

  function renderCharts(data, prob, anomaly, anomalousLabel) {
    var gaugeCanvas = document.getElementById("fraudGauge");
    var anomalyCanvas = document.getElementById("anomalyBar");
    var shapCanvas = document.getElementById("shapBar");

    if (typeof Chart === "undefined") {
      return;
    }

    // Destroy existing charts
    if (fraudGaugeChart) fraudGaugeChart.destroy();
    if (anomalyBarChart) anomalyBarChart.destroy();
    if (shapBarChart) shapBarChart.destroy();

    // Fraud gauge (doughnut)
    if (gaugeCanvas) {
      var riskColor;
      if (prob < 0.3) riskColor = "#22c55e";
      else if (prob < 0.7) riskColor = "#eab308";
      else riskColor = "#f97373";

      fraudGaugeChart = new Chart(gaugeCanvas.getContext("2d"), {
        type: "doughnut",
        data: {
          labels: ["Fraud risk", "Safe"],
          datasets: [
            {
              data: [prob, Math.max(0, 1 - prob)],
              backgroundColor: [riskColor, "rgba(31,41,55,0.8)"],
              borderWidth: 0,
            },
          ],
        },
        options: {
          cutout: "70%",
          plugins: {
            legend: { display: false },
          },
          animation: {
            animateRotate: true,
            duration: 700,
          },
        },
      });
    }

    // Anomaly bar (only meaningful for insurance; support negative scores)
    if (anomalyCanvas) {
      var anomalyNum = safeNum(anomaly);
      var anomalyColor =
        anomalousLabel === "Yes" ? "#f97373" : anomalousLabel === "No" ? "#22c55e" : "#4b5563";

      anomalyBarChart = new Chart(anomalyCanvas.getContext("2d"), {
        type: "bar",
        data: {
          labels: ["Anomaly score"],
          datasets: [
            {
              label: "Anomaly score",
              data: [anomalyNum],
              backgroundColor: anomalyColor,
            },
          ],
        },
        options: {
          indexAxis: "y",
          plugins: { legend: { display: false } },
          scales: {
            x: {
              min: 0,
              max: 10,
              ticks: { color: "#9ca3af" },
              grid: { color: "rgba(55,65,81,0.4)" },
            },
            y: {
              ticks: { color: "#9ca3af" },
              grid: { display: false },
            },
          },
          animation: { duration: 600 },
        },
      });
    }

    // SHAP bar chart
    if (shapCanvas && (data.top_features || []).length > 0) {
      var sorted = (data.top_features || [])
        .slice()
        .sort(function (a, b) {
          return Math.abs(b.shap_value) - Math.abs(a.shap_value);
        })
        .slice(0, 5);

      var labels = sorted.map(function (f) {
        return formatFeatureName(f.feature);
      });
      var values = sorted.map(function (f) {
        return safeNum(f.shap_value);
      });
      var colors = sorted.map(function (f) {
        return f.shap_value > 0 ? "#f97373" : "#22c55e";
      });

      shapBarChart = new Chart(shapCanvas.getContext("2d"), {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "SHAP impact",
              data: values,
              backgroundColor: colors,
            },
          ],
        },
        options: {
          indexAxis: "y",
          plugins: { legend: { display: false } },
          scales: {
            x: {
              ticks: { color: "#9ca3af" },
              grid: { color: "rgba(55,65,81,0.4)" },
            },
            y: {
              ticks: { color: "#9ca3af" },
              grid: { display: false },
            },
          },
          animation: { duration: 700 },
        },
      });
    }
  }

  function sendFeedback(answer) {
    if (!lastPrediction || !lastRequestPayload) {
      return;
    }
    var prob = lastPrediction.fraud_probability || 0;
    var label = prob >= 0.5 ? "fraud" : "legit";
    var body = {
      fraud_type: lastPrediction.fraud_type || "insurance",
      input_payload: lastRequestPayload,
      predicted_label: label,
      predicted_probability: prob,
      user_feedback: answer,
    };

    fetch("/feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    })
      .then(function () {
        if (feedbackNote) {
          feedbackNote.hidden = false;
        }
        if (feedbackYes) feedbackYes.disabled = true;
        if (feedbackNo) feedbackNo.disabled = true;
      })
      .catch(function () {
        // Feedback failures are non-fatal for users
      });
  }

  if (feedbackYes && feedbackNo) {
    feedbackYes.addEventListener("click", function () {
      sendFeedback("yes");
    });
    feedbackNo.addEventListener("click", function () {
      sendFeedback("no");
    });
  }

  // Nothing to initialise – both sections are visible independently.
})();
