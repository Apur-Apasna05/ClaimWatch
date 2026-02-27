(function () {
  "use strict";

  var insuranceForm = document.getElementById("insurance-form");
  var insuranceSubmitBtn = document.getElementById("insurance-submit");
  var jobForm = document.getElementById("job-form");
  var jobSubmitBtn = document.getElementById("job-submit");
  var errorEl = document.getElementById("error-message");
  var resultsSection = document.getElementById("results-section");
  var feedbackCard = document.getElementById("feedback-card");
  var feedbackYes = document.getElementById("feedback-yes");
  var feedbackNo = document.getElementById("feedback-no");
  var lastRequestPayload = null;
  var lastPrediction = null;

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

    var prob = data.fraud_probability;
    var probPct = (prob * 100).toFixed(1);
    document.getElementById("fraud-probability").textContent = probPct + "%";

    document.getElementById("anomaly-score").textContent = data.anomaly_score.toFixed(3);
    document.getElementById("is-anomalous").textContent = data.is_anomalous ? "Yes" : "No";

    // Fraud persona badge
    var personaEl = document.getElementById("fraud-persona");
    if (personaEl) {
      var persona = data.fraud_persona || "Unknown";
      personaEl.textContent = persona;

      personaEl.classList.remove("persona-low", "persona-medium", "persona-high", "persona-neutral");

      var lower = persona.toLowerCase();
      if (lower.indexOf("low risk") !== -1 || lower.indexOf("normal") !== -1) {
        personaEl.classList.add("persona-low");
      } else if (lower.indexOf("opportunistic") !== -1 || lower.indexOf("policy") !== -1) {
        personaEl.classList.add("persona-high");
      } else if (lower.indexOf("repeat") !== -1 || lower.indexOf("financial") !== -1) {
        personaEl.classList.add("persona-high");
      } else if (lower.indexOf("needs analyst review") !== -1) {
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
        li.textContent = kw.keyword + " (" + kw.score.toFixed(3) + ")";
        keywordsList.appendChild(li);
      });
      keywordsCard.hidden = false;
    } else {
      keywordsCard.hidden = true;
    }

    var tbody = document.querySelector("#features-table tbody");
    tbody.innerHTML = "";
    (data.top_features || []).forEach(function (f) {
      var tr = document.createElement("tr");
      var shapClass = f.shap_value > 0 ? "shap-positive" : "shap-negative";
      tr.innerHTML =
        "<td>" +
        formatFeatureName(f.feature) +
        "</td><td>" +
        Number(f.value).toFixed(2) +
        "</td><td class=\"" +
        shapClass +
        "\">" +
        f.shap_value.toFixed(4) +
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
    }).catch(function () {
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
