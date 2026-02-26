(function () {
  "use strict";

  var form = document.getElementById("claim-form");
  var submitBtn = document.getElementById("submit-btn");
  var errorEl = document.getElementById("error-message");
  var resultsSection = document.getElementById("results-section");

  function showError(message) {
    errorEl.textContent = message;
    errorEl.hidden = false;
    resultsSection.hidden = true;
  }

  function hideError() {
    errorEl.hidden = true;
  }

  function setLoading(loading) {
    submitBtn.disabled = loading;
    submitBtn.textContent = loading ? "Evaluating…" : "Evaluate claim";
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

    var prob = data.fraud_probability;
    var probPct = (prob * 100).toFixed(1);
    document.getElementById("fraud-probability").textContent = probPct + "%";

    document.getElementById("anomaly-score").textContent = data.anomaly_score.toFixed(3);
    document.getElementById("is-anomalous").textContent = data.is_anomalous ? "Yes" : "No";

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

  form.addEventListener("submit", function (e) {
    e.preventDefault();

    var payload = {
      claim_amount: Number(form.claim_amount.value) || 0,
      policy_tenure_days: Number(form.policy_tenure_days.value) || 0,
      num_prior_claims: Number(form.num_prior_claims.value) || 0,
      customer_age: Number(form.customer_age.value) || 0,
    };

    setLoading(true);

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
        setLoading(false);
      });
  });
})();
