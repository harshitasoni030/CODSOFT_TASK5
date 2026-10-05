const form = document.getElementById("generate-form");
const seedInput = document.getElementById("seed");
const lengthInput = document.getElementById("length");
const tempInput = document.getElementById("temperature");
const tempValue = document.getElementById("temp-value");
const button = document.getElementById("generate-btn");
const output = document.getElementById("output");

tempInput.addEventListener("input", () => {
  tempValue.textContent = tempInput.value;
});

function show(text, className) {
  output.textContent = text;
  output.className = "output " + (className || "");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    seed_text: seedInput.value,
    length: parseInt(lengthInput.value, 10),
    temperature: parseFloat(tempInput.value),
  };

  if (!payload.length || payload.length < 1 || payload.length > 1000) {
    show("Enter a length between 1 and 1000.", "error");
    return;
  }

  button.disabled = true;
  button.textContent = "Generating...";
  show("Generating...", "empty");

  try {
    const response = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      show(data.error || "Something went wrong.", "error");
    } else {
      show(data.generated_text);
    }
  } catch (err) {
    show("Could not reach the server. Is Flask running?", "error");
  } finally {
    button.disabled = false;
    button.textContent = "Generate text";
  }
});
