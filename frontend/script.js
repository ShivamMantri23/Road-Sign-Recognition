const API_URL = "https://your-render-api-url.onrender.com/predict";

async function predict() {
    const input = document.getElementById("imageInput").files[0];
    if (!input) return alert("Upload an image!");

    let formData = new FormData();
    formData.append("file", input);

    let response = await fetch(API_URL, {
        method: "POST",
        body: formData
    });

    let data = await response.json();
    document.getElementById("result").innerHTML =
        `Predicted Class: ${data.class} <br> Confidence: ${data.confidence}`;
}