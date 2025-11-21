const API_URL = "https://road-sign-recognition.onrender.com"; // Replace with Render API URL

async function predict() {
    const input = document.getElementById("imageInput").files[0];
    if (!input) return alert("Please upload an image");

    let formData = new FormData();
    formData.append("file", input);

    let res = await fetch(API_URL, {
        method: "POST",
        body: formData
    });

    let data = await res.json();

    document.getElementById("result").innerHTML =
        `Predicted Class: ${data.class} <br> Confidence: ${data.confidence}`;
}