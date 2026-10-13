const form = document.querySelector("form");
const button = document.querySelector("button");
const loading = document.querySelector(".loading");

form.addEventListener("submit", function () {
    button.disabled = true;
    button.innerText = "Analyzing...";

    if (loading) {
        loading.style.display = "block";
    }
});