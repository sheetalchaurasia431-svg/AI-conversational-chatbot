async function sendMessage() {

    const input = document.getElementById("message");
    const message = input.value.trim();

    if (!message) {
        return;
    }

    const chat = document.getElementById("chat");

    // Display user message
    chat.innerHTML += `
        <div class="message user">
            👤 ${message}
        </div>
    `;

    input.value = "";

    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: message,
                domain: "Artificial Intelligence",
                topic: "LangChain"
            })

        });

        const data = await response.json();

        // Display AI response
        chat.innerHTML += `
            <div class="message ai">
                🤖 ${data.response}
            </div>
        `;

    } catch (error) {

        chat.innerHTML += `
            <div class="message ai">
                ❌ Unable to connect to the AI server.
            </div>
        `;

        console.error(error);
    }
}