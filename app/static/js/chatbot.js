const chatBox = document.getElementById('chat-box');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const languageSelect = document.getElementById('language-select');
const backToChatButton = document.getElementById('back-to-chat');

function appendMessage(sender, message, source, confidence) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.textContent = message;
    if (source) {
        const sourceSpan = document.createElement('span');
        sourceSpan.style.fontSize = '0.8em';
        sourceSpan.textContent = ` (${source}, Confidence: ${confidence})`;
        messageDiv.appendChild(sourceSpan);
    }
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    appendMessage('user', message, null, null);
    chatInput.value = '';

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                language: languageSelect.value,
                session_id: 'user_session'
            })
        });
        const result = await response.json();
        appendMessage('bot', result.response, result.source, result.confidence);
    } catch (error) {
        appendMessage('bot', 'Error communicating with server.', 'Server', 1.0);
    }
}

sendButton.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

backToChatButton.addEventListener('click', () => {
    window.location.href = '/dashboard';
});

if (window.location.search.includes('from=admin')) {
    backToChatButton.style.display = 'block';
}