const chatWindow = document.getElementById('chatWindow');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const voiceBtn = document.getElementById('voiceBtn');

function addMessage(text, type = 'user') {
  const wrap = document.createElement('div');
  wrap.className = `msg ${type}`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  const time = document.createElement('div');
  time.className = 'time';
  const now = new Date();
  time.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  wrap.appendChild(bubble);
  wrap.appendChild(time);
  chatWindow.appendChild(wrap);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage() {
  const message = chatInput.value.trim();
  if (!message) return;
  addMessage(message, 'user');
  chatInput.value = '';

  const typing = document.createElement('div');
  typing.className = 'msg bot';
  typing.innerHTML = '<div class="bubble">Typing...</div>';
  chatWindow.appendChild(typing);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });
    const data = await res.json();
    typing.remove();
    addMessage(data.reply || 'Sorry, I could not understand that.', 'bot');
  } catch (err) {
    typing.remove();
    addMessage('Network error. Please try again.', 'bot');
  }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendMessage();
});

let recognition;
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.lang = 'en-IN';
  recognition.interimResults = false;

  recognition.onresult = (event) => {
    const text = event.results[0][0].transcript;
    chatInput.value = text;
    sendMessage();
  };

  recognition.onstart = () => {
    voiceBtn.textContent = 'Listening...';
    voiceBtn.classList.add('active');
  };

  recognition.onend = () => {
    voiceBtn.textContent = 'Start Voice';
    voiceBtn.classList.remove('active');
  };

  voiceBtn.addEventListener('click', () => {
    recognition.start();
  });
} else {
  voiceBtn.disabled = true;
  voiceBtn.textContent = 'Voice Not Supported';
}

addMessage('Hi! I am SVP Assist. Ask me about admissions, courses, fees, timetable, placements or college info.', 'bot');
