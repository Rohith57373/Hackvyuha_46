// SecureView Chatbot Widget
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the landing page - don't show chatbot on landing page
    const isLandingPage = document.body.classList.contains('landing-page');
    
    if (isLandingPage) {
        return; // Don't initialize chatbot on landing page
    }

    // Create chatbot elements
    const chatWidget = document.createElement('div');
    chatWidget.className = 'chat-widget';
    chatWidget.innerHTML = `
        <button class="chat-button" aria-label="Open chat">
            <i class="ri-chat-1-fill"></i>
            <div class="chat-notification">1</div>
        </button>
        <div class="chat-popup">
            <div class="chat-header">
                <h3>SecureView Assistant</h3>
                <button class="chat-close" aria-label="Close chat">
                    <i class="ri-close-line"></i>
                </button>
            </div>
            <div class="chat-messages">
                <div class="welcome-message">
                    Hi there! I'm your SecureView assistant. How can I help you today?
                </div>
                <div class="message message-bot">
                    Hello! I can help with questions about SecureView features, camera setup, or AI detection settings. What would you like to know?
                </div>
            </div>
            <div class="chat-input">
                <input type="text" placeholder="Type your message..." />
                <button class="chat-send" aria-label="Send message">
                    <i class="ri-send-plane-fill"></i>
                </button>
            </div>
        </div>
    `;
    
    // Add to page
    document.body.appendChild(chatWidget);
    
    // Select elements
    const chatButton = chatWidget.querySelector('.chat-button');
    const chatPopup = chatWidget.querySelector('.chat-popup');
    const closeButton = chatWidget.querySelector('.chat-close');
    const chatMessages = chatWidget.querySelector('.chat-messages');
    const chatInput = chatWidget.querySelector('input');
    const sendButton = chatWidget.querySelector('.chat-send');
    const notification = chatWidget.querySelector('.chat-notification');
    
    // Show notification on first load
    setTimeout(() => {
        notification.classList.add('active');
    }, 2000);
    
    // Toggle chat visibility
    chatButton.addEventListener('click', () => {
        chatPopup.classList.toggle('active');
        notification.classList.remove('active');
    });
    
    // Close chat
    closeButton.addEventListener('click', () => {
        chatPopup.classList.remove('active');
    });
    
    // Send message when clicking send button
    sendButton.addEventListener('click', sendMessage);
    
    // Send message on Enter key
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Function to send user message and get bot response
    function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage(message, 'user');
        
        // Clear input
        chatInput.value = '';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Call API to get bot response
        fetch('/api/chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Add bot response
            if (data.success) {
                addMessage(data.response, 'bot');
            } else {
                addMessage("Sorry, I'm having trouble responding right now. Please try again.", 'bot');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            removeTypingIndicator();
            addMessage("Sorry, there was a problem connecting to the assistant. Please try again later.", 'bot');
        });
    }
    
    // Function to add a message to the chat
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${sender === 'user' ? 'user' : 'bot'}`;
        messageDiv.textContent = text;
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to show typing indicator
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        typingDiv.id = 'typing-indicator';
        
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to remove typing indicator
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
}); 