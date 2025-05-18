import google.generativeai as genai

class GeminiChatbot:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """
        Initialize chatbot with API key and model.
        
        Args:
            api_key: Google AI Studio API key
            model: Model name (default: gemini-2.0-flash)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.chat = self.model.start_chat()
        self._setup_system_instruction()

    def _setup_system_instruction(self):
        """Set default chatbot behavior"""
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction="You're a helpful assistant specialized in SecureView CCTV systems. Keep responses concise and factual when answering questions about security cameras, AI detection, and surveillance features."
        )
        self.chat = self.model.start_chat()

    def stream_response(self, prompt: str):
        """Stream response chunks with error handling"""
        try:
            response = self.chat.send_message(prompt, stream=True)
            for chunk in response:
                yield chunk.text
        except Exception as e:
            yield f"Error: {str(e)}"
            
    def get_response(self, prompt: str):
        """Get full response as a string (for web interface)"""
        try:
            response = self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    def run_chat(self):
        """Main chat interface"""
        print("Gemini Chatbot: Hello! Type 'exit' to end chat.\n")
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ['exit', 'quit']:
                    print("\nChat ended.")
                    break
                
                print("Assistant: ", end="", flush=True)
                for chunk in self.stream_response(user_input):
                    print(chunk, end="", flush=True)
                print("\n")

            except KeyboardInterrupt:
                print("\nChat ended by user.")
                break

if __name__ == "__main__":
    API_KEY = "AIzaSyCkOeDL6LCJYOm_gogxL8cq_TTQCfLe3wE"
    
    # Initialize with your preferred model
    chatbot = GeminiChatbot(api_key=API_KEY)
    chatbot.run_chat() 