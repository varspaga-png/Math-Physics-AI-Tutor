import streamsync as ss
import requests
import os

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


class TutorUI(ss.Renderer):
    def __init__(self):
        super().__init__()
        self.messages = []
        self.current_subject = "math"
        self.current_difficulty = "intermediate"
        self.rag_enabled = False
        self.check_rag_status()

    def check_rag_status(self):
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                data = response.json()
                self.rag_enabled = data.get("rag_enabled", False)
        except Exception as e:
            print(f"Error checking RAG status: {e}")

    def ask_question(self, question: str):
        if not question.strip():
            return

        try:
            payload = {
                "question": question,
                "subject": self.current_subject,
                "difficulty": self.current_difficulty,
            }

            response = requests.post(
                f"{API_BASE_URL}/ask",
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                self.messages.append({
                    "role": "user",
                    "content": question,
                })
                self.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "subject": data["subject"],
                    "difficulty": data["difficulty"],
                    "steps": data.get("steps", []),
                })
            else:
                self.messages.append({
                    "role": "error",
                    "content": f"Error: {response.status_code}",
                })
        except Exception as e:
            self.messages.append({
                "role": "error",
                "content": f"Connection error: {str(e)}",
            })

    def clear_messages(self):
        self.messages = []


# Create state
state = TutorUI()

# UI Layout
main = ss.Page(
    title="Math & Physics AI Tutor",
    description="Interactive tutor for mathematics and physics with RAG support",
)

with main:
    with ss.Column(id="header"):
        ss.Heading("🧠 Math & Physics AI Tutor", level=1)
        ss.Text("Get step-by-step explanations for your math and physics questions")
        if state.rag_enabled:
            ss.Badge("RAG Enabled", color="green")

    with ss.Row(id="controls"):
        with ss.Column():
            subject_select = ss.Select(
                label="Subject",
                options=[
                    {"label": "Math", "value": "math"},
                    {"label": "Physics", "value": "physics"},
                ],
                value="math",
            )
            ss.bind(subject_select, state, "current_subject")

        with ss.Column():
            difficulty_select = ss.Select(
                label="Difficulty Level",
                options=[
                    {"label": "Beginner", "value": "beginner"},
                    {"label": "Intermediate", "value": "intermediate"},
                    {"label": "Advanced", "value": "advanced"},
                ],
                value="intermediate",
            )
            ss.bind(difficulty_select, state, "current_difficulty")

    with ss.Row(id="input_area"):
        question_input = ss.TextInput(
            label="Ask your question",
            placeholder="e.g., What is the derivative of x^2?",
        )

        def on_ask_click():
            state.ask_question(question_input.value)
            question_input.value = ""

        ask_button = ss.Button("Ask", on_click=on_ask_click)

    with ss.Row(id="chat_area"):
        chat = ss.DataFrame(
            data=state.messages,
            columns=["role", "content"],
        )
        ss.bind(chat, state, "messages")

    with ss.Row(id="actions"):
        clear_button = ss.Button("Clear Chat", on_click=lambda: state.clear_messages())
