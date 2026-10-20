import gradio as gr
import laya
import json

print("Loading Laya engine...")
# Router automatically dispatches between English (ModernBERT) and Multilingual (mmBERT)
agent = laya.Router()
print("✅ Laya is warm and ready.")

def evaluate_generic(state_input, questions_input):
    """
    Accepts:
      - state_input: dict or JSON string representing the state/content
      - questions_input: dict or JSON string defining the typed questions
    """
    try:
        # Parse inputs if passed as strings (via cURL / HTTP)
        state = json.loads(state_input) if isinstance(state_input, str) else state_input
        questions = json.loads(questions_input) if isinstance(questions_input, str) else questions_input

        # Forward directly to Laya's System 1 forward pass
        result = agent.system_one(state=state, questions=questions)

        # Serialize typed answer objects into clean JSON
        serialized_answers = {}
        for key, ans in result.answers.items():
            entry = {}
            if hasattr(ans, "choice"):
                entry["type"] = "choice"
                entry["choice"] = ans.choice
                entry["probabilities"] = ans.probabilities
            elif hasattr(ans, "noul"):
                entry["type"] = "noul"
                entry["probability"] = ans.noul
            elif hasattr(ans, "score"):
                entry["type"] = "score"
                entry["score"] = ans.score
                entry["distribution"] = getattr(ans, "distribution", None)
            serialized_answers[key] = entry

        return {
            "status": "success",
            "answers": serialized_answers,
            "usage": getattr(result, "usage", {})
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }

# Gradio Interface exposing the generic endpoint
demo = gr.Interface(
    fn=evaluate_generic,
    inputs=[
        gr.JSON(label="State (Any JSON or Text Object)"),
        gr.JSON(label="Questions Schema (Choices, Scores, Nouls)")
    ],
    outputs=gr.JSON(label="Laya Evaluated Decision"),
    title="Laya Generic System-1 API",
    api_name="system_one"
)

demo.launch()
