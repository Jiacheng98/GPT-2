import tiktoken
import torch
import os
from model import GPT
from torch.nn import functional as F
from flask import Flask, request, jsonify
import traceback

class SentenceCompletionModel:
    def __init__(self):
        # Load a model
        checkpoint_path = os.path.join("log/model_05000.pt")
        checkpoint = torch.load(checkpoint_path)
        self.model = GPT(checkpoint["config"])
        self.model.load_state_dict(checkpoint['model'])

    def complete_sentence(self, text: str) -> str:
        num_sequences = 1
        max_length = 50
        # Prefix tokens
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        x = self.model.generate_sequence(self.model, tokens, num_sequences, max_length)
        return enc.decode(x[0].tolist())
       
app = Flask(__name__)
model = SentenceCompletionModel()

@app.route('/complete-sentence', methods=['POST'])
def complete_sentence():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Check if 'text' key is present in the request data
        if 'text' not in data:
            return jsonify({"error": "Invalid input. 'text' key is required."}), 400

        # Extract the text from the payload
        text = data['text']

        # Call the model to get the completion
        completed_sentence = model.complete_sentence(text)

        # Return the completed sentence
        return jsonify({"completed_sentence": completed_sentence})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return 'OK', 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)