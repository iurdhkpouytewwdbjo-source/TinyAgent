
from flask import Flask, render_template, request, jsonify
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core import Agent
from src.tools import add, count_letter_in_string, compare, get_current_datetime, search_wikipedia
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(
    api_key="sk123",
    base_url="http://10.121.193.81:11084/v1",
)

# Initialize the agent
agent = Agent(
    client=client,
    model="google-claude-sonnet-4",
    tools=[get_current_datetime, add, compare, count_letter_in_string, search_wikipedia],
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = agent.get_completion(user_input)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=51904, debug=True)
