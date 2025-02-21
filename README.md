CampusBot - AI Chatbot for Campus Assistance

CampusBot is an AI-powered chatbot designed to assist students, faculty, and visitors with campus-related queries. It provides instant responses about campus facilities, events, academic information, and more.

Features

📌 Student Assistance - Get details about courses, faculty, and academic schedules.

🏛 Campus Navigation - Find locations of departments, libraries, and other facilities.

📅 Event Updates - Stay informed about campus events, workshops, and deadlines.

🍽 Cafeteria Menu - Check daily food options and prices.

📝 General Queries - Ask about admission, hostel, placements, and more.

🏆 AI-Powered - Uses OpenAI’s API for intelligent responses.

Overview

<img width="959" alt="image" src="https://github.com/user-attachments/assets/9fede338-ccff-4a01-84af-8875a5a83b08" />


CampusBot is an AI-powered chatbot designed to assist students, faculty, and visitors with campus-related queries. It provides instant responses about campus facilities, events, academic information, and more.

Features

📌 Student Assistance - Get details about courses, faculty, and academic schedules.

🏛 Campus Navigation - Find locations of departments, libraries, and other facilities.

📅 Event Updates - Stay informed about campus events, workshops, and deadlines.

🍽 Cafeteria Menu - Check daily food options and prices.

📝 General Queries - Ask about admission, hostel, placements, and more.

🏆 AI-Powered - Uses OpenAI’s API for intelligent responses.

Tech Stack

Frontend: Streamlit (for UI)

Backend: FastAPI

Database: FAISS (for embeddings)

NLP Model: OpenAI API (GPT-based chatbot)

Deployment: Streamlit Cloud

Installation

1️⃣ Clone the Repository

git clone https://github.com/your-username/CampusBot.git
cd CampusBot

2️⃣ Create a Virtual Environment

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Set Up Environment Variables

Create a .env file and add your OpenAI API key:

OPENAI_API_KEY=your-api-key

5️⃣ Run the Chatbot

streamlit run app.py

Deployment on Streamlit Cloud

Push your project to GitHub.

Go to Streamlit Cloud and create a new app.

Select your repository and set app.py as the entry point.

Add your API key in Secrets Management.

Click Deploy and enjoy your chatbot!

Folder Structure

CampusBot/
│── faiss_index/          # Stores FAISS embeddings
│── .env                  # API keys (excluded from GitHub)
│── app.py                # Main chatbot interface
│── generate.py           # Script to generate embeddings
│── embedding_model.pkl   # Pre-trained embedding model
│── requirements.txt      # Project dependencies
│── .gitignore            # Files to exclude from GitHub
│── .streamlit/secrets.toml  # Streamlit deployment secrets

Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

Contact

For any queries, reach out to me at devamsingh0009@gmail.com.

