### Project Overview:
This project was for a Generative AI class.  It creates an AI-powered HR assistant that answers questions about company HR policies using a Retrieval-Augmented Generation approach. The system processes PDF documents and provides intelligent responses to user queries.

### Technical Implementation:
I built this with Python using LangChain framework and OpenAI's GPT-3.5-turbo model. The system extracts text from PDFs, splits it into chunks, converts to vector embeddings, and stores them in a FAISS database for efficient retrieval.

### User Interface:
I used a Gradio web interface where users input questions and receive relevant policy information. The interface includes input fields, submit buttons, and an answer display area.

### Technical Requirements:
Requires Python with LangChain, OpenAI, FAISS, and Gradio packages. Needs an OpenAI API key and runs locally on port 7860.
