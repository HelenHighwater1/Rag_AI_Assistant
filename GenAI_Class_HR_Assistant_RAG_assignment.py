import os
import openai
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import gradio as gr

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = "Your API Key Here"


from langchain_community.document_loaders import PyPDFLoader

Doc_loader = PyPDFLoader("nestle_hr_policy_pdf_2012.pdf")
extracted_text = Doc_loader.load()


text_splitter  = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=50,
    separators=["\n\n", "\n", r"(?<=\. )", " ", ""]
)
splitted_text=text_splitter.split_documents(extracted_text)


embeddings = OpenAIEmbeddings()

vectordb = FAISS.from_documents(
    documents=splitted_text,
    embedding=embeddings
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


retriever_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       )


## Gradio Interface

def answer_question(question):
    if not question.strip():
        return "Please enter a question."
    try:
        result = retriever_chain.invoke({"query": question})
        answer = result.get("result", "(No answer)")
        srcs = result.get("source_documents", []) or []
        seen = []
        for d in srcs:
            s = (d.metadata or {}).get("source") or (d.metadata or {}).get("file_path") or "document"
            if s not in seen:
                seen.append(s)
        if seen:
            answer += "\n\nSources: " + ", ".join(seen[:3])
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Frontend

with gr.Blocks(title="HR Assistant - Nestle Policies") as demo:
    gr.Markdown("## HR Assistant")
    gr.Markdown("Ask questions about Nestle HR policies and get AI-powered answers based on the official documentation.")
    
    with gr.Row():
        question = gr.Textbox(
            lines=4, 
            placeholder="Type your question here... (e.g., 'What is the maternity leave policy?')", 
            label="Your Question",
            scale=4
        )
    
    with gr.Row():
        submit = gr.Button("Submit Question", variant="primary", size="lg")
        clear = gr.Button("Clear", variant="secondary")
    
    with gr.Row():
        answer = gr.Textbox(
            lines=8, 
            label="Answer", 
            interactive=False,
            scale=4
        )

    submit.click(fn=answer_question, inputs=question, outputs=answer)
    clear.click(lambda: ("", ""), outputs=[question, answer])
    


# Launch the interface
if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860, inbrowser=True)
