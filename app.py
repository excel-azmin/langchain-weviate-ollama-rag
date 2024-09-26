import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import weaviate
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields # type: ignore

# Initialize Flask and Flask-RESTx
app = Flask(__name__)
api = Api(app, version='1.0', title='RAG API',
          description='A Retrieval-Augmented Generation API')

# Define the API namespace
ns = api.namespace('api', description='RAG operations')

# Define the API models (for Swagger documentation)
query_model = api.model('Query', {
    'query': fields.String(required=True, description='The query to ask')
})

# Functions for PDF extraction and Weaviate
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)

pdf_path = "/home/azmin/Desktop/Research/rag/CKA-Part1.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)

# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")

# Check if class already exists
schema = client.schema.get()
client.schema.delete_all()
class_obj = {
    "class": "PDFTextChunk",
    "vectorizer": "none",
}
client.schema.create_class(class_obj)

# Load a pre-trained transformer model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy().flatten().tolist()  # Convert NumPy array to list

# Store chunks in Weaviate
for chunk in chunks:
    vector = embed_text(chunk)
    client.data_object.create(
        data_object={
            "text": chunk,
        },
        class_name="PDFTextChunk",
        vector=vector  # Pass the vector correctly
    )

def query_weaviate(query):
    vector = embed_text(query)
    response = client.query.get("PDFTextChunk", ["text"]).with_near_vector({"vector": vector, "certainty": 0.1}).with_limit(5).do()
    # Extract relevant chunks
    relevant_chunks = response['data']['Get']['PDFTextChunk']
    return relevant_chunks

# Set up the Langchain Ollama LLM
ollama_llm = Ollama(model="llama3.1", base_url="http://localhost:11434")

def generate_answer(chunks, query):
    # Make sure 'chunks' is a list of dictionaries with a 'text' key
    context = " ".join([chunk['text'] for chunk in chunks])

    # Create a Langchain PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI assistant skilled in analyzing and summarizing technical documents. Given the following context from a technical document and a user query, provide a structured and detailed response.

        Context: 
        {context}

        Question: 
        {question}

        Response should include:
        1. Provide a direct answer to the user's query.

        Make sure your response is clear, professional, and helpful.
        """

    )

    # Generate the full prompt
    prompt = prompt_template.format(context=context, question=query)
    
    # Use Ollama to generate the answer
    answer = ollama_llm.invoke(prompt)
    
    return answer.strip()  # Directly return the string response

# API Resource
@ns.route('/ask')
class AskResource(Resource):
    @ns.expect(query_model)
    @ns.response(200, 'Success')
    @ns.response(400, 'Validation Error')
    def post(self):
        '''Ask a question about the PDF content'''
        query = request.json.get('query')
        print("Query:", query)
        if not query:
            api.abort(400, "No query provided")
        
        relevant_chunks = query_weaviate(query)
        print("Relevant chunks:", relevant_chunks)
        answer = generate_answer(relevant_chunks, query)
        print("\n\n\nAnswer:", answer)    
        return jsonify({"answer": answer})


# Add namespace to the API
api.add_namespace(ns)

if __name__ == "__main__":
    app.run(debug=True)
