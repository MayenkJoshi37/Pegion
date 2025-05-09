import os
import json
import torch
import chromadb
import markdown2 
import pdfplumber

from docx import Document
from cryptography.fernet import Fernet
from langchain_ollama import OllamaLLM
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Running on device: {device}")

app = Flask(__name__, static_folder='static')

app.config['UPLOAD_FOLDER'] = 'uploades'
app.config["LOGS_FOLDER"] = "logs"
app.config['VECTOR_DB_LOG'] = os.path.join(app.config['LOGS_FOLDER'], 'vector_db_log.json')
app.config['PDF_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs')
app.config['TEXT_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'text')
app.config['IMAGE_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
app.config['DOCX_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'docs')
app.config['ENCRYPTED_FOLDER'] = 'E-Files'
app.config['KEYS_FOLDER'] = os.path.join(app.config['ENCRYPTED_FOLDER'], 'secret_keys')

os.makedirs(app.config['ENCRYPTED_FOLDER'], exist_ok=True)
os.makedirs(app.config['KEYS_FOLDER'], exist_ok=True)

for folder in [app.config['UPLOAD_FOLDER'], app.config['PDF_FOLDER'], app.config['TEXT_FOLDER'],
               app.config['IMAGE_FOLDER'], app.config['DOCX_FOLDER'], app.config['LOGS_FOLDER']]:
               os.makedirs(folder, exist_ok=True)

EXT_TO_FOLDER = {
    '.pdf': app.config['PDF_FOLDER'],
    '.txt': app.config['TEXT_FOLDER'],
    '.docx': app.config['DOCX_FOLDER'],
    '.jpg': app.config['IMAGE_FOLDER'],
    '.png': app.config['IMAGE_FOLDER']
}

embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
vector_db = chromadb.PersistentClient(path='./chroma_db')
collection = vector_db.get_or_create_collection(name='document_chunks', metadata={'hnsw:space': 'cosine'})
llm = OllamaLLM(model='gemma3:4b-it-q8_0') # gemma2:9b llama3.2:1b gemma3:12b-it-qat gemma3:12b-it-q4_K_M

def extract_text_from_pdf(pdf_path):
    text = ''
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + '\n'
    except Exception as e:
        text = f'Error extracting text from PDF: {str(e)}'
    return text

def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f'Error extracting text from TXT: {str(e)}'

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f'Error extracting text from DOCX: {str(e)}'

def extract_text_from_image(image_path):
    try:
        prompt = f"""
        You are analyzing a medical image.

        - If the image contains text (like a prescription, report), **extract all the text clearly**.
        - If it is a scan (like an X-ray, MRI, or CT), **describe what can be observed**.

        Analyze the attached image carefully.
        {image_path}
        """

        response = llm.invoke(prompt)
        return response.strip()
    
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def chunk_text_by_line(text):
    return [line.strip() for line in text.split('\n') if line.strip()]

def chunk_text_by_paragraph(text):
    return [para.strip() for para in text.split('\n\n') if para.strip()]

def add_document_to_db(text_chunks, source_filename):
    embedding = embedding_model.encode(text_chunks).tolist()
    ids = [f'{source_filename}_{i}' for i in range(len(text_chunks))]
    collection.add(embeddings=embedding, documents=text_chunks, ids=ids)

    log_filename = f'embeddings_for_{secure_filename(source_filename)}.json'
    log_filepath = os.path.join(app.config['LOGS_FOLDER'], log_filename)

    db_log = [{'id': doc_id, 'chunk': chunk, 'embedding': emb} for doc_id, chunk, emb in zip(ids, text_chunks, embedding)]

    with open(log_filepath, 'w', encoding='utf-8') as f:
        json.dump(db_log, f, indent=4)

    print(f"Saved vector DB log for {source_filename} at {log_filepath}")

def get_relevant_chunks(query, num_chunks):
    query_embedding = embedding_model.encode(query).tolist()
    result = collection.query(query_embeddings=[query_embedding], n_results=num_chunks)
    return [doc for sublist in result['documents'] for doc in sublist] or []

def generate_response(user_message, relevant_chunks):
    context = '\n\n'.join(relevant_chunks) if relevant_chunks else 'No relevant Documents found.'

    prompt = f"""
            You are a smart, polite, and professional AI medical assistant designed to help patients understand their medical documents.  
            Your job is to simplify complex medical language and provide clear, helpful responses based on the patient's uploaded reports.  
            Be accurate, respectful, and easy to understand. Keep things just long enough to be useful—never too short or too long.
            
            - Respond using proper Markdown formatting:
              - Use # for main headings and ## for subheadings.
              - Use bullet points (-) for lists.
              - Highlight key terms or results in **bold**.
              - Keep paragraphs short and readable.
            
            - Focus on Indian medical scenarios, but handle general health questions too.  
            - Be gentle and cautious when discussing serious findings, and always recommend consulting a doctor.  
            - If the user talks casually or off-topic, respond briefly and gently steer the conversation back to health.  
            - If needed, refer to past information from earlier in the session.  
            
            ### **Context (if any):**  
            {context}
            
            ### **User's Query:**  
            {user_message}
            
            ### **Your Response:**  

            """

    response = llm.invoke(prompt)
    response_final = markdown2.markdown(response, extras=["strip"])  
    return response_final

def generate_and_save_key(file_name):
    key = Fernet.generate_key()
    key_path = os.path.join(app.config['KEYS_FOLDER'], f"{file_name}.key")
    with open(key_path, "wb") as f:
        f.write(key)
    return key

def load_key(file_name):
    key_path = os.path.join(app.config['KEYS_FOLDER'], f"{file_name}.key")
    with open(key_path, "rb") as f:
        return f.read()

def encrypt_file(file_path, encrypted_dir, file_name):
    key = generate_and_save_key(file_name)
    fernet = Fernet(key)

    with open(file_path, "rb") as f:
        data = f.read()

    encrypted_data = fernet.encrypt(data)

    encrypted_path = os.path.join(encrypted_dir, f"{file_name}.enc")
    with open(encrypted_path, "wb") as f:
        f.write(encrypted_data)

    os.remove(file_path)
    print(f"[INFO] Encrypted and saved: {encrypted_path}")
    return encrypted_path

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/login")
def login():
    return app.send_static_file("login.html")

@app.route("/doctor")
def doctor_dashboard():
    return app.send_static_file("doctor.html")

@app.route("/patient")
def patient_dashboard():
    return app.send_static_file("patient.html")

@app.route("/chat")
def chat():
    return app.send_static_file("chat.html")

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files part'}), 400

        files = request.files.getlist('files[]')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400

        uploaded_files = []
        unsupported_files = []

        for file in files:
            filename = secure_filename(file.filename)
            ext = os.path.splitext(filename)[1].lower()
            name_without_ext = os.path.splitext(filename)[0]
            temp_save_path = os.path.join(EXT_TO_FOLDER.get(ext, app.config['UPLOAD_FOLDER']), filename)

            if ext in EXT_TO_FOLDER:
                file.save(temp_save_path)

                # Extract text before encryption
                if ext == '.pdf':
                    text = extract_text_from_pdf(temp_save_path)
                elif ext == '.txt':
                    text = extract_text_from_txt(temp_save_path)
                elif ext == '.docx':
                    text = extract_text_from_docx(temp_save_path)
                elif ext in ['.png', '.jpg']:
                    text = extract_text_from_image(temp_save_path)
                else:
                    text = ""

                text_chunks = chunk_text_by_line(text)
                add_document_to_db(text_chunks, filename)

                encrypt_file(temp_save_path, app.config['ENCRYPTED_FOLDER'], name_without_ext)

                # Save the original file extension in a .meta file
                keys_dir = os.path.join(app.config['ENCRYPTED_FOLDER'], 'secret_keys')
                os.makedirs(keys_dir, exist_ok=True)
                meta_path = os.path.join(keys_dir, name_without_ext + ".meta")
                with open(meta_path, "w") as meta_file:
                    meta_file.write(ext)

                uploaded_files.append(filename)
            else:
                unsupported_files.append(filename)

        if not uploaded_files:
            return jsonify({'message': 'No valid documents processed.', 'unsupported_files': unsupported_files}), 200

        return jsonify({
            'message': 'Files uploaded, processed, and encrypted successfully.',
            'files': uploaded_files,
            'unsupported_files': unsupported_files
        }), 200

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    relevant_chunks = get_relevant_chunks(user_message, 10)
    llm_output = generate_response(user_message, relevant_chunks)

    return jsonify({'response': llm_output})

if __name__ == "__main__":
    app.run(debug=True)