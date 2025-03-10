import os
import json
import uuid
import datetime
from flask import Flask, request, render_template, jsonify, send_file, session
from werkzeug.utils import secure_filename
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import io

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))  # For session management

# Configure OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'gif', 'tiff', 'jpeg', 'jpg', 'png', 'bmp', 'webp', 
                      'xls', 'xlsx', 'xlsm', 'xlsb', 'csv', 'xlt', 'xltx', 'xltm', 'xlam'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the table fields we want to extract
TABLE_FIELDS = [
    "office", "batch_no", "customer", "type", "reference_number", "booking_number", 
    "bol_number", "po_number", "container_number", "container_size", "container_type", 
    "pickup_location_name", "delivery_location_name", "delivery_street_address", 
    "delivery_city", "delivery_state", "delivery_zip", "return_location", 
    "container_weight", "commodity", "number_of_packages", "eta_date", 
    "steam_ship_line", "vessel", "voyage", "cut_off_date", "early_release_date", 
    "seal", "pickup_number", "pickup_appointment_date_time", 
    "delivery_appointment_date_time", "Options", "Tags", "Notes"
]

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file with improved debugging"""
    text = ""
    page_texts = []  # Store each page separately for debugging
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            print(f"PDF has {total_pages} pages")
            
            for page_num in range(total_pages):
                page_text = reader.pages[page_num].extract_text()
                page_texts.append(page_text)
                text += page_text + "\n\n=== PAGE BREAK ===\n\n"  # Clear page separator for model
                print(f"Extracted page {page_num+1}/{total_pages}, length: {len(page_text)} chars")
            
            # Debugging information
            print(f"Total extracted text length: {len(text)} chars")
            print(f"Individual page lengths: {[len(pt) for pt in page_texts]}")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def process_excel_file(file_path):
    """Extract data from Excel file"""
    try:
        df = pd.read_excel(file_path)
        # Basic processing - in a real app, you'd have more complex logic here
        return df.to_dict('records')
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        return None

def process_csv_file(file_path):
    """Extract data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        # Basic processing - in a real app, you'd have more complex logic here
        return df.to_dict('records')
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return None

def process_pdf_with_openai(pdf_text):
    """Send PDF text to OpenAI for extraction with improved multi-page handling"""
    try:
        # Create a field list string for the prompt
        field_list = "\n".join([f"- {field}" for field in TABLE_FIELDS])
        
        # Count approximate tokens (rough estimate: 4 chars ≈ 1 token)
        approx_tokens = len(pdf_text) // 4
        print(f"Approximate token count: {approx_tokens}")
        
        # If text is very large, we need to handle it differently
        if approx_tokens > 12000:  # GPT-4 context limit is around 8k-32k depending on version
            print("PDF text is very large, processing in chunks...")
            return process_large_pdf(pdf_text, field_list)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a specialized assistant for extracting shipping and logistics information from documents.
                    
                    IMPORTANT: The text comes from a multi-page PDF. Make sure to extract information from ALL pages, not just the first page.
                    Page breaks are marked with '=== PAGE BREAK ==='. Process the entire document.
                    
                    Extract the following fields from the provided text. For any field not found, leave it as an empty string.
                    Return the result as a JSON object with the following fields:
                    
                    {field_list}
                    
                    Format your response as a JSON array of objects. If multiple shipments are found, include multiple objects in the array.
                    Make sure to search the ENTIRE document across all pages.
                    Return ONLY the JSON array, no other text."""
                },
                {
                    "role": "user",
                    "content": f"Extract the shipping information from the following multi-page document:\n\n{pdf_text}"
                }
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        print(f"Extracted {len(result if isinstance(result, list) else result.get('data', [])) if result else 0} shipping records")
        return result
    except Exception as e:
        print(f"Error processing with OpenAI: {e}")
        return None

def process_large_pdf(pdf_text, field_list):
    """Process a large PDF by breaking it into chunks and merging results"""
    # Split by page breaks
    pages = pdf_text.split("=== PAGE BREAK ===")
    all_results = []
    
    # Process in chunks of 3 pages
    chunk_size = 3
    for i in range(0, len(pages), chunk_size):
        chunk = "\n\n".join(pages[i:i+chunk_size])
        print(f"Processing chunk {i//chunk_size + 1}/{(len(pages) + chunk_size - 1)//chunk_size}, pages {i+1}-{min(i+chunk_size, len(pages))}")
        
        if not chunk.strip():
            continue
            
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a specialized assistant for extracting shipping and logistics information from documents.
                    
                    This is chunk {i//chunk_size + 1} of a multi-page PDF. Extract all shipping information from this chunk.
                    
                    Extract the following fields from the provided text. For any field not found, leave it as an empty string.
                    Return the result as a JSON object with the following fields:
                    
                    {field_list}
                    
                    Format your response as a JSON array of objects. If multiple shipments are found, include multiple objects in the array.
                    Return ONLY the JSON array, no other text."""
                },
                {
                    "role": "user",
                    "content": f"Extract the shipping information from this chunk of the document:\n\n{chunk}"
                }
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            chunk_result = json.loads(response.choices[0].message.content)
            if isinstance(chunk_result, list):
                all_results.extend(chunk_result)
            elif 'data' in chunk_result:
                all_results.extend(chunk_result['data'])
            else:
                all_results.append(chunk_result)
            print(f"Found {len(chunk_result if isinstance(chunk_result, list) else chunk_result.get('data', [1])) if chunk_result else 0} records in chunk")
        except Exception as e:
            print(f"Error processing chunk: {e}")
    
    print(f"Total records found across all chunks: {len(all_results)}")
    return all_results

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Process uploaded files and extract data"""
    if 'files[]' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        return jsonify({'success': False, 'message': 'No files selected'})
    
    # Get EZ ID if provided
    ez_id = request.form.get('ez_id', '')
    
    all_results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process based on file type
            file_extension = filename.rsplit('.', 1)[1].lower()
            
            if file_extension == 'pdf':
                # Extract text from PDF
                print(f"Processing PDF file: {filename}")
                pdf_text = extract_text_from_pdf(filepath)
                if not pdf_text:
                    return jsonify({'success': False, 'message': f'Failed to extract text from {filename}'})
                
                # Process with OpenAI
                result = process_pdf_with_openai(pdf_text)
                if not result:
                    return jsonify({'success': False, 'message': f'Failed to process {filename} with OpenAI'})
                
            elif file_extension in ['xls', 'xlsx', 'xlsm', 'xlsb']:
                # Process Excel file
                result = process_excel_file(filepath)
                if not result:
                    return jsonify({'success': False, 'message': f'Failed to process Excel file {filename}'})
                
            elif file_extension == 'csv':
                # Process CSV file
                result = process_csv_file(filepath)
                if not result:
                    return jsonify({'success': False, 'message': f'Failed to process CSV file {filename}'})
            
            else:
                # For image files (in a real app, you'd use OCR here)
                return jsonify({'success': False, 'message': f'File type {file_extension} processing not implemented yet'})
            
            # Add results
            if isinstance(result, list):
                for item in result:
                    item['source_file'] = filename
                    # Add timestamp and unique ID for tracking
                    item['processed_at'] = datetime.datetime.now().isoformat()
                    item['record_id'] = str(uuid.uuid4())
                all_results.extend(result)
            elif isinstance(result, dict) and 'data' in result:
                # If result has a 'data' key containing the array
                data_items = result['data']
                for item in data_items:
                    item['source_file'] = filename
                    item['processed_at'] = datetime.datetime.now().isoformat()
                    item['record_id'] = str(uuid.uuid4())
                all_results.extend(data_items)
            else:
                # If result is directly the object
                result['source_file'] = filename
                result['processed_at'] = datetime.datetime.now().isoformat()
                result['record_id'] = str(uuid.uuid4())
                all_results.append(result)
    
    # Store results in session for later use
    session['extraction_results'] = all_results
    
    return jsonify({
        'success': True, 
        'count': len(all_results), 
        'labeled_data': all_results
    })

@app.route('/query', methods=['POST'])
def query():
    """Process a query about the extracted data"""
    question = request.form.get('question', '')
    results = session.get('extraction_results', [])
    
    if not results:
        return jsonify({'error': 'No extraction results available. Please upload and process files first.'})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that answers questions about shipping and logistics information. Use only the provided data to answer questions. Keep your answers concise and focused on the data provided."
                },
                {
                    "role": "user",
                    "content": f"Here is the shipping data:\n{json.dumps(results)}\n\nQuestion: {question}"
                }
            ]
        )
        answer = response.choices[0].message.content
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': f'Error processing question: {e}'})

@app.route('/export', methods=['POST'])
def export_data():
    """Export data in various formats"""
    data = request.json.get('data', [])
    format_type = request.json.get('format', 'csv')
    
    if not data:
        return jsonify({'success': False, 'message': 'No data to export'})
    
    try:
        if format_type == 'excel':
            # Create Excel file in memory
            output = io.BytesIO()
            df = pd.DataFrame(data)
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Shipping Data')
            output.seek(0)
            
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'shipping_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            )
            
        elif format_type == 'csv':
            # Create CSV in memory
            output = io.StringIO()
            df = pd.DataFrame(data)
            df.to_csv(output, index=False)
            output.seek(0)
            csv_data = output.getvalue()
            
            return jsonify({
                'success': True,
                'data': csv_data,
                'filename': f'shipping_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            })
            
        elif format_type == 'json':
            # Simply return the JSON data
            return jsonify({
                'success': True,
                'data': data,
                'filename': f'shipping_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            })
            
        else:
            return jsonify({'success': False, 'message': 'Unsupported export format'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error exporting data: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)