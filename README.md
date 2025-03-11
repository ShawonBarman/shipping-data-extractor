# Shipping Data Extractor using OpenAI

AI-powered tool for extracting structured shipping data from various document formats (PDF, Excel, CSV). Features multi-page processing, intelligent field extraction, interactive data table, column management, export options, and an AI chat assistant for data analysis.

## Live: https://shipping-data-extractor-using-openai.up.railway.app/

## Features

- **Multi-format Document Support**: Process PDFs, Excel files, CSVs, and images
- **Intelligent Data Extraction**: AI-powered field recognition for shipping documents
- **Multi-page Processing**: Full support for multi-page PDFs with page detection
- **Interactive Data Table**: Sort, filter, and manage shipping data with ease
- **Column Management**: Customize visible columns and their order
- **Export Options**: Export to Excel, CSV, or JSON
- **AI Assistant**: Ask questions about your shipping data
- **Single-page Application**: Clean, intuitive interface with responsive design

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python with Flask
- **AI Integration**: OpenAI API (GPT-4) for document processing
- **PDF Processing**: PyPDF2
- **Data Handling**: Pandas
- **Styling**: Bootstrap 5, Font Awesome

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ShawonBarman/shipping-data-extractor-using-openAI.git
   cd shipping-data-extractor-using-openAI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   SECRET_KEY=your_flask_secret_key_here
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

1. **Upload Documents**: Drag and drop shipping documents onto the upload area
2. **Process Files**: Click "Extract Data" to start the AI processing
3. **View Results**: Explore the extracted data in the interactive table
4. **Customize View**: Toggle column visibility or reorder columns as needed
5. **Export Data**: Export to your preferred format (Excel, CSV, JSON)
6. **Ask Questions**: Use the AI assistant to analyze your shipping data

## Project Structure

```
project_root/
│
├── app.py                  # Main Flask application
│
├── static/
│   ├── css/
│   │   └── styles.css      # Main stylesheet
│   │
│   ├── js/
│   │   └── main.js         # Application JavaScript
│
├── templates/
│   └── index.html          # Single-page application template
│
├── uploads/                # Folder for uploaded files
│
├── .env                    # Environment variables (not in repo)
│
└── requirements.txt        # Python dependencies
```

## Configuration

Key configuration options:

- `UPLOAD_FOLDER`: Directory where uploaded files are stored
- `ALLOWED_EXTENSIONS`: File extensions that can be processed
- `TABLE_FIELDS`: Fields to extract from shipping documents

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for GPT models powering the extraction logic
- Bootstrap team for responsive UI components
- Font Awesome for icons
