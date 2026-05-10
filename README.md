# Smart Job Description Analyzer

A Streamlit-based application that analyzes job descriptions using classical Natural Language Processing (NLP) techniques to extract valuable insights without requiring transformer models.

## Features

- **Skill Extraction**: Automatically identify technical and professional skills from job descriptions
- **Sentiment Analysis**: Analyze the tone and sentiment of job postings using VADER
- **Text Analytics**: Perform Named Entity Recognition, POS tagging, and lemmatization
- **TF-IDF Analysis**: Extract key terms and measure term importance using scikit-learn
- **Data Visualization**: Generate insightful charts and visualizations for analysis results
- **Bulk Processing**: Process multiple job descriptions efficiently

## Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/) — interactive web application
- **NLP**: [NLTK](https://www.nltk.org/) — tokenization, POS tagging, NER, lemmatization
- **Sentiment**: [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) — lexicon-based sentiment analysis
- **ML**: [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorization and similarity metrics
- **Data**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) — data manipulation
- **Visualization**: [Matplotlib](https://matplotlib.org/) — chart generation

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - On PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - On Command Prompt:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

Run the application using PowerShell:
```powershell
.\Run-App.ps1
```

Or manually start Streamlit:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### How to Use

1. **Input**: Paste or upload a job description
2. **Analyze**: The app will automatically extract skills, analyze sentiment, and generate insights
3. **Review**: Examine the extracted information, visualizations, and metrics
4. **Export**: Download results as needed

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── Run-App.ps1         # PowerShell script to launch the app
```

## Notes

- This application uses **classical NLP techniques** (no transformer models), making it lightweight and fast
- NLTK requires downloading language data on first run (handled automatically)
- The app is optimized for job description analysis but can work with any text

## License

[Add your license here if applicable]

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
