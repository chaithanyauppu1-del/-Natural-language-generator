# Natural Language Generator

A professional, intermediate-level internship project implementing a web-based **Natural Language Generation (NLG)** application. Built with **Python**, **PyTorch**, **Hugging Face Transformers**, and **Flask**, this application leverages the pretrained **FLAN-T5** sequence-to-sequence model to transform natural language instructions and structured data into fluent, coherent, and contextually accurate prose.

---

## 1. Project Overview

Natural Language Generation (NLG) is a core subfield of Artificial Intelligence and Natural Language Processing (NLP) focused on generating human-like natural text from structured data or unstructured prompts. 

This project provides an end-to-end web platform featuring:
* **Pretrained Transformer Inference**: Powered by Google's `google/flan-t5-base` model.
* **Dual Input Processing**: Converts both freeform natural language prompts and structured key-value data into polished textual paragraphs.
* **Style Customization**: Allows users to tailor generation tone across *General*, *Formal*, *Concise*, and *Descriptive* styles.
* **Automated NLP Evaluation**: Scores generated outputs against optional reference ground-truth text using **ROUGE** (ROUGE-1, ROUGE-2, ROUGE-L) and **BLEU** (BLEU-1 to BLEU-4) metrics.
* **Descriptive Text Analytics**: Computes character, word, and sentence count statistics.
* **Modern Web Interface**: Responsive HTML5/CSS3/JavaScript UI with dark mode, preset examples, live parameters control, copy to clipboard, file downloads (.txt / .json), and session history.

---

## 2. Problem Statement

Organizations and data systems frequently generate vast amounts of structured data (e.g., database tables, key-value metrics, analytical reports). However, presenting raw metrics directly to non-technical stakeholders can reduce clarity and readability. Traditional rule-based or template-driven text generators produce rigid, repetitive, and unnatural sentences.

**Natural Language Generator** solves this challenge by leveraging modern instruction-tuned transformer models (`FLAN-T5`) to intelligently synthesize raw metrics and prompts into natural, grammatically sound, and contextually rich human-readable reports.

---

## 3. Objectives

* **Practical NLP & Transformer Usage**: Implement model loading, tokenization, instruction prompt construction, sequence generation, and decoding using Hugging Face Transformers.
* **Data Transformation Engine**: Automatically parse structured tabular/key-value data (e.g., student pass percentages, financial growth rates) into coherent narrative prose.
* **Controllable Text Synthesis**: Provide intuitive parameter controls (`max_new_tokens`, `num_beams`, `temperature`, `repetition_penalty`) and tone selection.
* **Standardized Metric Evaluation**: Integrate ROUGE and BLEU automated evaluation algorithms to compare synthesized text against ground-truth references.
* **Robust Software Architecture**: Maintain modular code separation across model inference (`src/generator.py`), input preprocessing (`src/preprocessing.py`), evaluation (`src/evaluation.py`), Flask backend routes (`app.py`), and interactive frontend (`templates/` and `static/`).

---

## 4. Features

1. **Instruction-Tuned Transformer Engine**: Loaded once on server startup with PyTorch `@torch.inference_mode()` for fast CPU/GPU inference.
2. **Structured Data to Text Conversion**: Seamlessly converts key-value inputs into narrative paragraphs without simple string repetition.
3. **Multi-Style Tone Selector**:
   - **General**: Standard balanced natural text generation.
   - **Formal**: Professional, academic, and authoritative tone.
   - **Concise**: Brief, high-density summary of key facts.
   - **Descriptive**: Rich, detailed, and comprehensive narrative.
4. **Advanced Generation Hyper-Parameters**: Sliders to adjust maximum generation length, beam search count, sampling temperature, and repetition penalties.
5. **Live Text Analytics**: Instant display of character count, word count, sentence count, and average word length.
6. **Automated ROUGE & BLEU Metric Engine**: Evaluates n-gram overlap when reference text is provided, paired with explanatory disclaimers regarding metric limitations.
7. **Session Generation History**: In-memory log keeping track of past generations during the session with single-click clear.
8. **Export & Sharing Capabilities**: Quick copy to clipboard, `.TXT` file export, and structured `.JSON` report downloads.

---

## 5. How the System Works

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│  User Interface │ ──> │ Input Preprocessing  │ ──> │ Prompt Construction    │
│  (HTML5/CSS/JS) │     │ (Clean, KV Parsing)  │     │ (Style & Task Template)│
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                 │
┌─────────────────┐     ┌──────────────────────┐                 ▼
│ Output Dashboard│ <── │  Automated Metrics   │ <── ┌────────────────────────┐
│ (Copy, TXT, JSON│     │ (ROUGE, BLEU, Stats) │     │ FLAN-T5 Model Engine   │
└─────────────────┘     └──────────────────────┘     │ (PyTorch Beam Search)  │
                                                     └────────────────────────┘
```

1. **User Input & Configuration**: User selects a generation style, inputs a prompt or structured data, and optionally pastes a reference ground-truth text.
2. **Preprocessing**: `src/preprocessing.py` sanitizes inputs, formats key-value entries (e.g. `Revenue: ₹12.5M; Growth: 18%`), and applies style instruction templates.
3. **Inference Execution**: `src/generator.py` tokenizes the formatted prompt and executes FLAN-T5 sequence generation using PyTorch.
4. **Post-Processing & Decoding**: The generated token IDs are decoded into clean text, skipping special tokens.
5. **Evaluation**: `src/evaluation.py` calculates word/sentence statistics and computes ROUGE & BLEU scores if a reference text is present.
6. **UI Rendering**: The Flask backend returns a JSON payload and the interactive frontend updates output cards, bar charts, and history logs smoothly.

---

## 6. System Architecture

The application adopts a modular multi-tier architecture separating concerns between data preparation, model management, evaluation, API endpoints, and presentation:

* **Presentation Layer**: HTML5, CSS3 (Glassmorphic dark design), Vanilla JS (AJAX fetch requests, DOM manipulation).
* **API Layer**: Flask REST application handling `/generate`, `/evaluate`, `/history`, `/clear`, and `/health`.
* **Business Logic Layer**:
  - `src/preprocessing.py`: Input cleaning and instruction prompt building.
  - `src/generator.py`: Hugging Face pipeline, PyTorch device mapping (`cuda` / `cpu`), model singleton lifecycle.
  - `src/evaluation.py`: Descriptive statistics engine, `rouge_score` integration, and NLTK BLEU calculation with smoothing.
* **Configuration Layer**: `config.py` defining central defaults and environment variables.

---

## 7. Technology Stack

* **Language**: Python 3.11+
* **Deep Learning & NLP**: PyTorch, Hugging Face `transformers`
* **Pretrained Language Model**: `google/flan-t5-base` (with automatic fallback to `google/flan-t5-small`)
* **Data Processing & Analytics**: Pandas, NLTK, `rouge-score`
* **Web Framework**: Flask
* **Frontend**: HTML5, CSS3 (Vanilla CSS with Flexbox/Grid), JavaScript (ES6+ Fetch API)
* **Testing Suite**: PyTest

---

## 8. Project Structure

```text
Natural-Language-Generator/
│
├── app.py                     # Main Flask server entry point & API route handlers
├── config.py                  # Central configuration, style templates & defaults
├── requirements.txt           # Python dependency requirements file
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Git ignore rules
│
├── models/                    # Offline model weights & documentation
│   └── README.md
│
├── src/                       # Core python source package
│   ├── __init__.py            # Package initialization marker
│   ├── generator.py           # FLAN-T5 model loading, PyTorch device setup & generation
│   ├── preprocessing.py       # Text cleaning, key-value normalization & prompt builder
│   └── evaluation.py          # ROUGE & BLEU evaluation engine & text statistics
│
├── templates/                 # HTML templates
│   └── index.html             # Main responsive web application interface
│
├── static/                    # Static assets
│   ├── css/
│   │   └── style.css          # Glassmorphism dark mode stylesheet
│   └── js/
│       └── script.js          # Client-side UI logic & AJAX handlers
│
├── data/                      # Preset example data & benchmark inputs
│   └── README.md
│
└── tests/                     # Unit & integration test suite
    └── test_generator.py      # Automated tests for preprocessing, generator, metrics & Flask
```

---

## 9. Installation Instructions

### Prerequisites
* **Python**: 3.9 or higher installed.
* **Git**: Installed (optional).

### Step-by-Step Setup

1. **Clone or Navigate to the Workspace Directory**:
   ```bash
   cd "Natural-Language-Generator"
   ```

2. **Create and Activate a Virtual Environment** (Recommended):
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 10. How to Run

1. **Start the Flask Application Server**:
   ```bash
   python app.py
   ```

2. **Access the Web Interface**:
   Open your browser and navigate to:
   `http://127.0.0.1:5000` or `http://localhost:5000`

   *Note: On initial startup, Hugging Face will automatically download the lightweight `google/flan-t5-base` weights. Subsequent server starts load instantly from cache.*

---

## 11. Example Inputs & Outputs

### Example 1: Structured Data → Narrative Text

* **Input Structured Data**:
  ```text
  Department: Computer Science
  Students: 120
  Average Attendance: 91%
  Pass Percentage: 94%
  Top Subject: Data Structures
  ```
* **Style**: Formal
* **Generated Output**:
  > *"The Department of Computer Science currently maintains an enrollment of 120 students, achieving an average attendance rate of 91% and a high pass percentage of 94%. Data Structures remains the top performing subject."*

---

### Example 2: Corporate Financial Summary

* **Input Structured Data**:
  ```text
  Revenue: ₹12.5M
  Growth: 18%
  Customers: 24,500
  Churn: 4.2%
  ```
* **Style**: Concise
* **Generated Output**:
  > *"The business reported ₹12.5M in revenue with 18% annual growth, expanding its customer base to 24,500 while maintaining a low 4.2% churn rate."*

---

### Example 3: Natural Language Prompt

* **Input Prompt**:
  `Write a short formal description of a university computer science department.`
* **Style**: Descriptive
* **Generated Output**:
  > *"The University Computer Science Department offers comprehensive academic programs covering artificial intelligence, software engineering, and data science, empowering students with core computational theory and hands-on laboratory experience."*

---

## 12. Model Information

* **Base Architecture**: Transformer Encoder-Decoder (Text-to-Text Transfer Transformer).
* **Model Checkpoint**: `google/flan-t5-base` (250 Million Parameters).
* **Fallback Checkpoint**: `google/flan-t5-small` (60 Million Parameters).
* **Instruction Tuning**: FLAN (Fine-tuned Language Net) fine-tuning provides superior zero-shot and few-shot task execution without custom fine-tuning.
* **Inference Mode**: Evaluated in PyTorch `@torch.inference_mode()` with beam search decoding.

---

## 13. Evaluation Methodology & Metric Limitations

The system incorporates two standard NLP evaluation metrics when a ground-truth reference text is provided:

1. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**:
   - **ROUGE-1**: Unigram overlap measure.
   - **ROUGE-2**: Bigram overlap measure.
   - **ROUGE-L**: Longest Common Subsequence (LCS) measure.

2. **BLEU (Bilingual Evaluation Understudy)**:
   - Evaluates cumulative precision for n-grams up to 4-grams using NLTK sentence BLEU with smoothing.

### Critical Note on Automated Metric Limitations
> Automated metrics such as ROUGE and BLEU measure strictly **n-gram lexical surface overlap** between generated output and reference text. While useful for benchmark consistency, high ROUGE/BLEU scores do not guarantee semantic accuracy, factual truth, or human engagement. They should always be interpreted alongside qualitative human assessment.

---

## 14. Testing

The repository includes a comprehensive automated test suite in `tests/test_generator.py`.

Run tests using PyTest:
```bash
pytest tests/ -v
```

**Test Coverage**:
* Input cleaning & whitespace normalization.
* Key-value & JSON structured data parsing.
* Style prompt template building & max-length clamping.
* Text statistics calculation (characters, words, sentences).
* ROUGE & BLEU metric evaluation functions.
* Flask HTTP endpoints (`/`, `/health`, `/generate`, `/evaluate`, `/clear`).

---

## 15. Limitations & Future Enhancements

### Current Limitations
* **Model Size Balance**: `flan-t5-base` strikes a balance between generation quality and execution speed on standard CPU machines, but may produce simpler phrasing compared to multi-billion parameter models (e.g. T5-3B or Llama-3).
* **Domain Adaptation**: The model relies on zero-shot instruction prompt engineering rather than task-specific fine-tuning.

### Future Enhancements
* Support for fine-tuning on custom enterprise tabular datasets using Hugging Face `PEFT` / `LoRA`.
* Export outputs in PDF / Docx format.
* Streaming response generation via Server-Sent Events (SSE).
* Integration with vector store databases for Retrieval-Augmented Generation (RAG).

---

## 16. License

This project is open-source and available under the **MIT License**.
