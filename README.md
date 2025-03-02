# AI Myth Buster

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.8%2B-blue)

**AI Myth Buster** is a project that uses a fine-tuned T5-base transformer model to debunk common myths about artificial intelligence. The project expands a set of curated myth–debunk pairs into a large, diverse dataset, applies curriculum learning, and deploys an interactive web interface using Gradio.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model and Training](#model-and-training)
- [Inference](#inference)
- [Deployment](#deployment)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)

---

## Overview
This project demonstrates how to transform common AI myths into fact-based corrections using a text-to-text transformer (T5-base). The model is fine-tuned on an expanded dataset of 1000 myth–debunk pairs with a curriculum learning approach—starting with "Simple" examples and then introducing "Complex" ones.

---

## Dataset
- **Base Data:**  
  We started with 20 high-quality myth–debunk pairs.
- **Dataset Expansion:**  
  The base examples were expanded to 1000 examples using random variations (e.g., replacing “AI” with “Artificial Intelligence”) and appending expert-support suffix phrases.
- **Complexity Labeling:**  
  Each example is assigned a `"Complexity"` label ("Simple" for the first 500 and "Complex" for the next 500) based on a heuristic or, optionally, using readability scores (e.g., Flesch Reading Ease).

The expanded dataset is saved as `ai_myths_expanded_1000.json` (and further processed to include complexity labels in `ai_myths_expanded_1000_complexity.json`).

---

## Model and Training
- **Model:**  
  We use T5-base, a transformer model that casts all tasks in a text-to-text format. T5 is pre-trained on the C4 dataset using a span-corruption objective.
- **Fine-Tuning:**  
  The model is fine-tuned in two stages:
  1. **Stage 1:** Fine-tuning on "Simple" examples (first 500 entries).
  2. **Stage 2:** Fine-tuning on a combined dataset of both "Simple" and "Complex" examples.
  
  Key hyperparameters include:
  - Learning rate: `3e-5`
  - Batch size: `4` per device
  - Number of epochs: `7` per stage
  - Weight decay: `0.01`
  - Gradient accumulation to simulate a larger batch size
  
  Training is managed using Hugging Face’s Trainer API.

---

## Inference
The inference pipeline uses a **hybrid approach**:
- **Beam Search (Deterministic):**  
  First, the model generates an output using beam search with `num_beams=7` and `do_sample=False` for consistent, high-probability outputs.
- **Fallback Sampling:**  
  If the beam search output is unsatisfactory (e.g., it repeats the myth), the system falls back to a sampling approach (`do_sample=True` with `temperature=0.7`, `top_p=0.85`, and `top_k=50`) to encourage more diverse outputs.
- **Post-Processing:**  
  The system removes any unwanted prompt artifacts and uses a default response if necessary.

---

## Deployment
The project uses Gradio to create a simple web interface:
- A single text input for entering a myth.
- A text output displaying the debunked fact.
- You can launch the interface directly in Colab or deploy it to Hugging Face Spaces for a public demo.

---

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Julie-max/AI-Myth-Buster-Bot.git
   cd AI-Myth-Buster
   ```
2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the pre-trained model from Google Drive:**
    Since GitHub has a 100MB file size limit, we have stored the trained model on Google Drive.
    - Download the model from: [Google Drive Link](https://drive.google.com/file/d/1-DaOqVK9dvDQgLzCXcmRkLMmcNtjASLm/view?usp=drive_link)
    -Extract the model into the models/ directory:
    ```
    AI-Myth-Buster/
    ├── models/
    │   ├── t5_myth_buster/
    │   │   ├── config.json
    │   │   ├── generation_config.json
    │   │   ├── model.safetensors
    │   │   ├── tokenizer_config.json
    │   │   ├── spiece.model
    │   │   ├── special_tokens_map.json
    │   │   ├── added_tokens.json
    ```
---

# Project Structure
```
AI-Myth-Buster/
├── data/                      # Processed dataset files
│   ├── ai_myths_expanded_1000.json
│   ├── ai_myths_expanded_1000_complexity.json
│
├── models/                    # Pre-trained T5 model (downloaded from Google Drive)
│   ├── t5_myth_buster/
│
├── notebooks/                 # Jupyter notebooks for development
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_fine_tuning.ipynb
│   ├── 03_inference_testing.ipynb
│   ├── 04_deployment.ipynb
│
├── src/                       # Python scripts for running tasks
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── inference.py
│   ├── deploy.py
│
├── requirements.txt            # Required dependencies
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
```

---

# Usage
1. **Data Expansion**
    To generate an complexity labelled dataset, run:
    ```bash
    python src/data_preprocessing.py
    ```
    or open `notebooks/01_data_preprocessing.ipynb` and run the cells. 
2. **Fine-Tuning**
    To test on new myths:
    ```bash
    python src/train.py
    ```
    or run `notebooks/02_fine_tuning.ipynb`.
3. **Inference (Testing the Model)**
    To test on new myths:
    ```bash
    python src/inference.py
    ```
    or run `notebooks/03_inference_testing.ipynb`.
4. **Deployment (Gradio Interface)**
    To deploy as a Gradio web app:
    ```bash
    python src/deploy.py
    ```
    or run `notebooks/04_deployment.ipynb`.

---

# Future Improvements
- Improve debunking of complex myths (e.g., humor, stock predictions).
- Increase dataset size for better generalization.
- Explore retrieval-augmented generation (RAG) for fact-checking.

---

# License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🤝 Contributing
Feel free to submit a pull request if you want to improve the project!

---

## 📧 Contact
For any inquiries, reach out to **cbriantjuian@gmail.com** or open an issue on GitHub.

---

### 🌟 If you like this project, give it a ⭐ on GitHub!
