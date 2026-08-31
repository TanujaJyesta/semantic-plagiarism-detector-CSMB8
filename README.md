# Explainable Semantic Plagiarism Detection

## 1. Project Overview

Explainable Semantic Plagiarism Detection is an NLP-based system designed to detect plagiarism based on the **meaning of text**, rather than only exact word matching. The system aims to identify paraphrased and semantically similar content using **Sentence-BERT (SBERT)** and provide understandable evidence for the detected plagiarism.

## 2. Objectives

* Detect semantic and paraphrased plagiarism.
* Generate sentence embeddings using Sentence-BERT.
* Calculate semantic similarity using cosine similarity.
* Identify potentially plagiarized text and matching sentences.
* Provide explanations for plagiarism decisions.
* Evaluate the system using standard performance metrics.

## 3. Proposed Methodology

Input Documents
      ↓
Text Preprocessing
      ↓
Sentence Segmentation
      ↓
Sentence-BERT Embeddings
      ↓
Cosine Similarity
      ↓
Plagiarism Detection
      ↓
Explainable Results

The system will preprocess the input documents, generate sentence embeddings using SBERT, compare the embeddings using cosine similarity, and identify semantically similar content based on an experimentally determined threshold. The results will include similarity scores and matching sentences as evidence.

## 4. Technology Stack
  Language: Python
  NLP: Sentence-BERT, Sentence Transformers
  Similarity: Cosine Similarity
  Efficient Search: FAISS (if required)
  Interface: Streamlit
  Database: SQLite (if required)
  Version Control: Git & GitHub
## 5. Plan of Action

1. Conduct literature survey and study the base paper.
2. Collect and analyze suitable plagiarism/paraphrase datasets.
3. Preprocess and prepare the data.
4. Implement baseline plagiarism detection methods.
5. Implement SBERT-based semantic similarity.
6. Develop plagiarism detection and explainability modules.
7. Build the user interface.
8. Test and evaluate the system using Accuracy, Precision, Recall and F1-score.
9. Compare the proposed approach with baseline methods.
10. Document and deploy the final system.

## 6. Expected Outcome

The project aims to develop a system capable of detecting **paraphrased and semantic plagiarism** while providing **evidence and explanations** for its detection results.

