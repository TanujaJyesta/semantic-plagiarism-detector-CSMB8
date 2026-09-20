# Explainable Semantic Plagiarism Detection with Internet Source Checking

## 📌 Project Overview

This project focuses on developing an explainable plagiarism detection system that can identify potentially reused content, including paraphrased text, and provide evidence of possible online sources.

Traditional plagiarism detection methods mainly depend on lexical or exact word matching. Such methods can be less effective when the original content is paraphrased using different words while retaining the same meaning.

The proposed research combines lexical similarity and semantic similarity with Internet-source retrieval. The system retrieves potentially relevant web sources, compares the submitted document with the retrieved source content, and provides source-grounded evidence for potentially plagiarized passages.

> The system identifies potential plagiarism for human verification. A similarity result is not treated as proof of academic misconduct.

---

## 🎯 Objectives

The main objectives of this research are:

- To detect both lexical and semantically similar content.
- To identify potentially plagiarized passages from uploaded documents.
- To retrieve potentially relevant sources from the Internet.
- To compare lexical and semantic similarity using complementary methods.
- To investigate different strategies for combining similarity signals.
- To evaluate the effect of web-source retrieval quality on plagiarism detection.
- To provide source-grounded and understandable explanations for detected passages.
- To evaluate robustness across different levels of paraphrasing.

---

## 🔬 Research Question

**Can combining lexical similarity and semantic similarity with Internet-source retrieval improve the detection of paraphrased plagiarism while providing understandable, source-grounded evidence?**

---

## 🧠 Proposed Methodology

The proposed system follows the pipeline:

```text
PDF / DOCX Document
        ↓
Text Extraction
        ↓
Text Cleaning & Sentence Segmentation
        ↓
Passage Construction
        ↓
Internet Source Retrieval
        ↓
Webpage Text Extraction
        ↓
TF-IDF Similarity ─────┐
                       ├──→ Evidence Fusion
SBERT Similarity ──────┘
                       ↓
Potential Plagiarism Decision
                       ↓
Source-Grounded Explanation
