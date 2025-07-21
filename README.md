# Transformer Architectures, Fine-Tuning, and Multimodal Learning

## Overview

This project explores transformer-based models across a variety of tasks, including causal language modeling, text classification, translation, and multimodal learning. It covers both custom implementations and applications of pretrained models like BERT, T5, and CLIP. Through hands-on coding in PyTorch and Hugging Face Transformers, this project emphasizes architectural intuition, transfer learning, and prompt engineering.

---

## Tasks Covered

- **Causal Language Modeling**: Implemented a DIY transformer decoder from scratch in PyTorch and trained it on synthetic sequences to perform next-token prediction.
- **Text Classification**: Fine-tuned a pretrained BERT model on a sentiment classification task using the IMDb movie review dataset.
- **Translation**: Applied and fine-tuned the T5 model (Flan-T5) to perform English-to-German translation on a custom dataset of sentence pairs.
- **Multimodal Learning**: Used the pretrained CLIP model to classify image-text pairs and experimented with prompt engineering for zero-shot classification.

---

## Key Learning Objectives

- Understand the inner workings of transformer architectures, including attention mechanisms, positional encoding, and residual connections.
- Implement causal self-attention and masked multi-head attention in PyTorch.
- Train transformers from scratch using teacher forcing and cross-entropy loss.
- Fine-tune large language models (LLMs) like BERT and T5 for downstream tasks.
- Tokenize and preprocess text data using Hugging Face Tokenizers.
- Evaluate models using accuracy, BLEU score (for translation), and qualitative error analysis.
- Leverage CLIP’s image-text embedding space for zero-shot inference.
- Explore prompt engineering techniques to optimize zero-shot performance.
- Reflect on the societal implications and risks of large language models.

---

## Results Highlights

- Trained a transformer from scratch that learned to predict the next token in synthetic sequences with over 90% accuracy.
- Fine-tuned BERT on IMDb with high validation accuracy (~93%) using a small subset of training data.
- Achieved strong translation performance from English to German using Flan-T5, with BLEU scores confirming meaningful sentence-level accuracy.
- Used CLIP to classify images and prompts without any fine-tuning, demonstrating the power of pretrained multimodal embeddings.
- Improved zero-shot classification by refining text prompts and evaluating cosine similarity between text/image embeddings.

---

## Reflection

This project deepened my understanding of the transformer architecture and its real-world applications. By building components from scratch and combining them with state-of-the-art pretrained models, I developed a strong intuition for how LLMs process language and vision. I also learned to critically evaluate the capabilities and limitations of these systems — both technically and ethically. The exploration of multimodal learning and zero-shot transfer underscored how foundational models like CLIP are reshaping the boundaries of generalization.

---

## Contact

For questions or collaboration, reach out to aayushkashyap2018@gmail.com. Thank you!
