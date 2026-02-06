# 📄 Important papers on Deep Learning

*** Draft by LLM - still under review for correctness ***
---

## 1. **Weight Agnostic Neural Networks**  
📌 **Arxiv:** https://arxiv.org/abs/1906.04358  
**Authors:** Adam Gaier, David Ha  
**Core Idea:**  
This paper asks: *How important are weights versus architecture in neural networks?* The authors propose **Weight Agnostic Neural Networks (WANNs)** — architectures evolved to solve tasks **with almost no learned weights**. Instead of training parameters, they use a **single shared weight value** across all connections and evaluate performance over a range of shared weight settings. By evolving architectures that perform well regardless of the actual weight value, they find networks that can do reinforcement learning control tasks and achieve surprisingly high supervised performance (e.g., MNIST above chance) **without traditional training**. :contentReference[oaicite:0]{index=0}

**Core Contributions:**  
- Formulates a search method that **de-emphasizes learning weights** and instead finds architectures that are *intrinsically capable* of performing tasks. :contentReference[oaicite:1]{index=1}  
- Uses a **single shared weight parameter** to evaluate networks, highlighting the inductive bias of architecture itself. :contentReference[oaicite:2]{index=2}  
- Shows that evolved architectures can outperform chance on supervised tasks and solve control tasks without typical training. :contentReference[oaicite:3]{index=3}  

**Test your understanding:**  
1. What is a **weight agnostic neural network** and how does it differ from standard neural networks?  
2. Why do the authors use **a single shared weight** to evaluate network performance?  
3. What search method is used to evolve network architecture in this work?  
4. How do WANNs challenge the traditional view of learning in neural networks?  
5. What implications does the work have for the role of inductive bias in architecture design?

---

## 2. **Understanding Deep Learning Requires Rethinking Generalization**  
📌 **Arxiv:** https://arxiv.org/abs/1611.03530  
**Authors:** Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals  
**Core Idea:**  
This influential work shows that **standard deep neural networks can perfectly fit random labels and even random noise**, yet still generalize well on real data — contradicting classical statistical learning intuition. The authors systematically explore generalization by training overparameterized models on datasets with randomized labels or inputs, demonstrating that **explicit regularization (dropout, weight decay, data augmentation) is not necessary to explain generalization**. They argue that traditional complexity measures fail to capture how modern deep networks generalize. :contentReference[oaicite:4]{index=4}

**Core Contributions:**  
- Demonstrates that neural nets **can memorize training data with random labels** and still show small generalization error. :contentReference[oaicite:5]{index=5}  
- Shows that **classical forms of regularization** don’t fully explain why deep networks generalize. :contentReference[oaicite:6]{index=6}  
- Argues that understanding generalization in deep learning requires new theoretical frameworks. :contentReference[oaicite:7]{index=7}  

**Test Your understanding:**  
1. What is meant by *fitting random labels* and why is this surprising?  
2. Why do the authors argue that **explicit regularization alone does not explain generalization**?  
3. How do the experiments challenge traditional complexity measures like VC dimension?  
4. What are the implications of this work for understanding why deep learning works well in practice?  
5. How does the phenomenon of overparameterization relate to generalization according to the paper?

---

## 3. **ImageNet-trained CNNs Are Biased Towards Texture; Increasing Shape Bias Improves Accuracy and Robustness**  
📌 **Arxiv:** https://arxiv.org/abs/1811.12231  
**Authors:** Robert Geirhos, Patricia Rubisch, et al.  
**Core Idea:**  
This paper investigates how modern convolutional neural networks (CNNs) trained on ImageNet recognize objects. Contrary to the commonly held belief that CNNs learn *shape*, they find that CNNs are actually **biased toward texture features** rather than shape. By training on a stylized version of ImageNet designed to emphasize shape cues, they significantly increase the shape bias of models, which in turn boosts their robustness and generalization. :contentReference[oaicite:8]{index=8}

**Core Contributions:**  
- Empirically shows that CNNs rely more on **texture cues** than shapes when classifying images. :contentReference[oaicite:9]{index=9}  
- Demonstrates that **increasing shape bias** improves classification accuracy and robustness to distortions. :contentReference[oaicite:10]{index=10}  
- Provides insights into differences between computational models and human perception. :contentReference[oaicite:11]{index=11}  

**Questions to Test Understanding:**  
1. What does it mean for a CNN to be *biased toward texture* rather than shape?  
2. How did the authors manipulate the dataset to evaluate shape vs. texture bias?  
3. Why is shape bias linked to improved robustness?  
4. What does this study imply about CNNs as models of human vision?  
5. How can altering inductive biases affect learned representations?

---

## 4. **Taskonomy: Disentangling Task Transfer Learning**  
📌 **PDF:** http://taskonomy.stanford.edu/taskonomy_CVPR2018.pdf  
**Authors:** Amir R. Zamir, Alexander Sax, William Shen, Leonidas Guibas, Jitendra Malik, Silvio Savarese  
**Core Idea:**  
This paper proposes a **computational framework to understand the relationships between different visual tasks** by constructing a task graph that quantifies how well knowledge learned for one task transfers to another. The core idea is to measure transfer dependencies among a large dictionary of vision tasks (2D, 2.5D, 3D, semantic) and use that structure to create a “taskonomy” map. This map can be used to decide which tasks can most efficiently help other tasks, improving data efficiency in multi-task and transfer learning scenarios. :contentReference[oaicite:12]{index=12}

**Core Contributions:**  
- Provides a **systematic way to model transfer relationships** between tasks. :contentReference[oaicite:13]{index=13}  
- Shows that exploiting task structure can **reduce the amount of labeled data needed**. :contentReference[oaicite:14]{index=14}  
- Offers tools for selecting task supervision policies in practice. :contentReference[oaicite:15]{index=15}  

**Test your understanding:**  
1. What is the “taskonomy” proposed by the authors?  
2. How are transfer dependencies between tasks measured?  
3. Why is understanding task relationships useful for transfer learning?  
4. What does the paper suggest about reusing supervision across tasks?  
5. How can this computational task graph impact how we collect and annotate data?

---

## 5. **Do Vision Transformers See Like Convolutional Neural Networks?**  
📌 **Arxiv:** https://arxiv.org/abs/2108.08810  
**Authors:** Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, Alexey Dosovitskiy  
**Core Idea:**  
This paper analyzes whether **Vision Transformers (ViTs)**, which have shown strong performance on image classification tasks, *operate similarly* to convolutional neural networks (CNNs). By comparing internal representations and attention mechanisms between both types of models, the authors show that ViTs exhibit **more uniform layer representations**, gather information in a fundamentally different way, and preserve spatial information differently than CNNs — revealing architectural and functional differences in how they solve vision tasks. :contentReference[oaicite:16]{index=16}

**Core Contributions:**  
- Provides comparative analysis of **internal feature representations** of ViTs vs. CNNs. :contentReference[oaicite:17]{index=17}  
- Highlights the role of **self-attention and skip connections** in how ViTs process information. :contentReference[oaicite:18]{index=18}  
- Shows that architectural differences lead to different spatial localization and feature propagation behaviors. :contentReference[oaicite:19]{index=19}  

**Test Your understanding:**  
1. What are the main architectural differences between ViTs and CNNs?  
2. How do internal representations in ViTs differ across layers compared to CNNs?  
3. What role does **self-attention** play in ViTs?  
4. Why might ViTs preserve spatial information differently from CNNs?  
5. What implications does this study have for choosing architectures in vision tasks?

---

## 6. **Language Models are Few-Shot Learners (GPT-3)**  
📌 **Arxiv:** https://arxiv.org/abs/2005.14165  
**Authors:** Tom B. Brown, Benjamin Mann, Nick Ryder, et al. (OpenAI)  
**Core Idea:**  
This groundbreaking paper introduces **GPT-3**, a large autoregressive language model with *175 billion parameters*, and demonstrates that scaling up language model size dramatically improves **few-shot learning** performance (i.e., solving tasks from a handful of examples) without task-specific fine-tuning. GPT-3 is evaluated on a wide range of NLP benchmarks and shows strong performance purely by conditioning on natural language prompts and a few examples. :contentReference[oaicite:20]{index=20}

**Core Contributions:**  
- Shows that **massive model scaling** leads to strong performance in few-shot and zero-shot settings. :contentReference[oaicite:21]{index=21}  
- Demonstrates that GPT-3 can perform many NLP tasks without fine-tuning. :contentReference[oaicite:22]{index=22}  
- Highlights how **in-context learning** emerges from large transformer models. :contentReference[oaicite:23]{index=23}  

**Test Your understanding:**
1. What is *few-shot learning* and how does GPT-3 achieve it?  
2. Why does increasing model size improve task-agnostic performance?  
3. What is meant by **in-context learning** in GPT-3?  
4. How does GPT-3 differ from traditional fine-tuned language models?  
5. What limitations or challenges remain despite GPT-3’s performance?

## 7. **The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks**

📌 **Arxiv:** https://arxiv.org/abs/1803.03635  
**Authors:** Jonathan Frankle, Michael Carbin  

### Core Idea / Contribution
This paper shows that within large, randomly initialized neural networks exist **small subnetworks (“winning tickets”)** that—when trained in isolation with their original initialization—can match or exceed the performance of the full model. This challenges the idea that training large networks is essential and highlights the importance of **initialization + architecture**, not just optimization.

### Verification Questions
1. What is a *winning ticket* in the Lottery Ticket Hypothesis?
2. Why is **initialization** crucial for winning tickets?
3. How does pruning interact with training dynamics?
4. What does this imply about overparameterized models?
5. How does this paper challenge standard training assumptions?

---

## 8. **Implicit Bias of Gradient Descent on Linear Convolutional Networks**

📌 **Arxiv:** https://arxiv.org/abs/1806.00468  
**Authors:** Sanjeev Arora, Nadav Cohen, Wei Hu, Yuping Luo  

### Core Idea / Contribution
This paper investigates **implicit regularization**: even without explicit regularizers, gradient descent converges to specific solutions biased by **architecture and optimization dynamics**. The authors show that convolutional structures impose implicit norms that affect generalization.

** Test your understanding:**
1. What is **implicit bias** in optimization?
2. How does gradient descent favor certain solutions?
3. Why do convolutional architectures impose structural bias?
4. How does this differ from explicit regularization?
5. What does this imply for theoretical generalization bounds?

---

## 3. **A Mathematical Theory of Deep Convolutional Neural Networks for Feature Extraction**

📌 **Arxiv:** https://arxiv.org/abs/1606.09539  
**Authors:** Stéphane Mallat  

### Core Idea / Contribution
This paper provides a **theoretical foundation** for why deep convolutional networks work by analyzing them as hierarchical operators that build **invariant and stable representations**. It connects CNNs to wavelets and signal processing, showing that depth enables separation of variations like translation and deformation.

**Test your understanding:**
1. What types of invariances do CNNs encode?
2. Why is depth critical for separating variations?
3. How does this relate CNNs to wavelet transforms?
4. What stability properties are proven?
5. How does this theory complement empirical findings?

---

## 4. **On the Measure of Intelligence**

📌 **Arxiv:** https://arxiv.org/abs/1911.01547  
**Authors:** François Chollet  

### Core Idea
This paper argues that benchmark performance does **not equate to intelligence**. Chollet introduces the concept of **generalization difficulty** and proposes measuring intelligence by how efficiently a system adapts to **novel tasks**, emphasizing abstraction, compositionality, and prior knowledge.

###  Questions
1. Why does benchmark accuracy fail to measure intelligence?
2. What is **generalization difficulty**?
3. How does prior knowledge affect learning efficiency?
4. Why are current datasets insufficient?
5. How does this relate to representation learning?

---
