
# Spatiotemporal Non-Uniformity-Aware Online Task Scheduling in Collaborative Edge Computing

🔗 **[Open the Colab Notebook](https://colab.research.google.com/drive/1XXe0Tx1g0j5d6dn0HOuuH65EAKpzjn0F)**  

---

## 📄 Reference Paper
**Title:** Spatiotemporal Non-Uniformity-Aware Online Task Scheduling in Collaborative Edge Computing for Industrial Internet of Things  
**Authors:** Yang Li, Xing Zhang, Yukun Sun, Wenbo Wang, Bo Lei:contentReference[oaicite:0]{index=0}

The paper addresses the **NP-hard task scheduling problem** in Industrial IoT edge computing, where tasks arrive in non-uniform spatial and temporal patterns. To solve this, the authors propose an **online optimization framework** that integrates Lyapunov optimization with graph models.  

A critical contribution of this work is the adoption of **heuristic search algorithms**:  
- They reduce the search space for feasible scheduling decisions.  
- They enable **real-time decision-making** without prior knowledge of future requests.  
- They strike a balance between **delay minimization and cost control**, outperforming traditional methods like DRL or genetic algorithms.  

---

## 🎯 Why Heuristic Algorithms (PSO & HS)?
The task scheduling problem is **NP-hard**, making exact optimization infeasible in real-time IIoT environments. Heuristic methods are used because:

- **Particle Swarm Optimization (PSO)**  
  - Inspired by social behavior of bird flocking.  
  - Efficient in **global exploration** of the solution space.  
  - Used here to **determine category assignments** of edge nodes (source, sink, isolated).  
  - **Implemented by:** *C. B. Harinie*.

- **Harmony Search (HS)**  
  - Inspired by the musical process of improvisation.  
  - Efficient in **fine-tuning local solutions** within constraints.  
  - Used here to **optimize the number of tasks forwarded** once categories are fixed.  
  - **Implemented by:** *K. Kanishthika*.

These two complement each other: PSO efficiently explores possible structures, while HS fine-tunes scheduling decisions within those structures. This combination balances **efficiency** and **solution quality**.

---

## ⚡ Efficiency Gains
- Traditional optimization methods require prior system knowledge and become intractable at scale.  
- The PSO + HS framework reduces search space significantly:
  - PSO narrows feasible node-role assignments.  
  - HS quickly optimizes within that narrowed space.  
- This leads to **lower delay, reduced cost, and faster runtime** compared to baseline algorithms (e.g., DRL, genetic algorithms).  

---

## 🚀 Implementations in the Colab Notebook

- **PSO Implementation – by C. B. Harinie**  
  Applies **Particle Swarm Optimization** to assign roles (source, sink, or isolated) to edge nodes.  
  It provides efficient **global exploration**, narrowing down feasible scheduling structures.  

- **Harmony Search Implementation – by K. Kanishthika**  
  Uses **Harmony Search** to optimize task allocation once roles are assigned.  
  It fine-tunes scheduling decisions for **better cost efficiency and reduced delay**.  

- **Final Integrated Framework – by C. B. Harinie & K. Kanishthika**  
  Combines **PSO + HS**, achieving near-optimal results.  
  This hybrid approach leverages the **exploration power of PSO** and the **exploitation strength of HS**, resulting in **faster, more efficient task scheduling**.  

---

## 📌 Key Takeaway
This repository demonstrates how **heuristic algorithms from Computational Intelligence (PSO and HS)** can be effectively applied to **real-time IIoT edge task scheduling**.  
By combining global exploration with local refinement, the framework offers a **practical, scalable, and efficient solution** to complex optimization challenges in collaborative edge computing.  

---
