# LGMcRec: Large Language Models-augmented Light Graph Model for Multi-criteria Recommendation

<p align="center">
<img src="LGMcRec.png" alt="LGMcRec" width=75%>
</p>

## **Overview**
In the digital era of personalization, multi-criteria recommender systems (MCRSs) have emerged as a critical tool for capturing the multidimensional nature of user preferences by leveraging multiple evaluation criteria rather than relying solely on single overall ratings. Despite their potential, existing MCRSs face significant challenges, including graph sparsity, lack of criterion interdependency modeling, and the underutilization of semantic information for recommendation tasks.

To address these limitations, we present **LGMcRec** (Large Language Models-augmented Light Graph Model for Multi-criteria Recommendation), a novel framework that integrates the strengths of **graph neural networks (GNNs)** and **large language models (LLMs)** to enhance representation learning and recommendation performance in multi-criteria settings.

Key innovations of LGMcRec include:
1. **Tripartite Graph Construction**:
   - Captures **user-item interactions**, **item-criterion associations**, and **criterion interdependencies** within a unified structure, mitigating issues of sparsity and unmodeled correlations.
   
2. **Enhanced Embedding Learning**:
   - Extends the **LightGCN architecture** to learn robust graph-based embeddings.
   - Enriches these embeddings through **semantic alignment** with LLM-generated embeddings derived from textual user and item profiles.

3. **Contrastive Learning Integration**:
   - Bridges the gap between graph-based and LLM-based embeddings by maximizing mutual information between the two embedding spaces, resulting in cohesive and comprehensive user and item representations.

Experimental results on three multi-criteria datasets demonstrate that LGMcRec significantly outperforms state-of-the-art methods, establishing it as a promising solution for modern recommendation tasks. The implementation code and datasets will be made publicly available to ensure reproducibility and facilitate further research in this area.
