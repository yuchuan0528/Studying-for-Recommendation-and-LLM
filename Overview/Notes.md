1015/10/8

**A COMPREHENSIVE REVIEW OF RECOMMENDER SYSTEMS:  TRANSITIONING FROM THEORY TO PRACTICE**

## 传统推荐方法
* **1. 协同过滤 collaborative filtering (CF)**：基于用户间的相似度（UserCF）或物品间的相似度（ItemCF）做推荐。分为两大类：
    (1) Memory-based CF：从用户的历史行为中计算。分为Usercf与ItemCF
    (2) Model-based CF：例如矩阵分解，因式分解，将用户-物品的矩阵分解到潜在空间
* **2. 基于内容的过滤 content-based filtering (CBF)** 根据用户过去的偏好和物品特征，使用诸如词频-逆文档频率( TF-IDF )、余弦相似度和神经网络等技术进行物品表示来推荐物品。基于物品和用户的相似度。缺点：冷启动问题。CF CBF都面临计算复杂度的限制
* **3. 混合方法 hybrid approaches** 集合以上两种方法，以不同权重将CF CBF的预测结果得分相加

## 先进方法
* **1. 基于图**图神经网络可以利用学习复杂的用户-物品的交互信息，节点代表用户和物品，边代表交互。可以从知识图谱中融合用户和物品特征来解决冷启动问题。
* **2. 基于序列和会话**例如RNN LSTM transformer等。通过同时考虑长期和短期内的用户兴趣，可以处理冷启动问题
  * 1. Sequential RS：考虑用户的历史交互来预测未来的偏好
  * 2. Session-based RS：基于会话，实时推荐短期内的活动
* **3. 基于知识** Knowledge Bases (KB), 特别是 Knowledge Graphs (KG)
* **4. 基于强化学习** 推荐系统本身作为强化学习的智能体，在由用户和物品交互组成的环境中进行学习，不断提高推荐的个性化和相关性。
* **5. 基于大语言模型** Transformer架构及其注意力机制允许高效地处理长序列关系。
* **6. 多模态**多模态是指同时处理分析多种数据类型

## 特殊的推荐系统
* **内容感知推荐系统CARS** 考虑了更多的背景内容信息，例如时间、地点、社交设置和用户行为模式等其他维度，以提供更相关和及时的建议。
* **基于评论的推荐系统** 利用用户产生的评论