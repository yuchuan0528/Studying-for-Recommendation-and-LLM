# 召回

## <font color=red>Item MF </font>
* 物品-物品模型，训练模型以得到每个物品的向量表示（n_aids * n_factor）。在测试阶段，对于每个session，可以通过last-n向量的加权求和（越新的物品赋予更高的权重），得到的向量作为查询向量，使用faiss库进行向量的快速最近邻查找，得到推荐物品
* 

## <font color=red>User MF</font>

原代码有误：
如果希望进行流行度惩罚，可以传入coef = 1 / np.log1p(aid_size)
```python
# kaggle_otto2/cand_generator/user_mf/model.py
def forward(self, session, aid, aid_size):
        session_emb = self.session_embeddings(session)
        aid_emb = self.aid_embeddings(aid)
        return (session_emb * aid_emb).sum(dim=1) * aid_size # 不应该乘以aid_size，这不是流行度惩罚。直接删掉或者类似User MF，传入1/aid_size有关的coef 
```

## <font color = red> Word2Vec</font>
* **背景**：Word2Vec 是NLP中一种用于学习“词嵌入”（Word Embeddings）的技术，可以将词汇表中的每一个词都表示成一个低维的向量（词向量），这些向量能够蕴含词语的语义信息。
* **Word2Vec基本思想**：通过训练一个语言模型完成与“上下文预测”相关的任务，得到模型训练完后的副产物：模型参数(这里特指神经网络的权重)，并将这些参数作为输入 x 的某种向量化的表示，这个向量便叫做——**词向量**。主要思想是上下文相似的两个词，它们的词向量也应该相似。大致有两种模型：
  * 1. CBOW(Continuous Bag-of-Word)：以上下文（在window大小内）词汇预测当前词，即用$w_{t-2}w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}$这些上下文的词向量，进行平均等操作进行合并，用合并后的向量去预测 $w_{t}$。**CBOW适合大型数据集、高频词**，因为它将多个上下文词合并成一个向量，对于一个中心词，只需要进行一次前向和反向传播，训练更快。并且平均上下文向量的操作使得它对常见词的表示更稳定。
  * 2. SkipGram：以当前词预测其上下文词汇，即用$w_{t}$去预测$w_{t-2}w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}$。**SkipGram更适合中小型数据集、注重低频词的表示**，因为它为每个 (中心词, 上下文词) 对创建训练样本，能够从有限的数据中提取更多信息（相当于数据增强，并且直接使用中心词（包括罕见词）去预测上下文，使得罕见词的向量也能得到充分的训练和更新（而CBOW容易在平均时抹除低频词的信号）。**但是训练慢**，对于一个中心词和 2 * window 个上下文词，它需要进行 2 * window 次独立的预测和梯度更新计算
  * 3. 如何高效训练？难点在于输出层要对词汇表的每一个词计算softmax，计算量极大
    * (1). Hierarchical Softmax (分层 Softmax)：Softmax 的扁平结构改造成一个二叉树结构（通常是 Huffman 树），词汇表中的每个词对应树上的一个叶子节点。变成二分类的乘积
    * (2). Negative Sampling (负采样)：模型不再预测每个词的概率，而是变成了一系列的二分类问题。详细来讲，对于训练样本将其视为正样本，然后从词汇表中随机抽取几个词作为负样本。模型的目标就变成了区分“哪个是正样本，哪些是负样本”，只更新与正样本和选中的负样本相关的权重，而不是更新整个输出层的权重。
*  **OTTO中的word2vec**:
   *  1. **基本思想**：经常一起出现在同一个session的物品，它们的向量会变得更相似。训练word2vec模型，并提供使用 Skip-gram 算法、CBOW算法两种选择，使用负采样进行高效计算。

<font color=red>代码中的错误</font>
读入参数时设置错误
```python
# 应该从item2vec里读入
self.lr = self.config.yaml.cg.user_mf.get("lr", 0.1)
# Word2Vec 的“上下文窗口”大小，例如会考虑前面5个和后面5个物品
self.window = self.config.yaml.cg.user_mf.get("window", 5)
# sg ==1: 使用 Skip-gram 算法，对低频物品更好；sg == 0:CBOW
self.sg = self.config.yaml.cg.user_mf.get("sg", 1)
# 负采样时的采样个数
self.negative = self.config.yaml.cg.user_mf.get("negative", 20)
# 忽略出现次数低于阈值的物品
self.min_count = self.config.yaml.cg.user_mf.get("min_count", 1)
self.user_min_inters = self.config.yaml.cg.user_mf.get("user_min_inters", 20)
self.gen_cand_topk = self.config.yaml.cg.user_mf.get("gen_cand_topk", 40)

```

代码功能重复
```python

with TimeUtil.timer("prepare data"):
            if seed_type == "seq":
                # 使用最近的k个物品的aid -> 得到对应的整数索引 -> 得到对应的词向量 -> 加权平均作为查询向量
                weights = np.linspace(1.0, 0.2, 17)
                k = len(weights)
                with TimeUtil.timer("get test_aids"):
                    # test sessionごとにitemのlistを取得
                    test_aids_df = (
                        self.data_loader.get_test_df()
                        .sort(["session", "ts"], reverse=[False, True])
                        .with_columns(
                            pl.col("aid")
                            .apply(lambda x: aid2idx[x]) # 转换为整数索引
                            .cast(pl.Int32)
                            .alias("aid_idx")
                        )
                        # 像是代码重复，不需要
                        .join(self.data_loader.get_aid_idx_df(), on="aid")
                        .groupby("session")
                        .agg(pl.col("aid_idx"))
                    )
```

# 特征工程
## user_inter_dow
方法里使用的是使用的是测试集数据 test_df，而不是训练集 train_df。
```python
test_df = self.data_loader.get_test_df().to_pandas()
```

UserItemNum也是对测试集进行的

# 排序
## LightGBM

* **GBDT**
  * GBDT (Gradient Boosting Decision Trees - 梯度提升决策树)是一种基于boosting集成学习思想的加法模型，每次迭代都学习一棵CART树来拟合之前 t-1 棵树的预测结果与训练样本真实值的残差。
  * 更进一步的，将这个“拟合残差”的思想，推广到了任意可微的损失函数上，即“拟合负梯度” ，**用负梯度近似模拟残差**。当损失函数是平方误差时，负梯度就是残差。
  GBDT是一个加法模型，其最终模型由M个基学习器（这里是决策树）相加而成
![alt text](images/image.png)
  * 为了学习这个模型，GBDT采用前向分布算法：每一步只学习一个基学习器，并保持之前的所有学习器固定。为了在函数空间中执行梯度下降，我们需要计算损失函数关于函数F的梯度‘
  
* **LightGBM**是GBDT的一种高效实现，显著提升了训练速度。
    * 主要策略：
    * 1. 梯度单边采样。它发现梯度小的样本（模型已经预测得比较准的）对信息增益贡献较小，因此在构建新树时，它会保留所有梯度大的样本（难样本），并随机采样一部分梯度小的样本，从而在不损失太多精度的情况下减少计算量。
    * 2. Leaf-wise Growth (带深度限制): 按叶子节点生长。传统 GBDT 按层(level-wise)生长树，效率较低（在同一层的所有叶子节点中寻找最佳分裂点，完成这一层所有节点的分裂后，再进入下一层）。LightGBM 默认按叶子(leaf-wise)生长，每次选择能带来最大分裂增益的叶子节点进行分裂，（而是每次都从当前所有叶子节点中，找到那个分裂后能带来最大收益（损失降低最多）的叶子节点进行分裂。）可以更快地降低损失，但需要配合 max_depth 防止过拟合。
    * 3. 基于Histogram的决策树算法。将连续的数据数据离散化为整数，将大规模的数据放在了直方图中。因此只保存特征离散化后的值，内存占用减小。遍历特征值时需要遍历的数据量也更少。（离散化影响精度吗？决策树本来就是弱模型，分割点是不是精确并不是太重要；较粗的分割点也有正则化的效果，可以有效地防止过拟合）
* **学习排序**(Learning to Rank - LTR)，训练模型来对列表 (List) 中的项目 (Items) 进行排序。LTR 任务通常使用专门的排序指标来评估，例如：
  * NDCG (Normalized Discounted Cumulative Gain)：关注排名靠前的相关项目，并对靠前的相关项目给予更高权重。是 LTR 最常用的指标之一。
  * LambdaRank:不直接定义损失函数，而是定义了损失函数的梯度 (Lambda Gradient)。这个梯度的大小与交换一对物品 $(d_i, d_j)$ 会对最终排序指标（如 NDCG）造成多大影响 $(\Delta NDCG_{ij})$ 成正比。通过让 GBDT 拟合这些梯度，模型被引导着去优化 NDCG 指标。


* **XGBOOST**基于预排序方法的决策树算法。
  * 首先，对所有特征都按照特征的数值进行预排序。其次，在遍历分割点的时候用的代价找到一个特征上的最好分割点。最后，在找到一个特征的最好分割点后，将数据分裂成左右子节点。
  * 缺点：空间消耗大。这样的算法需要保存数据的特征值，还保存了特征排序的结果。在遍历每一个分割点的时候，都需要进行分裂增益的计算，消耗的代价大。