# Agito: Algorithmic Gene-Compression for Integrated Task Operations (AGITO)​

为下游任务提供的算法化基因压缩方案。

Adaptive Genomic-Informed Task-Optimizer (AGITO)​

    —— 为了人类， 为了所有的Agito。 🧬⚡️

该项目模块组成（暂定）：

* ground: 对碱基序列实现嵌入和映射操作，将完整的碱基表达为单一向量
* storm: 对低于阈值的向量进行合并操作
* flame: 实现最终的压缩以及后续验证

-----------------------------------------------------------------
# 2026/1/29 关于《超能力战争》

    “不给你的项目起个名字吗？”
    “我用首字母凑出了一个 Agito。”
    “为什么？”
    “可能是因为……旧平成我只看过 Agito 和 Decade 😅”

半年前的三分钟热度，借用了Agito这个名字，借用了一点点勇气，去面对毕业后的世界；

而半年之后，现实世界却用一纸公告，为我——也为所有曾在深渊中紧握过 Agito 之光的人——按下了名为“续集”的启动键。

二十五年，足够一部作品成为一代人的精神图腾。

我很庆幸，在那个万念俱灰的冬日，让 Agito 成为了我人生中的一个注脚。

现实世界为《假面骑士亚极陀》按下了**重启**。

这个仓库的代码，也终于等到了**重启**的时刻。

![alt text](image.png)


## 📅 开发日志

### notice 📌

----------------------------------------
### Date: 2025-7-5

#### ✅ The work of today

- normalizing all of variant and module name
- coding a draw script to reflect the trend in loss
- now, we finish the coding of embedding, encoder, and attention modules, and now we can get output consisted of tokens which are unorderable and have not space or time relationship.

#### 🧠 aspirations and thinkings

- we use attention mechanism replace max pooling approach to include "平移不变性"（Idk how to spell this word）

#### 🐛 question recording

- noep

#### 📈 明日计划 / TODO

- learn the graph network and study how to use "independent slot token"

```
Slot tokens	
Common in memory-based models	
"Each slot token stores distinct information"
```


----------------------------------------
### Date: 2025-7-11 21:00

#### ✅ The work of today

- we carefully review the method of getting global information in module "setAtten". And delete "MultiAttention".
- Then We executed a simple test showing that the decline speed of loss curve is worse then above test(use MultiAttention).

#### 🧠 aspirations and thinkings

- In my View, this question may be caused by two aspects: Firstly, the database set we used is compiled by ViceMusic(author), and is just used in programmer running fluently and checks if the error is in code. This dataset only contains rows in same label, by the way is not a reasonable dataset. Secondly, it is a simple meaning operator and linear map, no ability for coping with "slot-tokens", and may be just suitable for highly global tokens.

#### 🐛 question recording

- The resuly shown in loss graphs reflects that removing the code of Multi-attention has put down the model's correction.
- However, Multi-attention will mix global information, and we has done it in pooling and fusing(above operation of MA). And I think that it is relative with the simple and unreasonable linear mapping......(in the end step of "Agito-ground").
- So we just "封印" this ability of Agito until we finish the next step of dealing with "slot-tokens" produced by ssm and pooling.

#### 📈 明日计划 / TODO

- nope
- study english
- study "Slot-to-Slot" graph structure auto modeling
- study contrastive study
- By the way, I find a interesting loss function called "香农熵计算信息密度"



# Sharing the message of tech

>
> Maybe these informations can help you in some aspects.
> 
> 📦: Shannon entropy
>

香农熵可以用来判断一段文本中，平均每个位置承载了多少“有用信息”或“新知识”，冗余多、重复多的文本，虽然看上去很长，但平均信息量很低，香农熵就低。
 
"啊啊啊啊啊宝宝你真的是一个可爱的可爱的小蛋糕呜呜呜"

其中有效的信息只有"可爱"。

平均下来每个位置的信息量就非常低，这句话本身的**总信息量**也非常低下

**总信息量＝香农熵\*序列长度**

    如果我有一个只有 4 种符号的序列（比如 DNA 中的 ACGT），那它的最大熵是 log₂(4) = 2 bit；如果是英文字母（26 个），最大熵是 log₂(26) ≈ 4.7 bit。

但是香农熵本质只是判断一个句子能有多少新信息， 换句话说就是每个位置需要多少yes/no的问题才能确定？ 但是不能确定在某个领域下的信息是否有用。


一些关键词的方向

"Region-based DNA compression"

"Functional vs non-functional sequence identification"

"Mask-guided compression"

"DNA sequence segmentation and abstraction"

"Context-aware sequence compaction"

