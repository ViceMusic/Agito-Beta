# Agito: Algorithmic Gene-Compression for Integrated Task Operations (AGITO)​

为下游任务提供的算法化基因压缩方案。

Adaptive Genomic-Informed Task-Optimizer (AGITO)​

    当​​觉醒的压缩之力​​将基因的混沌转化为信息的秩序，生物学的新纪元便随α到Ω的轨迹降临。
    —— 此名既是算法的宣言，亦是向跨越时空的骑士精神的致敬。 🧬⚡️

算法组成：

* ground
* storm
* flame

-----------------------------------------------------------------



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