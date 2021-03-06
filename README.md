# Udacity_final_project
# 1.入门资料：(统计自然语言处理基础）

## 全局看：（大概看看就好）中文自然语言处理知识入门与应用 http://bit.baidu.com/Course/detail/id/56.html

### 交叉歧义问题：
```
致毕业`和` `尚未`毕业的同学  —— 校友`和`老师给`尚未`毕业同学的一封信

致毕业`和尚` `未`毕业的同学  —— 本科`未`毕业可以当`和尚`吗
```
### 分词（Word Segmentation）
```
根据词库，将句子切分成有语意的词 -----`致` `毕业` `和` `尚未` `毕业` `的` `同学`

​                             a   `校友` `和` `老师` `给` `尚未` `毕业` `同学` `的` `一` `封` `信` 

​                             b  `本科` `未` `毕业` `可以` `当` `和尚` `吗` 
```
------

### 未登录词问题：
```
                                 `天使` `爱` `美丽` `在线` `观看`

`天使` `爱` `美丽` `_`  `土豆`  `高清`  `视频`      `在线` `直播` `爱` `美丽` `的` `天使`
```
> 天使爱美丽没有输入到词库

### 命名实体识别（Named Entity Recognition）

识别自然语言文本中具有特定意义的实体（人、地、机构、时间、作品等）
```
分词                                    `天使` `爱` `美丽` `在线` `观看`

实体                                `天使` `爱` `美丽` —— 电影：天使爱美丽

实体                       电影：天使爱美丽

​            `天使` `爱` `美丽` `_`  `土豆`  `高清`  `视频`      `在线` `直播` `爱` `美丽` `的` `天使`
```
------

### 结构歧义问题：
```

                评论：房间里还可以欣赏日出

​                          |(观点抽取)

            a 房间里还可以      b 可以欣赏日出
```
### 词性标注（Part-of-Speech Tagging）

为自然语言文本中的每个词汇赋予一个词性（名词、动词、形容词等）

### 依存句法分析（Dependency Parsing）

自动分析句子中的句法成分（主语、俄日语、宾语、状语和补语等成分
```

分词：     root  `房间` `里` `还` `可以` `欣赏` `日出` 

词性：            名    方位 副    动       动     动名

观点：                              `可以` `欣赏` `日出`
```

------

### 词汇语义相似度

西瓜 — 西瓜在语义上更像是呆瓜还是草莓？

- 呆瓜？
- 草莓？

### 词向量与语义相似度

（Word Embedding & Sematic Similarity）
```

​                           西瓜

向量化表示                 0.018...

相似度计算    0.115                      0.325   

向量表示  -0.028...                          0.225...

​        呆瓜                                   草莓
```

------

### 文本语义相似度
```
​                     车头如何放置车牌

向量化表示                 0.018...

相似度计算     0.715                   0.325   

向量表示  -0.028...                           0.225...

​       前牌照怎么装                         如何办理北京牌照
```

### 文本语义相似度（Text Semantic Similarity）

依托全网海量数据和深度神经网络技术，实现文本间的语义相似度计算
