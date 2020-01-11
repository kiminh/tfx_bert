# 如何使用 TFX 将官方 BERT 模型运行为基于Docker的RestFUL服务

TFX即TensorFlow Extended是官方提供的部署方案（[https://www.tensorflow.org/tfx](https://www.tensorflow.org/tfx)）

它可以很方便的把已经保存了的TensorFlow有其实TF2的模型，转变为可用的服务，并提供一定的性能保障。

下文以如何将官方的中文BERT模型部署为RestFUL服务为例

下载官方bert分词器

```bash
$ wget https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
```

下载官方中文bert模型（TF2版本）

```bash
$ wget -O 1.tar.gz https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/1?tf-hub-format=compressed

```

解压模型到当前目录`bert/1`的路径

```bash
$ mkdir bert
$ mkdir bert/1
$ cd bert/1/
$ tar xvzf ../../1.tar.gz
$ cd ../..
```

启动Docker镜像，开启RestFUL服务。

以下命令中：

- -p 为将Docker内部的8501端口映射到主机的8500端口
- -v 是把当前路径下的bert目录，映射为容器中的/models/bert陌路
- -e 为设置一个环境变量，告诉TFX当前的模型名
- -it 为一次性交互型运行，如果是服务可以写 -d

```bash
$ docker run \
  -p 8500:8501 \
  -v ${PWD}/bert:/models/bert \
  -e MODEL_NAME=bert \
  -it tensorflow/serving
```

用CURL进行预测测试

```bash
$ curl -XPOST http://localhost:8500/v1/models/bert:predict \
  -H 'content-type: application/json' \
  -d '{
  "instances": [
    {
      "input_word_ids": [1, 1, 1],
      "input_mask": [1, 1, 1],
      "input_type_ids": [0, 0, 0]
      }
  ]
}'
```

Python的测试

```python

import requests
import numpy as np
import tensorflow_hub as hub
import tokenization

# 下面这部分主要是为了加载分词器
bert_layer = hub.KerasLayer('./bert/1')
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

# 句子转换
sent1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('我爱你'))
sent2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('我恨你'))

# RestFUL调用
url = 'http://localhost:8500/v1/models/bert:predict'
data = {
    "instances": [
        {
            "input_word_ids": sent1,
            "input_mask": [1] * len(sent1),
            "input_type_ids": [0] * len(sent1)
        },
        {
            "input_word_ids": sent2,
            "input_mask": [1] * len(sent2),
            "input_type_ids": [0] * len(sent2)
        },
    ]
}
r = requests.post(url=url, json=data)

# pool 句子聚合的结果，即 [CLS] 的结果
# (2, 768)
r2 = [
    x['bert_model']
    for x in r.json().get('predictions')
]

# sequence 句子序列的结果
# (2, 3, 768)
r1 = [
    x['bert_model_1']
    for x in r.json().get('predictions')
]

r1 = np.array(r1)
r2 = np.array(r2)
print(r1.shape, r2.shape)

```
