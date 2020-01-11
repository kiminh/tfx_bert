
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
