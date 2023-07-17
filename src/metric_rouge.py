# rouge-L score로 비교
from kiwipiepy import Kiwi
from rouge import Rouge
kiwi = Kiwi()
kiwi.prepare()
rouge = Rouge()

# 품사 필터링
main = ['NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'MM', 'MAG', 'XR', 'SN'] # 품사 선택: 명사류, 동사, 형용사, 관형사, 일반부사, 어근
def pos_filter(text):
    raw_pos_tagged = kiwi.tokenize(text, normalize_coda=False)
    tag_filter = []
    for token in raw_pos_tagged:
        if token.tag in main:
          tag_filter.append(token.form)
    result = ' '.join(tag_filter)
    return result

def get_rouge(instruction, topic):
    instruction = pos_filter(instruction)
    topic = pos_filter(topic)
    return rouge.get_scores(instruction, topic)[0]['rouge-l']['f'] #f-score로 비교