
def get_rouge(instruction, topic):
    instruction = pos_filter(instruction)
    topic = pos_filter(topic)
    return rouge.get_scores(instruction, topic)[0]['rouge-l']['f'] #f-score로 비교