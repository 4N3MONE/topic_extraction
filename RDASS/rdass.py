from sentence_transformers import SentenceTransformer, util

def compute_RDASS(prediction : str, reference : str, document : str):
    global sbert_for_RDASS
    v_p, v_r, v_d = sbert_for_RDASS.encode([prediction, reference, document])
    s_pr = util.cos_sim(v_p, v_r)
    s_pd = util.cos_sim(v_p, v_d)
    rdass_score = (s_pr + s_pd) / 2
    return rdass_score

if __name__=='__main__':
    sbert_for_RDASS = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    d = '다음 코끼리를 분류합니다.'
    p = '숫자 세기'
    r = '코끼리 분류'

    rdass_score = compute_RDASS(p, r, d)
    print(rdass_score)