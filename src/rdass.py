from sentence_transformers import SentenceTransformer, util

class RDASS:
    def __init__(self):
        self.sbert = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
        
    def compute(self, pred :str, ref: str, doc: str):
        v_p, v_r, v_d = self.sbert.encode([pred, ref, doc])
        s_pr = util.cos_sim(v_p, v_r)
        s_pd = util.cos_sim(v_p, v_d)
        rdass_score = (s_pr + s_pd) / 2
        return rdass_score