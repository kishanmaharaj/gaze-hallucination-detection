# Global Attention Bias Module

The following shows a visualization of modelling global attention bias using an ensemble approach. Each model m1,m2,...,m5 computes sentence embeddings for claim and context sentences. The sentence which is most similar to the claim is found by using cosine similarity between the claim embedding and sentence embedding for each model. The context sentence voted as most similar by the majority of models is taken as the output of the global attention model.


![global_attention](https://github.com/kishanmaharaj/gaze-hallucination-detection/assets/16451688/edb08f79-1d70-412c-9285-adb95f34f33e)

