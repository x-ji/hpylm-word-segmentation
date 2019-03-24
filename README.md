# HPYLM Word Segmentation

<!-- Note: The old code is in `julia-hpylm`. However, it is very inefficient and still contains some errors. The new code currently being worked upon (which tries to properly implement the infinite-gram character-level HPYLM model) is in the folder `julia-nhpylm`. -->

- The folder `julia-hpylm` contains some old code, which is obsolete.
- The folder `julia-nhpylm` contains the Julia implementation of the model. For details on running the program, please refer to the README.md file in that folder.
- The folder `rust-nhpylm` contains the Rust implementation of the model. For details on running the program, please refer to the README.md file in that folder.

## Motivation, method, hypotheses

Teh (2006) proposed a Bayesian language model based on hierarchical Pitman-Yor process, which can be used to construct n-gram models effectively. The Pitman-Yor process is able to produce power-law distributions which are observed in natural languages. The model approximates interpolated Kneser-Ney smoothing for n-gram models.

Mochihashi et al. (2009) applied the model on Chinese/Japanese text segmentation by essentially treating a word as an n-gram of individual characters. They reported significantly better results than previous unsupervised segmentation of Chinese and Japanese. They also reported being able to easily modify the model to incorporate elements of semi-supervised or completely supervised learning, which further improved accuracy.

Granted, the state-of-the-art results on language modeling and word segmentation are still achieved by supervised learning methods, just as is the case with many other tasks. However, unsupervised learning methods could still be interesting, especially on languages for which transcriptions/gold standard data are lacking or inherently harder to obtain.

This project attempts to implement the model as described by Mochihashi et al. (2009). Besides testing on data from the languages mentioned in the original paper, attention will also be paid on testing other languages, as well as potentially incorporating supervised learning methods into the model.

<!-- The minimum expectation would be to successfully build the hierarchical Bayesian model as described by Teh (2006). Testing would be performed at least on the publicly available AP News data, which is originally used by Teh, and the Brown corpus and the State of the Union corpus, which are used by Dr. Dyer and his colleagues in their testing. Additional testing on different languages might also be performed to observe differences in performances. -->

<!-- Currently, the plan is to first understand the implementation by Victor Chahuneau and Dr. Chris Dyer, and then write my own implementation in another language. The implementation is currently being done in Rust. -->

## Relevant literature

- Teh (2006) [
  A hierarchical Bayesian language model based on Pitman-Yor processes
  ](https://dl.acm.org/citation.cfm?id=1220299)

- Mochihashi et al. (2009) [
  Bayesian unsupervised word segmentation with nested Pitman-Yor language modeling
  ](https://dl.acm.org/citation.cfm?id=1687894)

## Data

- Data

  - [AP News](https://ibm.ent.box.com/s/ls61p8ovc1y87w45oa02zink2zl7l6z4)
  - [Brown & State of the Union corpus](http://demo.clab.cs.cmu.edu/cdyer/dhpyplm-data.tar.gz)
  - [the 4th SIGHAN workshop](http://sighan.cs.uchicago.edu/bakeoff2005/)