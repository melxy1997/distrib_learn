# Label Distribution Learning

PyTorch implement for tools of Label Distribution Learning

## Loss Function

* [x] JS-divergence
  $$
  D(p||q)=\sum\limits_{i=1}^np(x)\log\frac{p(x)}{q(x)}
  $$
  

* [ ] Wasserstein Distance
  $$
  Loss=−\frac{1}{N}\sum_{j=1,zj∼Pz}^{N}
  f_θ(g_ω(z_j))\\ω^∗=arg\min_ω(Loss)
  $$
  

  

## Evaluation Metric

* [ ] Chebyshev

$$
Dis(D, \widehat{D})=\max _{i}\left|d_{i}-\widehat{d}_{i}\right|
$$



* [ ] Clark
  $$
  Dis(D, \widehat{D})=\sqrt{\sum_{i=1}^{c} \frac{\left(d_{i}-\widehat{d_{i}}\right)^{2}}{\left(d_{i}+\widehat{d_{i}}\right)^{2}}}
  $$
  

* [ ] Canberra
  $$
  Dis(D, \widehat{D})=\sum_{i=1}^{c} \frac{\left|d_{i}-\widehat{d}_{i}\right|}{d_{i}+\widehat{d}_{i}}
  $$
  

* [ ] Intersection

$$
S i m {1}(D, \widehat{D})=\sum_{i=1}^{c} \min \left(d_{i}, \widehat{d}_{i}\right)
$$

