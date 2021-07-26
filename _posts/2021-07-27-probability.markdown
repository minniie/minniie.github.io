---
layout: post
title:  "확률"
date:   2021-07-27 22:58:00
categories: 확률
tags: featured
image: /assets/article_images/2021-07-26/wallpaper.jpg
use_math: true
comments: true
line-height: 3.5
---

## Definitions

#### sample: 자연의 동일한 조건 내에서 관측한 것
- sample 1 = 1, sample 2 = 0, sample 3 = 1

#### data: sample 의 집합
- data = {1,0,1}

#### event: sample 의 종류
- event 1 = 1, event 2 = 0

#### event space: event 의 집합
- event space = {1,0}
- ~ state space

#### random variable(확률변수): 자연에서 일어나는 것들을 수학적으로 설명하기 위한 수단
- variable: 확률변수는 각 사건을 값으로 가진다
- function: 각 사건에 그 사건이 발생할 확률이 매핑되어 있다
- P(X=x): X is random variable, x is an event
- discrete random variable(이산확률변수): event 의 type 이 countable 한 경우<br>
<span>&ensp;</span><span style="line-height: 1.0;">○ countable: finite (동전, 주사위) or countably infinite (integers)</span>
- continuous random variable(연속확률변수): event 의 type 이 uncountable 한 경우<br>
<span>&ensp;</span><span style="line-height: 1.0;">○ uncountable: uncountably infinite (real number)</span>

#### probability mass function: for discrete X, input: x, output: P(X=x)
- $\sum_{x \in X} P(X=x) = 1$
- P(X=x) in (0,1)

#### probability density function: for continuous X, input: x, output: P(X=x)
- $\int_{x \in X} P(X=x) = 1$
- P(X=x) might be > 1 if size(event space/interval) < 1 

#### hypothesis: 변수 or 확률변수의 값을 가정
- h=p

#### hypothesis testing: 가설이 진짜인지 검증 (?)
<br>
#### distribution(분포): 해당 확률변수의 모든 event와 그에 해당하는 확률값을 정의/요약
- pmf, pdf 와 동일

#### parameter: distribution 이 closed form 일 경우(?) 그 식을 결정하는 값들
<br>
#### expectation(기댓값):
- if X == variable, X 의 값을 이미 알고 있음
- if X == random variable, X 의 값을 모르고 있음<br>
<span>&ensp;</span><span style="line-height: 1.0;">○ 이때 대신에 X 의 기댓값을 알수있음</span><br>
<span>&ensp;</span><span style="line-height: 1.0;">○ X 가 가지는 모든 event x * event x 가 발생할 확률</span>
- $E_{X\sim P} (X)$: random variable X 는 probability distribution P 를 따른다

#### mean: 해당 확률변수의 기댓값
- $E(X) = \sum_{x \in X}{x * P(X=x)}$

#### variance: 해당 확률변수가 얼마나 퍼져있는지
- $Var(X) = \sum_{x \in X}{(x-E(X))^2} * P(X=x)$

#### bias: 해당 확률변수의 sample mean 과 real/latent mean 이 얼마나 차이나는지 (?)
<br>
#### probabilistic == stochastic == uncertain
<br>
#### frequentist probability: p = 특정 사건의 빈도 / 모든 사건들의 빈도
- 이미 발생한 과거의 사건들을 설명

#### bayesian probability: p = 특정 사건의 확실성 (level of certainty)
- 앞으로 발생할 미래의 사건들을 설명
- p=1 은 확실히 맞다, p=0 은 확실히 아니다

## Properties of random variables

#### Definitions
- A, B: random variables
- a, b: events

#### Joint probability: P(A,B) = P(A$\cap$B) = P(A|B) * P(B) = P(B|A) * P(A)
- B 가 발생하고 A 가 발생할 확률
- A 가 발생하고 B 가 발생할 확률
- event space = {A,B} (cartesian of two random variables)
- Bayes' Rule

#### Conditional probability: P(A|B) = P(B|A) * P(A) / P(B)
- B 가 발생한 상황이 자연에서 새로운 조건으로 주어졌을때, A 가 발생할 확률
- event space = {A}

#### Marginal probability: P(A) = $\sum_{b \in B}P(A, B=b) = \sum_{b\in B}P(A|b)*P(b)$
- 어떤 확률변수 A 의 prior 을 모를때, 또 다른 확률변수 B 의 각 사건의 prior * 각 사건이 주어졌을때 A 가 발생할 likelihood 로 구할수 있다
- 어떤 확률변수 A 의 prior 은 모르고 likelihood 만 아는 경우

#### independence: A and B are independent iff P(A|B) = P(A) and P(B|A) = P(B)
- 어떤 확률변수의 분포가, 다른 확률변수/event 가 조건으로 주어진 상황에서의 분포와 동일하면 두 확률변수는 독립이다

## Distributions of random variables

#### Definitions
- p: parameter of the distribution == 가설 == 분포의 모양을 결정

#### Types
- discrete random variables: bernoulli, binomial, categorical, multinomial
- continuous random variables: gaussian, beta, dirichlet

#### bernoulli distribution (베르누이분포):
- X has bernoulli distribution iff:<br>
<span>&ensp;</span><span style="line-height: 1.0;">○ $P(X=x) = p^x * (1-p)^{1-x}$</span>
        - $P(X=1) = p$
        - $P(X=0) = 1-p$
- Definitions:<br>
<span>&ensp;</span><span style="line-height: 1.0;"><p>○ 확률변수 X: {0,1} 중에 1 이 발생할 확률 = p 인 실험을 1 번 실험 했을때, 1 이 발생하는 횟수<br>
○ 사건 x: 1 이 발생하는 횟수<br>
        - x $\in$ {0,1} (discrete)<br>
○ 가설 p: {0,1} 중에 1 이 발생할 확률</p></span>
        - p 이기 때문에 parameter 1개
- Example:<br>
<span>&ensp;</span><span style="line-height: 1.0;">○ 확률변수 X: {뒷면, 앞면} 중에 앞면이 발생할 확률 = p 인 실험을 1번 실험 했을때, 앞면이 발생하는 횟수</span><br>
<span>&ensp;</span><span style="line-height: 1.0;">○ 사건 x: 앞면이 발생하는 횟수</span><br>
<span>&ensp;</span><span style="line-height: 1.0;">○ 가설 p: {뒷면, 앞면} 중에 앞면이 발생할 확률</span>
- 특징:<br>
<span>&ensp;</span><span style="line-height: 1.0;">○ mean = E(X) = p</span><br>
<span>&ensp;</span><span style="line-height: 1.0;">○ variance  = Var(X) = p(1-p)</span><br>
<span>&ensp;</span><span style="line-height: 1.0;">○ special case of binomial distribution when n=1</span>
        
#### binomial distribution (이항분포):
- X has binomial distribution iff:
    - $P(X=x) = \binom{n}{x} p^x (1-p)^{n-x}$
- Definitions:
    - 확률변수 X: {0,1} 중에 1 이 발생할 확률 = p 인 실험을 n 번 실험 했을때, 1 이 발생하는 횟수
    - 사건 x: 1 이 발생하는 횟수
        - x $\in$ {0,1,...,n} (discrete)
    - 가설 p: {0,1} 중에 1 이 발생할 확률
        - p 이기 때문에 parameter 1개
- 특징:
    - mean = E(X) = np
    - variance = Var(X) = np(1-p)

#### categorial distribution (카테고리분포):
- X has categorical distribution iff:
    - $P(X=x_i) = p_1^{x_1} p_2^{x_2} ... p_k^{x_V}$
        - $P(X=x_i) = p_i$
- Definitions:
    - 확률변수 X: {0,...,V} 중에 i 가 발생할 확률 = p_i 인 실험을 1 번 실험 했을때, i 가 발생하는 횟수
    - 사건 x_i: i 가 발생하는 횟수
        - x_i $\in$ {0,1} (discrete)
    - 가설 p_i: {0,...,V} 중에 i 가 발생할 확률
        - p_1, ... , p_V 이기 때문에 parameter 은 V개
    - 특징:
        - NN 마지막 레이어의 softmax 값 [y1, ..., yV] == [p1, ..., pV]
        - x_i 가 categorical variable 이므로 mean, variance 계산하지 않는다

#### multinomial distribution (다항분포):
- X has multinomial distribution iff:
    - $P(X=x_i) = \frac{n!}{x_1!x_2!...x_V!} p_1^{x_1} p_2^{x_2} ... p_V^{x_V}$
- Definitions:
    - 확률변수 X: {0,...,V} 중에 i 가 발생할 확률 = p_i 인 실험을 n 번 실험 했을때, i 가 발생하는 횟수
    - 사건 x_i: i 가 발생하는 횟수
        - x_i $\in$ {0,1,...,n} (discrete)
    - 가설 p_i: {0,...,V} 중에 i 가 발생할 확률
        - p_1, ... , p_V 이기 때문에 parameter 은 V개
- 특징:
    - x_i 가 categorical variable 이므로 mean, variance 계산하지 않는다

#### gaussian/normal distribution (가우시안분포/정규분포):
- X has gaussian distribution iff:
    - $P(X=x) = N(\mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma} 
      \exp(-\frac{(x-\mu)^2}{2\sigma^2})$
- Definitions:
    - 확률변수 X: 평균 = $\mu$, 표준편차 = $\sigma$ 인 실험을 1번 했을때, 발생하는 값
    - 사건 x: 발생하는 값
    - 가설 $\mu, \sigma$
- 특징:
    - standard gaussian distribution:
        - Z has standard gaussian distribution iff:
            - $P(Z=z) = N(0, 1^2) = \frac{1}{\sqrt{2\pi}}\text{exp}(-\frac{z^2}{2})$
        - $X \sim N(\mu, \sigma^2)$ 를 정규화하면 $Z = \frac{X - \mu}{\frac{\sigma}{\sqrt{n}}} \sim N(0, 1^2)$
        - $P(a < X < b) = P(\frac{a - \mu}{\frac{\sigma}{\sqrt{n}}} < Z < \frac{b - \mu}{\frac{\sigma}{\sqrt{n}}}) = \phi(Z=\frac{b - \mu}{\frac{\sigma}{\sqrt{n}}}) - \phi(Z=\frac{a - \mu}{\frac{\sigma}{\sqrt{n}}})$
        - $\phi(Z=z)$: CDF($N(0, 1^2)$)
    - 동일한 분산을 가진 분포들 중, 정규분포는 maximum entropy 를 가짐
        - 가설 $\theta$ 를 정규분포로 설정하면, minimum prior knowledge 를 가짐

## Statistics

#### Notations:
- $X$: 모집단을 나타내는 확률변수
    - $E(X) = \mu, Var(X) = \sigma^2$ 인 "any" 분포를 따름
- $\bar X$: 샘플의 평균을 나타내는 확률변수
    - $\frac{1}{n}\sum_{i=1}^n x_i$
    - $X$ 에서 뽑힌 n 개의 샘플의 평균

#### 1) Law of large numbers (큰 수의 법칙)
- 정의:
    - If n >> 30, then $E(\bar X) \approx E(X)$
- 의미:
    - 모집단에서 n개의 샘플을 한번 뽑는다
    - 이때 샘플의 수가 크면, 샘플의 평균은 모집단의 평균과 비슷하다

#### 2) Central limit theorem (중심극한정리)
- 정의:
    - if n >> 30, then $\bar X \sim N(\mu, \frac{\sigma^2}{n})$
- 의미:
    - 모집단에서 n개의 샘플을 뽑는 실험을 무한히 반복한다고 가정
    - 그러면 각 실험에서 뽑힌 n개의 샘플의 평균을 어떤 확률변수로 설정할수 있음
    - 이때 샘플의 수가 크면, 모집단이 실제로 어떤 분포를 따르냐에 상관없이, 샘플의 평균은 정규분포를 따른다
    - 결국 실제로 실험을 여러번 하지 않아도 샘플의 평균의 기댓값과 표준편차를 구할수 있다
    - 또한, 하나의 확률변수에서 n개의 샘플을 뽑는 경우 뿐만 아니라, 동일한 분포를 따르는 n개의 독립 확률변수에서 각각 샘플을 뽑는 경우에도 적용된다

#### p-value
- 의미:
    - p: significance probability
        - null hypothesis 와 현재 hypothesis 의 겹치는 정도
    - a: significance level
        - a $\in$ (0,1)
- p < a: null hypothesis 기각
- p > a: null hypothesis 기각 불가
