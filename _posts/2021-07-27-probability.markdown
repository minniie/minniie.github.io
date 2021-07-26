---
layout: post
title:  "확률"
date:   2021-07-27 22:58:00
categories: 확률
tags: featured
image: /assets/article_images/2021-07-26/wallpaper.jpg
---

## Definitions

- sample: 자연의 동일한 조건 내에서 관측한 것
    - sample 1 = 1, sample 2 = 0, sample 3 = 1
- data: sample 의 집합
    - data = {1,0,1}
- event: sample 의 종류
    - event 1 = 1, event 2 = 0
- event space: event 의 집합
    - event space = {1,0}
    - ~ state space
- random variable(확률변수): 자연에서 일어나는 것들을 수학적으로 설명하기 위한 수단
    - variable: 확률변수는 각 사건을 값으로 가진다
    - function: 각 사건에 그 사건이 발생할 확률이 매핑되어 있다
    - P(X=x): X is random variable, x is an event
    - discrete random variable(이산확률변수): event 의 type 이 countable 한 경우
        - countable: finite (동전, 주사위) or countably infinite (integers)
    - continuous random variable(연속확률변수): event 의 type 이 uncountable 한 경우
        - uncountable: uncountably infinite (real number)
    - probability mass function: for discrete X, input: x, output: P(X=x)
        - $\sum_{x \in X} P(X=x) = 1$
        - P(X=x) in (0,1)
    - probability density function: for continuous X, input: x, output: P(X=x)
        - $\int_{x \in X} P(X=x) = 1$
        - P(X=x) might be > 1 if size(event space/interval) < 1
- hypothesis: 변수 or 확률변수의 값을 가정
    - h=p
- hypothesis testing: 가설이 진짜인지 검증 (?)
- distribution(분포): 해당 확률변수의 모든 event와 그에 해당하는 확률값을 정의/요약
    - pmf, pdf 와 동일
- parameter: distribution 이 closed form 일 경우(?) 그 식을 결정하는 값들
- expectation(기댓값):
    - if X == variable, X 의 값을 이미 알고 있음
    - if X == random variable, X 의 값을 모르고 있음
        > 이때 대신에 X 의 기댓값을 알수있음
        > X 가 가지는 모든 event x * event x 가 발생할 확률
    - $E_{X\sim P} (X)$: random variable X 는 probability distribution P 를 따른다
- mean: 해당 확률변수의 기댓값
    - $E(X) = \sum_{x \in X}{x * P(X=x)}$
- variance: 해당 확률변수가 얼마나 퍼져있는지
    - $Var(X) = \sum_{x \in X}{(x-E(X))^2} * P(X=x)$
- bias: 해당 확률변수의 sample mean 과 real/latent mean 이 얼마나 차이나는지 (?)
- probabilistic == stochastic == uncertain
- frequentist probability: p = 특정 사건의 빈도 / 모든 사건들의 빈도
    - 이미 발생한 과거의 사건들을 설명
- bayesian probability: p = 특정 사건의 확실성 (level of certainty)
    - 앞으로 발생할 미래의 사건들을 설명
    - p=1 은 확실히 맞다, p=0 은 확실히 아니다
