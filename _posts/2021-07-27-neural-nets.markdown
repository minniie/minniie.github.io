---
layout: post
title:  "신경망"
date:   2021-07-27 07:14:00
categories: 신경망
tags: notes
image: /assets/article_images/2021-07-27-neural-nets/wallpaper.JPG
use_math: true
comments: true
---

## Definitions

- input: a sequence of tokens (x_1, x_2, ..., x_n)
- output: a sequence of tokens (y_1, y_2, ..., y_m)
- hidden state: representation of current token
    - he_i: encodes information in (x_1, ..., x_i)
    - hd_j: encodes information in (y_1, ..., y_j)
    - he_n: context vector, feature vector, representation of the input sequence
- y^_j: probability distribution over y_j produced by the model

## RNN

- 의미:
    - text, audio 같이 sequential 한 데이터에 적합
    - 반대로 CNN 은 image 같이 local region 에 정보가 들어있는 데이터에 적합
- encoder:
    - input: input sequence
    - output: he_n (context vector)
    - i-th encoder block
        - input: x_i + he_(i-1)
        - output: he_i
        - function: he_i = tanh(W_hh * he_(i-1) + W_xh * x_i + bias)
            - W_hh, W_xh: same at all i's
            - tanh($\cdot$) $\in$ (-1,1)
- decoder:
    - input: he_n (context vector)
    - output: distribution over output sequence
    - i-th decoder block
        - input: y^\_(j-1) + hd\_(j-1) + he_n
        - output: y^_j + hd_j
        - function: ?

## LSTM

- 의미:
    - 기존 방법의 한계 (RNN):
        - input sequence 가 길어지면 gradient vanishing problem 이 생김
        - 각 레이어에서의 gradient < 1 (항상?) 이기 때문에, 레이어를 거치며 gradient 가 곱해질수록 최종  gradient 의 절댓값이 작아짐
        - 즉, 앞단 레이어의 weight update 가 거의 안됨
    - 해결방법 (LSTM):
        - RNN 의 hidden state 에 cell state 를 추가함
        - cell state는 gradient 가 잘 전파됨 (?)
        - GRU: LSTM 과 비슷한 성능이지만 구조적으로 단순함

## Bi-LSTM

- 의미:
    - 기존 방법의 한계 (RNN, LSTM):
        - 각 단어를 생성할 때 autoregressive 하게 생성된 h_i 를 참조하기 때문에, 각 단어의 앞 단어들의 의미만 참조함
        - 하지만 문장에서 각 단어는 앞 단어들, 뒷 단어들 모두의 영향을 받음
    - 해결방법 (Bi-LSTM):
        - 각 단어를 생성할때, forward LSTM 으로 만들어진 앞 단어들의 context 와, backward LSTM 으로 만들어진 뒷 단어들의 context 를 모두 참조함
        - function:
            - y^_j = softmax(linear([f_LSTM(y_(j-1)) | back_LSTM(y_(j-1))]))
            - 두 context 를 concat 하고 vocab space 로 projection 한 후 softmax 를 취함

## RNN with Attention Decoder

- 의미
    - 기존 방법의 한계 (RNN, LSTM):
        - encoder 에서 하나의 context vector 에 input sequence 전체의 의미를 담고자 함
        - 하지만 input sequence 가 길어지는 경우, 하나의 context vector 에 의미가 충분히 담기지 못함
        - 또한, 각 단어는 모든 단어로부터 정확히 같은 영향을 받지 않고, 특정 단어에 더 영향을 받음
    - 해결방법 (Attention):
        - 각 단어를 생성할때 input sequence 전체를 한꺼번에 참조하는 것이 아니라, 특정 단어를 더 참조하도록 함
        - 즉, 각 단어를 decoding 할때 context vector 을 다르게 함
        - attention score 을 계산하여, encoder 의 어느 hidden state 에 더 attention 을 줄것인지 결정
- encoder: same as RNN
- decoder:
    - input: he_1, he_2, ..., he_n
    - output: distribution over output sequence
    - i-th decoder block
        - input: y^_(j-1) + hd_(j-1) + he_1, he_2, ..., he_n
        - output: y^_j + hd_j
        - function:
            - s_i = he_i $\boldsymbol{\cdot}$ hd_(j-1)
            - sfmax_i = softmax(s_1, ..., s_n)
            - hd_j = $\sum_{i}$ sfmax_i * he_i
        - 특징:
            - 결국 각 input token 의 context 와 현재 시점의 token 의 context 가 얼마나 비슷한지가, 각 input token 의 context 를 얼마나 참조하여 다음 시점의 token 을 생성할지를 결정 (비슷할수록 많이 참조한다)
            - key == value

## Transformer

- 의미:
    - 기존 방법의 한계 (-):
        - 가정: 문장 = 단어들의 sequence
        - each encoder processes each input token, thus sequential processing
        - running time = O(n)
    - 해결방안 (Transformer):
        - 가정: 문장 = 단어들의 합
        - each transformer encoder processed all input tokens at once, thus parallel processing (?)
        - running time < O(n) (= C?)
- encoder:
    - encoder block:
        - 0) x_i = x_i + p_i
        - 1) self-attention
            - input:
                - 1st encoder block: x_1, ..., x_n
                - other encoder blocks: he_1, ..., he_n of previous encoder block
            - output: he'_1, ..., he'_n
                - with residual & layer normalization: he'_1, ..., he'_n = layer_norm(x_1+he'_1, ..., x_n+he'_n)
            - function:
                - q_i = W_q $\boldsymbol{\cdot}$ x_i
                - k_i = W_k $\boldsymbol{\cdot}$ x_i
                - v_i = W_v $\boldsymbol{\cdot}$ x_i
                - he'_k = $\sum_{i}$ sfmax(q_k $\boldsymbol{\cdot}$ k_i / sqrt(d_he)) * v_i
            - 특징:
                - 의미: 각 input token 의 representation 을 만들때 나머지 input tokens 의 의미를 개별적으로, 각각 다른 정도로 참조한다
                - 각 current token query vector 에 대해서, 나머지 input token key vectors 와 얼마나 비슷한지 similarity score 을 측정하고, 그 similarity score 로 input token value vectors 를 weigh 한것이 각 current token 의 새로운 hidden state 임 (비슷할수록 많이 참조한다)
                - O(C): 각 input token 에 대해 parallel computing 가능
                - 나머지 input token 의 key, value vectors 를 참조해야 하기 때문에 dependency 는 있음 (즉, input sequence 에 token 이 추가될 경우 출력의 he'_1, ..., he'_n 값이 변함)
            - multi-headed self-attention
                - multiple representation subspaces
                - he'_k = W_o  
                $\boldsymbol{\cdot}$ [he'1_k | he'2_k | ... | he'8_k]
        - 2) feed forward neural network
            - input: he'_1, ..., he'_n
            - output: he_1, ..., he_n
                - with residual & layer normalization: he_1, ..., he_n = layer_norm(he'_1+he_1, ..., he'_n+he_n)
            - function:
                - he_1, ..., he_n = FC2(FC1(he'_1, ..., he'_n))
            - 특징:
                - 의미: 각 input token 의 representation 을 독립적으로 변형하여 feature 을 뽑는다
                - hidden state 를 larger dimension 으로 projection 했다가 다시 기존 dimension 으로 projection 한다 (왜?)
                - O(C): 각 input token 에 대해 parallel computing 가능
                - 나머지 input token 에 dependency 도 없음 (즉, input sequence 에 token 이 추가되어도 출력의 he'_1, ..., he'_n 값이 변하지 않음)
        - parameters:
            - (W_q, W_k, W_v) * 8, W_o, FC1, FC2 per encoder block
            - encoder blocks do not share weights
- decoder:
    - decoder block:
        - 1) masked self-attention
            - input:
                - 1st decoder block: y^_k + y_bos, y_1, ..., y_(k-1)
                    - training: y_k + y_bos, y_1, ..., y_(k-1)
                    - inference: y^_k + y_bos, y^_1, ..., y^_(k-1)
                - other decoder blocks: hd_k + hd_1, ..., hd_(k-1) of previous decoder block
            - output: hd'_k
                - with residual & layer normalization: hd'_k = layer_norm(y^_k+hd'_k)
            - function:
                - q_j = W_q $\boldsymbol{\cdot}$ y_j
                - k_j = W_k $\boldsymbol{\cdot}$ y_j
                - v_j = W_v $\boldsymbol{\cdot}$ y_j
                - hd'_k = $\sum_{j \in (1,k)}$ sfmax(q_k $\boldsymbol{\cdot}$ k_j / sqrt(d_hd)) * v_j
            - 특징:
                - 의미: 각 생성될 token 의 representation 을 찾을때 (?), 앞에 생성된 token 들의 representation 을 개별적으로, 각각 다른 정도로 참조한다
                - 각 current token query vector 에 대해서, previous token key vectors 와 얼마나 비슷한지 similarity score 을 측정하고, 그 similarity score 로 previous token value vectors 를 weigh 한것이 각 current token 의 새로운 hidden state 임 (previous output 의 단어 중 비슷할수록 많이 참조한다)
                - parallel computing 불가능
                - 구현상, 각 생성될 token 에 대해 나머지 token 과의 similarity score 을 계산한 후 뒤 tokens 에 해당하는 score 들은 -inf 로 세팅한 후에 softmax 를 취해서 최종 weight 를 구한다
        - 2) encoder-decoder attention
            - input: hd'_k + k_1, ..., k_n + v_1, ..., v_n
                - k_i = W_k $\boldsymbol{\cdot}$ he_i of last encoder block
                - v_i = W_v $\boldsymbol{\cdot}$ he_i of last encoder block
            - output: hd''_k
                - with residual & layer normalization: hd''_k = layer_norm(hd'_k+hd''_k)
            - function:
                - q_k = W_q $\boldsymbol{\cdot}$ hd'_k
                - hd''_k = $\sum_{i}$ sfmax(q_k $\boldsymbol{\cdot}$ k_i / sqrt(d_hd)) * v_i
            - 특징:
                - 의미: 각 생성될 token 의 representation 을 찾을때, 모든 input token 들의 representation 을 개별적으로, 각각 다른 정도로 참조한다
                - 각 current token query vector 에 대해서, all input token key vectors 와 얼마나 비슷한지 similarity score 을 측정하고, 그 similarity score 로 all input token value vectors 를 weigh 한것이 각 current token 의 새로운 hidden state 임 (input sequence 의 단어 중 비슷할수록 많이 참조한다)
        - 3) feed forward neural network
            - input: hd''_k
            - output: hd_k
                - with residual & layer normalization: hd_k = layer_norm(hd''_k+hd_k)
            - function:
                - hd_k = FC2(FC1(hd''_k))
            - 특징:
                - 의미: 각 생성될 token 의 representation 을 독립적으로 변형하여 feature 을 뽑는다
        - 4) projection to vocab space
            - input: hd_k
            - output: y^_k
            - function: y^_k = softmax(Linear(hd_k))
                - Linear($\cdot$) = embedding matrix
                - greedy decoding: y^_k = argmax(Linear(hd_k))
                    - 단점: 매번 deterministic 하게 같은 문장이 나옴
                - beam search: y^ = argmax($\prod_{i=1}^{m}$top k of Linear(hd_k))
                    - 단점: 문장 전체의 확률이 가장 높은 애를 선택하므로 "Thank you" 같은 generic 문장만 계속 생성할수 있음
                - random sampling: y^_k = softmax(Linear(hd_k))
                - top k sampling: y^_k = softmax(top k of Linear(hd_k))
                    - 단점: Linear(hd_k) == broad distribution 인 경우 diverse 한 단어가 정답일수 있는 상황에서 억지로 top k 만 보게 됨
                - top p sampling(nucleus sampling): y^_k = CDF(softmax(Linear(hd_k))) < p
                    - dynamic top k since we don't need to define k
                    - 단점: ?
                - temperature sampling: y^_k = softmax(Linear(hd_k) / T)
                    - T == 0: argmax
                    - T == $\infin$: uniform
        - parameters:
            - (W_q, W_k, W_v) * 8, W_o, MLP1, MLP2 per decoder block + W_v
            - decoder blocks do not share weights

## GPT

- 의미:
    - 기존 방법의 한계 (-):
    - 해결방안 (GPT):
        - attention layer 을 깊이 쌓아서 더 complex 한
        - 데이터 양 늘리기
        - decoder block (layer) 수 늘리기
            - small, medium, large, xl: 12, 24, 36, 48
        - decoder dimension 늘리기
            - small, medium, large, xl: 768, 1024, 1280, 1600
        - attention head 수 늘리기
            - small, medium, large, xl: 12, 16, 20, 25
- decoder:
    - 1) masked self-attention
    - 2) feed forward neural network
- BPE(byte pair encoding):
    - 데이터에서 가장 많이 등장한 sequence of characters 을 merge
    - 데이터 압축

## BERT

- 의미: attention layer 을 깊이 쌓아서 더 complex 한 representation 만들기
- encoder:
- wordpiece:
