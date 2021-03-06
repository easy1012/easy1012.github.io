---
layout: post
title:  "Neural Network part1"
date:   2020-07-01
author: 이지원
categories: deepLearning
---

이번에 포스팅하게 된 내용은 deep learning의 기초인 Neural Network부터 시작해 보려고 합니다. 이번 포스팅은 널리 알려진 iris데이터를 예시로 진행하도록 하겠습니다. iris데이터는 Sepal Length 꽃받침 길이, Sepal Width 꽃받침의 너비, Petal Length 꽃잎의 길이, Petal Width 꽃잎의 너비 정보, Species 꽃의 종류의 변수를 가진 데이터입니다. 보통 꽃의 종류를 y값, 나머지 변수를 x값으로하여 분류하는 모델 연습에 사용하고, 일반적인 통계적 모델을 사용해도 잘 분류가 되는 데이터입니다.  

  
신경세포는 dendrite로 받아들인 정보를 cell body에 저장했다가 자신의 용량을 넘어가면 axon을 통해 전달물질을 밖으로 내보내는 구조를 가지고 있습니다. 이러한 구조를 추상화하여 만든 구조를 ANN(Artifitial Neural Network)라고 합니다. ANN도 유사하게 데이터가 들어오면, 들어온 데이터에 따라 가중치를 취해주어 더해주고, 그 과정이 다 끝나면 다음 단계로 보내는 구조를 가지고 있습니다. 두 구조의 역할을 매칭 시켜보면, cell body=노드, dendrite = 입력값, axon= 출력값, Synapse = 가중치로 연결시켜 생각할 수 있습니다.
<img src="https://github.com/easy1012/easy1012.github.io/blob/master/assets/nnpic.jpg?raw=true">

다음으로 ANN의 세부 구조에 대해 알아보겠습니다.  위의 그림으로 보면, x들을 모아,  어떤 함수를 취해 결과값 y를 도출합니다. 우리는 이 함수f를 activaion function이라고 합니다. 그리고, 이 f함수가 한 번 취해진 결과 y를 하나의 layer라고 합니다. 이 layers를 많이 쌓게 되면, 우리가 말하는 딥Deep Nenural Network이라고 부르고, 조금 쌓으면, Shallow Neural Network라고 합니다.  activation 함수에 대해서는 나중에 알아보도록 하겠습니다.

이렇게 그냥 쌓기만 하면, 왜 우리의 데이터를 잘 표현할 수 있는 함수가 만들어 지는 걸까요? 이 이론적 기반은 Universal approximation theorem을 기반으로 하고 있습니다. 이 이론의 대략적인 내용은 적절한 노드를 가진 하나의 hidden layer로 모든 함수를 근사시키는 것이 가능하다라는 내용입니다. 그렇다면, 왜 여러개의 deep layers를 쌓느냐?라는 의문이 생길 수 있습니다. 저는 이러한 부분은 하나의 layer에 큰 차원을 쌓는 것보다 작은 차원을 여러개 쌓는 것이 계산적 효율성이 좋아서 그렇지 않을까란 생각을 합니다.

파이썬 코드와 자세한 내용은 2편에서 이어서 다루도록 하겠습니다.
