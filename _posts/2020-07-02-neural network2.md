
---
layout: post
title:  "Neural Network part2"
date:   2020-07-02
author: 이지원
categories: deepLearning
---

이번에는 이 방법이 어떻게 학습되는지 알아보겠습니다. NN은 각각의 weight를 학습하는 과정입니다. weight들은 어떻게 학습되어야 할까요?

 학습을 하려면 일단 기준이 필요할 것입니다. 분류 문제를 생각해보면, x데이터를 넣었을 때, **우리가 예측한 결과($\hat{y}$)가 실제 y와 최대한 같게** 즉, 최대한 틀린 것이 적게 학습하기를 바랄 것입니다. 이런 기준을 수식으로 표현한 것을 **loss function**이라고 합니다. 우리가 딥러닝으로 이 loss function을 푸는 문제를 한다면, 이 loss function에 들어가야하는 요소를 살펴보면, (x,w,activationfunction,y)가 들어갈 것입니다. 이 중 나머지는 우리가 다 가지고 있거나 정하고 시작하기 때문에, 가지고 있지 않은 변수는 w 하나 입니다. 따라서, 우리는 w를 변화시켜, y \hat{y}를 유사하게 만드는 함수를 찾는 문제를 푸는 것이고, 이를 만족하는 최적의 w를 찾는 문제입니다. 그럼 w를 찾는 문제인 건 알았는데, 이걸 어떻게 학습시킬것인가의 문제가 남았습니다. 임의의 w를 하나하나 대입해서 찾기는 너무 가짓수가 많기 때문입니다. 그래서, 우리는 뒤에서 부터 이 weight를 학습하기 시작합니다. 이 방법을 **backpropogation**이라고 부릅니다. 

 이것이 가능한 이유는 우리가 극대, 혹은 극솟값을 구할 때, 흔히 미분을 통해 구하였습니다. 우리의 문제도 극대 혹은 극솟값을 찾는 문제인데, 따라서, 미분을 통해서 구합니다. 이 때, 함수 안의 함수는 **chain rule**을 통해 구하는데, 우리의 activation안에 activation이 이러한 구조를 취하고 있으며, 이를 통해서 우리는 간단한 곱으로 함수의 미분값을 표현할 수 있게 됩니다. 이것이 우리가 deep learning을 gpu로 돌리는 이유입니다.gpu는 간단한 연산에 특화 되어있기 때문이죠. 그림으로 확인해 보도록 하겠습니다.
<img src ="https://github.com/easy1012/easy1012.github.io/blob/master/assets/irismodel3.jpg?raw=true ">
위의 그림은 irisdata로 hidden layer를 두개 가진 NN모델을 만든 모델입니다. 위의 모델을 기준으로 공부하도록 하겠습니다. x1,x2,x3,x4는 각각 꽃받침 길이,너비, 꽃잎 길이 ,너비입니다. 그리고, 각 노드(x1,x2,x3,x4)를 연결하고 있는 선은 weight들을 의미합니다. 이 때, h1을 구성하는 식을 써보면,
$h_1 =f(w1*x1+w2*x2+w3*x3+w4*x4)$
$h_2 =f(w5*x1+w6*x2+w7*x3+w8*x4)$
$h_3,h_4$도 같은 방식으로 진행됩니다. 여기서 f는 activation function을 의미합니다.
$h_5 = f(w_{17}*h_1+w_{18}*h_2+w_{19}*h_3+w_{20}*h_4)$
$h_6,h_7,h_8$까지 같은 방식으로 진행합니다. 마지막, $\hat{y_1},\hat{y_2},\hat{y_3}$부분은 위와 같은 방식으로 진행하는데, 꽃이 3가지 종류이기 때문에, 어떤 꽃일것 같은지 확률값으로 도출하기 위해 다른 activation function을 이용하여 값을 도출합니다. 이 값을 loss function을 기준으로 하여 진짜 y값과 비교하여, weight들을 학습합니다.

이제 backpropogation에 대해서 확인해 봅시다.$\hat{y_1}$에 연결된 w는 $w_{33},w_{34},w_{35},w_{36}$(갈색)입니다. 다 하기에는 너무 많으니 이중 하나 $w_{33}$ 만 확인해서 봅시다. $k = \partial\hat{y_1}/\partial w_{33} =\partial f(w_{33}*h_5+w_{34}*h_6+w_{35}*h_7+w_{36}*h_8)/\partial w_{33}$을 통해 구할 수 있습니다. 하지만, $h_5,h_6,h_7,h_8$은 앞의 weight들과 관련이 있습니다. 그럼 앞에있는 weight값들도 최적값을 구해야 할텐데, 앞의 weight값을 살펴보면, $w_{17},w_{18},w_{19},w_{20}$(초록색)은 $\partial\hat{y_1}/\partial w_{17} = k*\partial h_5/\partial w_{17}$이런 식으로 미분 값들의 곱으로 표현 될 수 있습니다. 앞의 layer의 weight들도 마찬가지 입니다. 따라서, loss function의 조건을 만족하는 weigth들은 chain rule을 통해 쉽게 계산이 될 수 있고, 뒤에서부터 계산하는 backpropagation을 사용하는 이유가 여기에 있습니다.

파이썬 코드는 part3에서 다루겠습니다.



