# DRL学习笔记

***

## 0.几个重要概念

* 马尔可夫性质:
$$P({S_{t + 1}}|{S_t},{A_t}) = P({S_{t + 1}}|{S_1},{A_1},{S_2},{A_2},...,{S_t},{A_t})$$
* 状态转移函数(实际环境决定)：
$${p_t}(s'|s,a) = P(S{'_{t + 1}} = s'|{S_t} = s,{A_t} = a)$$
* 策略函数(需要智能体学习)：
$$\pi (a|s) = P(A = a|S = s)$$
* 奖励：
$${r_t} = r({s_t},{a_t})$$
* 回报：
$${G_t} = \sum\limits_{i = t}^n {{\gamma ^{i - t}}{R_i}} $$
* 动作值函数：
$$Q(s,a) = E[{G_t}|{s_t} = s,{a_t} = a]$$
* 最优动作值函数：
$${Q_*}(s,a) = \mathop {\max }\limits_\pi  Q(s,a)$$
* 状态值值函数：
$$V(s) = E[{G_t}|{s_t} = s]$$
* 蒙特卡洛方法：对于期望的无偏估计，随机抽样以近似期望

***

## 1.贝尔曼方程

### 1.1 状态值贝尔曼方程

$${\upsilon _\pi }(s) = \sum\limits_a {\pi (a|s)\sum\limits_{s',r} {p[s',r|s,a](r + \gamma {\upsilon _\pi }(s'))} } ,\forall s \in S$$

* 推导过程：
![Bellman_1](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/Bellman_1.png)

* 实际意义：给出了当前状态值函数的一个递推形式，可以转化为下一时刻的奖励和下一时刻的状态值函数
![Bellman_2](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/Bellman_2.png)

### 1.2动作值贝尔曼方程

$${Q_\pi }(s,a) = \sum\limits_{s',r} {p[s',r|s,a](r + \gamma \sum\limits_{a'} {\pi (a'|s')} {Q_\pi }(s',a'))} $$

* 根据动作值和状态值函数的关系以及状态值贝尔曼方程可以轻松导出
![Bellman_3](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/Bellman_3.png)

***

## 2.价值学习

### 2.1 SARSA算法

观察动作贝尔曼方程可知，实际可写为期望形式：
$${Q_\pi }(s,a) = {E_{s',a'}}[r + \gamma {Q_\pi }(s',a')]$$
根据蒙特卡洛方法进行估计，实际可以估计为：
$${{\hat y}_t} = {r_t} + \gamma q({s_{t + 1}},{{\tilde a}_{t + 1}})$$

上述估计利用 $r_t$ 和新的状态输入神经网络计算出的 $q({s_{t + 1}},{{\tilde a}_{t + 1}})$ 进行估计，实际上要比 ${q }(s_{t},a_{t})$ 直接计算更为精确，以两者的差作为TD误差。

由此我们可以得到SARSA的梯度计算方法(以差的平方作为损失函数)：
$$\begin{array}{l}
L(\omega ) = \frac{1}{2}{[Q(s,a;\omega ) - y]^2}\\
{\nabla _\omega }L(\omega ) = ({{\hat q}_t} - {{\hat y}_t}) \cdot {\nabla_\omega }Q(s,a;\omega )
\end{array}$$

综上所述可以写出SARSA算法的具体流程：
    （1）观测当前状态 $s_t$ ,以当前策略抽样获得动作 $a_t$ 
    （2）使用价值网络计算出 ${\hat q}_t=q(s,a;\omega_{now})$
    （3）智能体执行完 $a_t$ 后，观察到 $r_t$ 和新的 $s_{t+1}$
    （4）根据新状态继续抽样，得到新动作 ${\tilde a}_{t + 1}$ ，只用于估计，不进行执行
    （5）再次使用价值网络计算出 ${\hat q}_{t+1}=q(s_{t+1},{\tilde a}_{t + 1};\omega_{now})$ ,从而得到：
    $${{\hat y}_t} = {r_t} + \gamma {\hat q}_{t+1}$$ $${\delta_t} = {{\hat q}_t} - {{\hat y}_t}$$
    （6）对价值网络反向传播获得 ${\nabla_\omega }Q(s,a;\omega)$
    （7）最终更新网络参数：
    $${\omega _{new}} = {w_{now}} - \alpha  \cdot {\delta _t} \cdot {\nabla_\omega }Q(s,a;\omega )$$

具有的一定问题：

* 蒙特卡洛方法虽然为无偏估计，但是估计的方差较大
* 算法中存在自举问题，即使用价值网络自身的估计去更新自身的权重

### 2.2 Q学习

Q学习基于最优动作贝尔曼方程，用于近似最优动作函数，算法上的不同在于 ${\hat q}_{t+1}$ 的获取，基于动作最优贝尔曼方程，可以写为：
$${{\hat q}_{t + 1}} = \mathop {\max }\limits_a Q({s_{t + 1}},a;{w_{now}})$$

Q学习与SARSA算法的不同之处具体见表格所示：

|学习类型|与Q的关系|策略类型|是否可以经验回放|
|-|-|-|-|
|Q学习|近似最优动作函数|异策略|可以使用|
|SARSA|近似动作价值函数|同策略|不可以用|

***

## 3.策略学习-AC算法

### 3.1 策略梯度定理

目标函数为状态价值函数的期望：
$$J(\theta ) = E[V(s)]$$

为使目标函数增大，我们使用梯度上升法：
$${\theta _{new}} = {\theta_{now}} + \beta  \cdot {\nabla _\theta }J({\theta_{now}})$$

关键难难点在于目标函数梯度的求取，我们把其中的梯度称之为策略梯度。

策略梯度定理可由状态价值贝尔曼方程导出，鉴于其证明复杂，这里不作深究，下面直接给出：
$${\nabla _\theta }J({\theta _{now}}) = {E_S}[{E_{A\~\pi }}[{Q_\pi }(s,a) \cdot {\nabla _\theta }\ln \pi (a|s;\theta )]]$$

使用蒙特卡罗卡洛方法进行无偏估计：
$$g(s,a;\theta ) = {Q_\pi }(s,a) \cdot {\nabla_\theta }\ln \pi (a|s;\theta )$$

只要求得 $g(s,a;\theta )$ ,就可以对参数进行更新， ${\nabla_\theta }\ln \pi (a|s;\theta )$ 可以通过策略网络的反向传播求得，难点在于动作价值函数 ${Q_\pi }(s,a)$ 未知。

### 3.2 REINFORCE方法

使用蒙特卡罗卡洛方法对动作价值函数 ${Q_\pi }(s,a)$ 进行近似，以实际回报值 $u_t$ 代替动作价值函数：
$$g(s,a;\theta ) = {u_t} \cdot {\nabla_\theta }\ln \pi (a|s;\theta )$$

算法较为简单，此处略去。

### 3.3 actor-critic方法

使用SARSA训练价值网络以近似动作价值函数$ {Q_\pi }(s,a)$，从而可以更新策略网络。
价值网络相当于评委，策略网络相当于演员。
演员根据当前环境和评委给出的回报值做出更新，做出动作。
同时为了避免策略网络的训练是为了迎合评委的喜好，需要同时对价值网络进行更新训练。

![AC_1](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/AC_1.png)

算法流程如下：
    （1）观察当前状态 $s_t$ ,根据策略网络做出 $a_t$ ，并让智能体进行执行
    （2）从环境中观测到奖励 $r_t$ 和新的状态 $s_{t+1}$
    （3）根据策略网络得到新动作 ${\tilde a}_{t + 1}$ ，只用于估计，不进行执行
    （4）让价值网络打分获得 ${\hat q}_t=q(s,a;\omega_{now})$ 和 ${\hat q}_{t+1}=q(s_{t+1},{\tilde a}_{t + 1};\omega_{now})$
    （5）计算TD目标和TD误差：
    $${{\hat y}_t} = {r_t} + \gamma {\hat q}_{t+1}$$ $${\delta_t} = {{\hat q}_t} - {{\hat y}_t}$$
    （6）更新价值网络：
    $${\omega _{new}} = {w_{now}} - \alpha  \cdot {\delta _t} \cdot {\nabla_\omega }q(s_t,a_t;\omega )$$
    （7）更新策略网络：
    $${\theta_{new}} = {\theta _{now}} + \beta  \cdot {{\hat q}_t} \cdot {\nabla _\theta }\ln \pi ({a_t}|{s_t};\theta )$$

### 3.4 使用baseline的AC方法(A2C)

带基线的策略梯度定理如下所示：
$${\nabla _\theta }J({\theta_{now}}) = {E_S}[{E_A}[[{Q_\pi }(s,a) - b] \cdot {\nabla_\theta }\ln \pi (a|s;\theta )]]$$

加入baselien后，策略梯度的期望不发生改变，为了使得策略梯度估计的方差最小，使得训练结果更为好，选取 $b = {V_\pi }(s)$ 。

A2C的训练过程与AC方法有些不同，根本原因在于引入baseline后，价值网络需要提供的状态价值函数的值而非动作价值函数。

* 先对价值网络做一些修正，此时的价值网络需要比较准确的估算状态价值。

利用状态值贝尔曼方程和蒙特卡洛方法可以导出：
$${{\hat y}_t} = {r_t} + \gamma  \cdot \nu ({s_{t + 1}};\omega )$$
TD误差为：
$${\delta _t} = {{\hat \nu }_t} - {{\hat y}_t}$$
梯度更新公式为：
$$\omega  = \omega  - \alpha  \cdot {\delta_t} \cdot {\nabla _\omega }\nu ({s_t};\omega )$$

* 带基线的策略网络的近似策略梯度可以根据动作值贝尔曼方程和蒙特卡洛方法导出。

$$g({s_t},{a_t};\theta ) =  - {\delta _t} \cdot {\nabla_\theta }\ln \pi ({a_t}|{s_t};\theta )$$

梯度更新公式为：
$$\theta  = \theta  + \beta  \cdot g({s_t},{a_t};\theta )$$

* TD误差为价值网络传递给策略网络的信息，内含对于动作值好坏的评判

A2C具体训练流程与AC类似，不同的只有价值网络传递给策略网络的值，这里不作赘述。

### 3.5 异步并行计算的A2C方法(A3C)

采用异步并行的方式，同时设立多个worker节点，每个节点都都有自己独立的运行环境，独立做梯度的计算，随时可以与服务器通信。

虽然客观上算法的收敛速度会因样本的减少而变慢，但是异步并行的工作方式大大提高了节点的利用率，实际所用时间要小于同步方式。

![AC_2](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/AC_2.png)

具体的计算步骤如下所示：

* worker端的计算：每个worker都有独立的环境，本地都有一个策略网络，一个价值网络，一个目标网络，设当前节点的参数为 ${\theta _{now}},{w_{now}},{w^ - }_{now}$
  (1)向服务器发出请求获得最新的参数， ${\theta _{new}},{w_{new}}$
  (2)更新本地的目标网络：
  $${w^ - }_{new} = \tau  \cdot {w_{new}} + (1 - \tau ) \cdot {w^ - }_{now}$$
  (3)重复以下步骤n次（用户设计的超参数），获得累计梯度：

    (a)基于当前状态 $s_t$ 和策略网络作出的动作 $a_t$ ,获得观测到的 $r_t$ 和新状态 $s_{t+1}$
    (b)计算TD目标和TD误差：
    $${{\hat y}_t} = {r_t} + \gamma  \cdot \nu ({s_{t + 1}};w_{new}^ - )$$ $${\delta _t} = \nu ({s_t};{w_{new}}) - {{\hat y}_t}$$
    (c)累计梯度：
    $$g_w^k = g_w^k + {\delta_t} \cdot {\nabla _w}\nu ({s_t};{w_{new}})$$ $$g_w^\theta  = g_w^\theta  + {\delta _t} \cdot {\nabla_\theta }\ln \pi ({a_t}|{s_t};{\theta_{new}})$$ 

* 服务器端的计算：存储有一份模型参数的备份，每当有一个worker节点发送来累计梯度，即进行梯度下降更新参数：
$$\begin{array}{l}
{w_{new}} = {w_{new}} - \alpha  \cdot g_w^k\\
{\theta_{new}} = {\theta _{new}} - \beta  \cdot g_w^\theta
\end{array}$$

### 3.6 DDPG（连续控制）

之前提到的内容动作空间是一个离散的集合，DDPG是目前最常用的连续控制方法。核心思想是将连续的动作空间离散化，这里注意动作空间的自由度不能太高，否则动作数量太多难以训练。

DDPG是AC框架下的连续控制方法。与普通AC方法不同,DDPG的策略网络输出的不是离散动作空间的概率分布值（前面由softmax层输出的概率值），而是确定性的动作；DDPG的价值网络输入不只有状态s，因为其价值网络的评判值只针对于一个确定的动作，所以输入也包括动作a。

DDPG使用异策略进行训练，经验回放的策略与目标策略不同，训练价值网络时从经验回放缓存中取出四元组,训练时策略网络的目标函数是动作价值函数的期望值。

$$J(\theta ) = E[Q(s,\mu (s;\theta );\omega )]$$

最大化动作价值函数的期望实际上是要求最优动作函数，即我们需要:
$$Q(s,\mu (s;\theta );\omega ) = \max {Q_*}(s,a)$$

下面是训练流程：
    1.从经验回放中取出四元组： $({s_j},{a_j},{r_j},{s_{j + 1}})$ ,使用策略网络预测 ${{\hat a}_j}$ 和 ${{\hat a}_{j+1}}$
    2.使用价值网络预测：
    $$\begin{array}{l}
{{\hat q}_j} = q({s_j},{a_j};{\omega _{now}})\\
{{\hat q}_{j + 1}} = q({s_{j + 1}},{{\hat a}_{j + 1}};{\omega_{now}})
\end{array}$$
    注意这里的 ${{\hat q}_j}$ 使用经验缓存中的动作而非策略网络的预测（像DQN，异策略的体现）
    3.计算TD目标和误差：
    $${{\hat y}_t} = {r_t} + \gamma {\hat q}_{t+1}$$ $${\delta_t} = {{\hat q}_t} - {{\hat y}_t}$$
    4.更新价值网络：
    $${\omega _{new}} = {w_{now}} - \alpha  \cdot {\delta _t} \cdot {\nabla_\omega }q(s_t,a_t;\omega )$$
    5.更新策略网络：
    $${\theta_{new}} = {\theta _{now}} + \beta  \cdot {\nabla_\theta }\mu ({s_j};\theta ) \cdot {\nabla_a}q({s_j},{{\hat a}_j};{\omega _{now}})$$

### 3.7 TD3(双延迟确定性策略梯度)

* 高估问题：前面提到的DQN和DDPG都存在的问题，根本原因是由于TD方法无可避免的自举使得最优动作函数被高估，训练效果可能不好

因此我们引入目标网络来克服自举高估带来的问题，新的方法包括四个网络，目标策略网络和目标价值网络的任务是计算TD目标时使用，尽可能减少高估的影响：
$$\begin{array}{l}
{{\hat y}_j} = {r_j} + \gamma  \cdot q({s_{j + 1}},{{\hat a}_{j + 1}};{\omega ^ - })\\
{{\hat a}_{j + 1}} = \mu ({s_{j + 1}};{\theta ^ - })
\end{array}$$

这样做程度还不够，TD3算法又在三个方面做了进一步的改进：

* 截断双Q学习：使用两个价值网络和一个策略网络（双评委？），加上目标网络一共6个网络，取更小的y估计值作为TD目标更新价值网络（尽可能低估？）
* 在动作中加入噪声：
$${{\hat a}^ - }_{j + 1} = \mu ({s_{j + 1}};{\theta ^ - }) + \varepsilon $$
噪声是符合截断正态分布的随机向量
* 降低更新频率：每间隔k轮更新一次策略网络和三个目标网络，以防不可靠的频繁打分带来更坏的结果

以下是TD3算法流程：
    1.分别使用目标策略网络（带噪声）和两个目标价值网络做预测，选取更小的TD目标
    2.使用两个价值网络做预测，分别计算两个TD误差，分别更新两个价值网络
    3.每间隔k轮更新一次策略网络和三个目标网络

![TD3](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/TD3.png)

***