# environment

外部接口包，包含ICE、redis、xml、mysql接口文件，以及部分通用计算函数

## 使用前注意
把environment放在你的工程外部

## 使用方法
在工程主文件顶端添加：
```python
import sys
sys.path.append('..')
sys.path.append('../environment')
sys.path.append('../environment/comm_sim')
```
引用方法：
```python
from env.environment.comm_sim.comm_ice import CommIce
from env.environment.comm_sim.comm_xml import CommXML
from env.environment.comm_sim.comm_redis import CommRedis
```
