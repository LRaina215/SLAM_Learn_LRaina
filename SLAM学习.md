<<<<<<< HEAD


## 目录

### 	一.Date 5.16 -5.23 周总结  [Click](#一)

### 	二.Date5.23 - Date6.7周总结 [Click](#二)

### 	三.Date 7.7 学习记录 [Click](#三)

### 	四.Date 7.14 ~ 7.19 周总结 [Click](#四)

### 	五.Date7.21 ~ 7.27 周总结 [Click](#五)

### 六.Date7.28 ~ 8.2 周总结 [Click](#六)

---

## Date 5.16 -5.23 周总结  <span id="一"> </span>

### 	**学习内容：**

#### 	**1.《视觉SLAM十四讲：从理论到实践》**

​	第二部分（实践应用）-- 第七讲（视觉里程计）7.1--7.6

![image-20250524004028069](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524004028069.png)

​	通过对该讲目前的学习

​	①掌握了从图像中提取特征点的方法--寻找关键点与描述子。SLAM方案中，为了质量与性能兼得，我们采用提取ORB特征点的方式。主要步骤为，找关键点，对每个关键点计算描述子。

![image-20250524102850225](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524102850225.png)

​	当我们得到两张由于相机的位姿发生变换而拍摄得到的图片后，我们可以通过计算两张图片描述子之间的汉明距离，给两张图片中的特征点进行匹配。

​	![image-20250524102914113](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524102914113.png)

​	然而，为了获取质量更高的特征匹配，我们通常会对一对匹配点间的汉明距离与设定的阈值进行比较，当汉明距离超过三十 且 不大于两倍的所有匹配点间最小汉明距离 时对该对点进行保留。

​	![image-20250524102944750](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524102944750.png)

​	且均已在Ubuntu20.04环境下成功使用代码复现。

​	②对对极几何知识进行了基本学习。对极几何用于对相机运动的运动进行计算（主要解决2D-2D（已知一堆2D点）求解位姿问题），得到的结果为**相机由1处运动到2处发生的旋转**（得到旋转矩阵R）与由1处运动到2处发生的平移（得到平移量t）。**（注意！！！2D-2D通常以第一帧相机的坐标系为参考系，故解出来的R、t均为相对于第一帧相机的相对变化关系。 而3D-2D则不一样，3D-2D以世界坐标系为参考系，解出来的R、t是将世界坐标系变换到相机坐标系的变换关系。）**

​	该部分知识牵涉大量线性代数知识，大量推导过程。要得到R，t，首先需要根据对极约束计算出本质矩阵E或基本矩阵F，再利用E或F求解R与t。其中需要辨析的是，要求得本质矩阵E需要知道三维空间中特征点在两次相机不同位姿下的归一化坐标x1与x2；而要求得基本矩阵则需要知道三维空间中特征点在两次相机不同位姿下的像素坐标p1与p2。要解出R与t，（我倾向于使用本质矩阵E解R、t），需要使用奇异值分解（SVD）方法，从而解出两对不同的R与t，根据特征点相对于相机位置深度为正，可以排除掉三组情况，得到正确的旋转矩阵R与平移量t。

​	当两特征点处于同一平面上时（特征点共面or相机发生纯旋转），此时无法用本质矩阵与基本矩阵解出相机位姿R、t，需要引入单应矩阵H。然后可利用数值法、解析法解R、t（书中未详细提及）。

![image-20250524002523255](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524002523255.png)

​	实际上，上述复杂的数学推导在代码中直接调用OpenCV函数即可。比如求解本质矩阵E，经需要传入特征点的像素坐标p1、p2与相机的内参（焦距与光心距），findEssentialMat函数会自动将p1、p2转换为归一化坐标x1、x2进行计算，得到本质矩阵E。基本矩阵F需要传入特征点的像素坐标p1、p2与八点法（FM_8POINT），findFundamentalMat函数会自动计算出F。求解单应矩阵也是如此，传入特征点p1、p2后传入随机采样一致性（RANSAC）方法求解（RANSAC书中未详细提及，仅仅说优于最小二乘解法，适用于很多带错误数据的情况，能够处理带有错误匹配的数据）。求解R、t也只用调用recoverPose并传入本质矩阵E，特征点坐标p1、p2，光心距、焦距即可。

![image-20250524002544633](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524002544633.png)

​	通过本质矩阵解出R、t后，可以带回对极化约束之中检验x1.t * E * x2 与0的误差（误差＜1e-3则说明解出R、t较为精准）。（E = t^R）

​		![image-20250524002617816](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524002617816.png)

​		均已在Ubuntu中代码复现。

​		③单目相机通过R、t 估计特征点深度 -- 三角测量。得到深度，共分为四步。首先，又对极几何约束进行列式。其次，做成x2的反对称矩阵(x2^)（等价于x2叉乘......）。然后，通过等式左侧为0，且R、t，x1、x2已知解出相机在第一个位姿下特征点的深度s1。最后通过s1，解出相机在第二个位姿下特征点的深度s2。从而该对点由2D坐标通过计算得到3D空间坐标。

![481bdec463c2c256922b6f8c514b6a1](C:\Users\Li\Documents\WeChat Files\wxid_3j6yc6bus6a732\FileStorage\Temp\481bdec463c2c256922b6f8c514b6a1.png)

​		与求解R、t一样，代码实现上无需考虑那么多数学过程，直接调用triangulation函数并传入特征点p1、p2、匹配点对、R、t与用于接受三维点结果的vector<Point3d>容器即可。

​	![image-20250524103119093](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524103119093.png)

​	![image-20250524103151198](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524103151198.png)			

​		均在Ubuntu复现。

​		④了解了PnP与ICP的概念与相关知识。PnP用于利用已知一对2D-3D的点对求解R、t的问题（2D指像素平面上特征点的投影点为2D，而3D指特征点为3D点）。相较于2D-2D，2D-3D对与所需要的点对数由8个及以上锐减至最少3个点对。且最后将2D-3D的点对转化为3D-3D的点对，坐标系也由世界坐标系转换到相机坐标系上。ICP概念利用迭代最近点思想，在3D-3D点对之间，通过迭代的方式最小化两组点云间距离。（具体如何如何运用目前还不清楚。）

​		PnP问题有很多解法，大致分为直接求解、非线性优化求解两类型。其中直接求解包含P3P，直接线性变换(DLT)，EPnP，UPnP。非线性优化求解即为构造最小二乘问题并迭代求解（光束法平差(BA)）。

​		P3P即为其中用三对点对求解2D-3D下R、t的方法。由于2D-3D的特征点坐标系为世界坐标系，而非相机坐标系中坐标，最终通过计算可以得到空间中特征点在相机坐标系下的坐标，同时也得到像素平面上投影点的3D坐标，从而变为3D-3D问题。（后续过程解出R、t还未学到）

​		DLT也是用于处理3D-2D问题，通过一组已知世界坐标系下的特征点P与其投影在像素平面上的坐标p解出由世界坐标系变换到相机坐标系的变换关系R、t。即p = MR。（此处的M为旋转矩阵R与平移量p的增广矩阵），将M解出后，即可通过系列数学方法解出R、t。

​		由于该知识的实践部分还没学到，故没有代码成功展示。

#### 	2.《自动驾驶与机器人中的SLAM技术：从理论到实践》

​	主要对二章进行了学习，而第二章大多为上一本书前六章知识的回顾。

​	![image-20250524014007708](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524014007708.png)

### 	学习感想：

​	1.**前期学习规划错误**，过于想赶进度进入激光SLAM学习。但在经过调研与思考后，其实先完成对视觉SLAM的学习对新手来说会更好，能够为学好激光SLAM打好基础。且后续研究方向包含多传感器融合，故对于视觉SLAM的学习也是不可或缺的。故目前从激光SLAM调头继续学习视觉SLAM部分。

​	2.**数学知识欠缺严重。**大多数数学知识相当于对于我来说完全是新知识，边看书边学，尽管每个数学公式都跟着推了，但对知识的印象仍不深。后续或考虑快速过一遍书目内容？不死磕一个地方的推理与证明，侧重实际代码实现？（代码实现上基本都是调用库函数，不牵扯这么多的数学推导证明）但数学知识也不能直接放弃，快速过完一遍，对视觉SLAM算法有足够了解后或许对众多不熟悉数学知识之间能有重点与非重点之分？然后再返回学习？

​	3.对Eigen3、Pangolin、OpenCV**函数库的调用仍然不熟**，仍然停留在代码模仿的层面。是否要专门学习OpenCV等重要的库函数使用？还是说在实践中慢慢增加对其掌握程度？



### g2o 学习：（参考[SLAM从0到1——6. 图优化g2o：从看懂代码到动手编写（长文） - 知乎](https://zhuanlan.zhihu.com/p/121628349)）

1.图优化中的点是相机位姿，即优化变量（状态变量）。

2.图优化中的边是指位姿之间的变换关系，通常表示误差项。

官方文档中经典的g2o框架：

![img](https://pic3.zhimg.com/v2-7f06dfa0db13584f48d6c56712d94b50_1440w.jpg)

对这个结构框图做一个简单介绍（注意图中三种箭头的含义（右上角注解））：

（1）整个g2o框架可以分为上下两部分，两部分中间的连接点：**SparseOptimizer 就是整个g2o的核心部分。**

（2）往上看，SparseOpyimizer其实是一个[Optimizable Graph](https://zhida.zhihu.com/search?content_id=115158590&content_type=Article&match_order=1&q=Optimizable+Graph&zhida_source=entity)，从而也是一个**超图（HyperGraph）**。

（3）**超图有很多顶点和边**。顶点继承自 [Base Vertex](https://zhida.zhihu.com/search?content_id=115158590&content_type=Article&match_order=1&q=Base+Vertex&zhida_source=entity)，也即OptimizableGraph::Vertex；而边可以继承自[ BaseUnaryEdge](https://zhida.zhihu.com/search?content_id=115158590&content_type=Article&match_order=1&q=+BaseUnaryEdge&zhida_source=entity)（单边）,[ BaseBinaryEdge](https://zhida.zhihu.com/search?content_id=115158590&content_type=Article&match_order=1&q=+BaseBinaryEdge&zhida_source=entity)（双边）或[BaseMultiEdge](https://zhida.zhihu.com/search?content_id=115158590&content_type=Article&match_order=1&q=BaseMultiEdge&zhida_source=entity)（多边），它们都叫做OptimizableGraph::Edge。

（4）往下看，SparseOptimizer包含一个**优化算法部分OptimizationAlgorithm**，它是通过OptimizationWithHessian 来实现的。其中迭代策略可以从**[Gauss-Newton](https://zhida.zhihu.com/search?content_id=115158590&content_type=Article&match_order=1&q=Gauss-Newton&zhida_source=entity)（高斯牛顿法，简称GN）、 Levernberg-Marquardt（简称LM法）、Powell's dogleg 三者中间选择一个**（常用的是GN和LM）。(列文伯格法与高斯牛顿法差别在于H是否+ λI)

（5）对优化算法部分进行求解的时**求解器solver，它实际由BlockSolver组成**。BlockSolver由两部分组成：**一个是SparseBlockMatrix**，它由于求解稀疏矩阵(雅克比和海塞)；**另一个部分是LinearSolver**，它用来求解线性方程 HΔx=−b 得到待求增量，因此这一部分是非常重要的，它可以从PCG/CSparse/Choldmod选择求解方法。



**框架搭建步骤：**

**整体按照结构框图从下到上逐渐搭建（从底层到顶层），共分为六步：**

```cpp
typedef g2o::BlockSolver< g2o::BlockSolverTraits<3,1> > Block;  // 每个误差项优化变量维度为3，误差值维度为1
//另一种写法（更直接，直接创建底层Block求解器）
typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> BlockSolverType;
```

**①创建线性求解器Linear Solver（此处若要解雅可比矩阵or海塞矩阵则选择创建SparseBlockMatrix）**

```cpp
/*************** 第1步：创建一个线性求解器LinearSolver*************************/
Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); 

//另一种写法（更直接，直接创建底层线性求解器）
typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
```

**②创建BlockSolver，传入①中定义的Linear Solver求解器初始化。**

```cpp
/*************** 第2步：创建BlockSolver。并用上面定义的线性求解器初始化**********/
Block* solver_ptr = new Block( linearSolver )
```

**③创建总求解器Solver，并从GN（GaussNewton）/LM(Lervernberg-Marquardt）/DogeLeg选一个作为迭代策略，在传入②中定义的块求解器BlcokSolver初始化。**

```cpp
/*************** 第3步：创建总求解器solver。并从GN, LM, DogLeg 中选一个，再用上述块求解器BlockSolver初始化****/
g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );

//另一种写法（更直接，直接创建总求解器solver并选择优化方法，并传入包含线性求解器LinearSolverType的block求解器BlockSolverType，分别使用独享指针也使其更安全）(②、③步合一)
auto solver = new g2o::OptimizationAlgorithmLevenberg(
	g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
);
```

**④创建图优化的核心：稀疏优化器SparseOptimizer，并传入总求解器Solver初始化，打开调试输出（即在优化过程中输出信息）。**

```cpp
/*************** 第4步：创建图优化的核心：稀疏优化器（SparseOptimizer）**********/
g2o::SparseOptimizer optimizer;     // 图模型
optimizer.setAlgorithm( solver );   // 设置求解器
optimizer.setVerbose( true );       // 打开调试输出
```

**⑤定义图的定点和边，并添加到稀疏优化器SparseOptimizer中（SparseOptimizer.addVertex()  SparseOptimizer.addEdge() ）。**

```cpp
/*************** 第5步：定义图的顶点和边。并添加到SparseOptimizer中**********/
CurveFittingVertex* v = new CurveFittingVertex(); //往图中增加顶点
v->setEstimate( Eigen::Vector3d(0,0,0) );
v->setId(0);
optimizer.addVertex( v );
for ( int i=0; i<N; i++ )    // 往图中增加边
    {
  CurveFittingEdge* edge = new CurveFittingEdge( x_data[i] );
  edge->setId(i);
  edge->setVertex( 0, v );                // 设置连接的顶点
  edge->setMeasurement( y_data[i] );      // 观测数值
  edge->setInformation( Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma) ); // 信息矩阵：协方差矩阵之逆
  optimizer.addEdge( edge );
	}
```

**⑥设置优化参数，开始执行优化（设置迭代次数）**

```cpp
/*************** 第6步：设置优化参数，开始执行优化**********/
optimizer.initializeOptimization();
optimizer.optimize(100);    //设置迭代次数
```



**六步的详细解释：**

**（1）创建一个线性求解器LinearSolver**

这一步中我们可以选择不同的求解方式来求解线性方程 HΔx=−b ，g2o中提供的求解方式主要有：

- LinearSolverCholmod ：使用sparse cholesky分解法，继承自LinearSolverCCS。
- LinearSolverCSparse：使用CSparse法，继承自LinearSolverCCS。
- LinearSolverPCG ：使用preconditioned conjugate gradient 法，继承自LinearSolver。
- LinearSolverDense ：使用dense cholesky分解法，继承自LinearSolver。
- LinearSolverEigen： 依赖项只有eigen，使用eigen中sparse Cholesky 求解，因此编译好后可以方便的在其他地方使用，性能和CSparse差不多，继承自LinearSolver。

可以对照上面程序的代码去看求解方式在哪里设置。

**（2）创建BlockSolver，并用定义的线性求解器初始化**

BlockSolver有两种定义方式：

```cpp
// 固定变量的solver。 p代表pose的维度（是流形manifold下的最小表示），l表示landmark的维度
using BlockSolverPL = BlockSolver< BlockSolverTraits<p, l> >;

// 可变尺寸的solver。Pose和Landmark在程序开始时并不能确定，所有参数都在中间过程中被确定。
using BlockSolverX = BlockSolverPL<Eigen::Dynamic, Eigen::Dynamic>;
```

此外g2o还预定义了以下几种常用类型：

- BlockSolver_6_3 ：表示pose 是6维，观测点是3维，用于3D SLAM中的BA。
- BlockSolver_7_3：在BlockSolver_6_3 的基础上多了一个scale。
- BlockSolver_3_2：表示pose 是3维，观测点是2维。

**（3）创建总求解器solver**

注意看程序中只使用了一行代码进行创建：右侧是初始化；左侧含有我们选择的迭代策略，在这一部分，我们有三迭代策略可以选择：

- g2o::OptimizationAlgorithmGaussNewton
- g2o::OptimizationAlgorithmLevenberg
- g2o::OptimizationAlgorithmDogleg

**（4）创建图优化的核心：稀疏优化器**

根据程序中的代码示例，创建稀疏优化器：

```cpp
g2o::SparseOptimizer  optimizer;
```

设置求解方法：

```cpp
SparseOptimizer::setAlgorithm(OptimizationAlgorithm* algorithm)
```

设置优化过程输出信息：

```cpp
SparseOptimizer::setVerbose(bool verbose)
```

**（5）定义图的顶点和边，并添加到SparseOptimizer中**

看下面的具体讲解。

**（6）设置优化参数，开始执行优化**

设置SparseOptimizer的初始化、迭代次数、保存结果等。

初始化：

```cpp
SparseOptimizer::initializeOptimization(HyperGraph::EdgeSet& eset)
```

设置迭代次数：

```cpp
SparseOptimizer::optimize(int iterations,bool online)
```

————————————————————————

下面专门讲讲第5步：**定义图的顶点和边**。这一部分使比较重要且比较难的部分，但是如果要入门g2o，这又是必不可少的一部分

#### 1. 点 Vertex

在g2o中定义Vertex有一个通用的类模板：BaseVertex。在结构框图中可以看到它的位置就是HyperGraph继承的根源。

同时在图中我们注意到BaseVertex具有两个参数D/T，**这两个参数非常重要**，我们来看一下：

- D 是int 类型，表示vertex的最小维度，例如3D空间中旋转是3维的，则 D = 3（为BaseVertex<,>第一空所填）
- T 是待估计vertex的数据类型，例如用四元数表达三维旋转，则 T 就是Quaternion 类型（为BaseVertex<,>第二空所填）

```cpp
static const int Dimension = D; ///< dimension of the estimate (minimal) in the manifold space

typedef T EstimateType;
EstimateType _estimate;
```

特别注意的是这个D不是顶点(状态变量)的维度，而是**其在流形空间(manifold)的最小表示。**

**>>>如何自己定义Vertex**

在我们动手定义自己的Vertex之前，可以先看下g2o本身已经定义了一些常用的顶点类型：

```cpp
ertexSE2 : public BaseVertex<3, SE2>  
//2D pose Vertex, (x,y,theta)

VertexSE3 : public BaseVertex<6, Isometry3> //Isometry3使欧式变换矩阵T，实质是4*4矩阵
//6d vector (x,y,z,qx,qy,qz) (note that we leave out the w part of the quaternion)

VertexPointXY : public BaseVertex<2, Vector2>
VertexPointXYZ : public BaseVertex<3, Vector3>
VertexSBAPointXYZ : public BaseVertex<3, Vector3>

// SE3 Vertex parameterized internally with a transformation matrix and externally with its exponential map
VertexSE3Expmap : public BaseVertex<6, SE3Quat>

// SBACam Vertex, (x,y,z,qw,qx,qy,qz),(x,y,z,qx,qy,qz) (note that we leave out the w part of the quaternion.
// qw is assumed to be positive, otherwise there is an ambiguity in qx,qy,qz as a rotation
VertexCam : public BaseVertex<6, SBACam>

// Sim3 Vertex, (x,y,z,qw,qx,qy,qz),7d vector,(x,y,z,qx,qy,qz) (note that we leave out the w part of the quaternion.
VertexSim3Expmap : public BaseVertex<7, Sim3>
```

但是！如果在使用中发现没有我们可以直接使用的Vertex，那就需要自己来定义了。一般来说定义Vertex需要重写这几个函数（注意注释）：

```cpp
virtual bool read(std::istream& is);
virtual bool write(std::ostream& os) const;
// 分别是读盘、存盘函数，一般情况下不需要进行读/写操作的话，仅仅声明一下就可以

virtual void oplusImpl(const number_t* update);
//顶点更新函数

virtual void setToOriginImpl();
//顶点重置函数，设定被优化变量的原始值。
```

请注意里面的**oplusImpl函数，是非常重要的函数**，主要用于优化过程中增量△x 的计算。根据增量方程计算出增量后，**通过这个函数对估计值进行调整**，因此该函数的内容要重视。

根据上面四个函数可以得到定义顶点的基本格式：

```cpp
class myVertex: public g2o::BaseVertex<Dim, Type> //初始化g2o库中的自定义顶点BaseVertex，并传入参数优化变量维数(Dimension)与数据类型(Type)
  {
      public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      myVertex(){}

      virtual void read(std::istream& is) {}
      virtual void write(std::ostream& os) const {}

      virtual void setOriginImpl()
      {
          _estimate = Type();
      }
      virtual void oplusImpl(const double* update) override
      {
          _estimate += update;
      }
  };
```

如果还不太明白，那么继续看下面的实例：

```cpp
class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // 字节对齐

    virtual void setToOriginImpl() // 重置，设定被优化变量的原始值
    {
        _estimate << 0,0,0;
    }

    virtual void oplusImpl( const double* update ) // 更新
    {
        _estimate += Eigen::Vector3d(update);   //update强制类型转换为Vector3d
    }
    // 存盘和读盘：留空
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
};
```

另外值得注意的是，优化变量更新**并不是所有时候都可以像上面两个一样直接 += 就可以**，这要**看优化变量使用的类型（是否对加法封闭）**。

**>>> 向图中添加顶点**

接着上面定义完的顶点，我们把它添加到图中：

```cpp
CurveFittingVertex* v = new CurveFittingVertex(); // 创建顶点实例
v->setEstimate( Eigen::Vector3d(0,0,0) )；  // 设定初始值
v->setId(0);                               // 定义节点编号
optimizer.addVertex( v );                  // 把节点添加到图中
```

三个步骤对应三行代码，注释已经解释了作用。

#### 2.边 Edge

图优化中的边：BaseUnaryEdge，BaseBinaryEdge，BaseMultiEdge 分别表示一元边，两元边，多元边。

顾名思义，一元边可以理解为一条边只连接一个顶点，两元边理解为一条边连接两个顶点（常见），多元边理解为一条边可以连接多个（3个以上）顶点。

以最常见的二元边为例分析一下他们的参数：D, E, VertexXi, VertexXj：

- D 是 int 型，表示测量值的维度 （dimension）
- E 表示测量值的数据类型
- VertexXi，VertexXj 分别表示不同顶点的类型

```cpp
BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>
```

上面这行代码表示二元边，参数1是说测量值是2维的；参数2对应测量值的类型是Vector2D，参数3和4表示两个顶点也就是优化变量分别是三维点 VertexSBAPointXYZ，和李群位姿VertexSE3Expmap。

**>>> 如何定义一个边**

除了上面那行定义语句，还要复写一些重要的成员函数：

```cpp
virtual bool read(std::istream& is);
virtual bool write(std::ostream& os) const;
// 分别是读盘、存盘函数，一般情况下不需要进行读/写操作的话，仅仅声明一下就可以

virtual void computeError();
// 非常重要，是使用当前顶点值计算的测量值与真实测量值之间的误差

virtual void linearizeOplus();
// 非常重要，是在当前顶点的值下，该误差对优化变量的偏导数，也就是Jacobian矩阵
```

除了上面四个函数，还有几个重要的成员变量以及函数：

```cpp
_measurement； // 存储观测值
_error;  // 存储computeError() 函数计算的误差
_vertices[]; // 存储顶点信息，比如二元边，_vertices[]大小为2
//存储顺序和调用setVertex(int, vertex) 和设定的int有关（0或1）

setId(int);  // 定义边的编号（决定了在H矩阵中的位置）
setMeasurement(type);  // 定义观测值
setVertex(int, vertex);  // 定义顶点
setInformation();  // 定义协方差矩阵的逆
```

有了上面那些重要的成员变量和成员函数，就可以用来定义一条边了：

```cpp
class myEdge: public g2o::BaseBinaryEdge<errorDim, errorType, Vertex1Type, Vertex2Type>
  {
      public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW      

      myEdge(){}     
      virtual bool read(istream& in) {}
      virtual bool write(ostream& out) const {}      
      virtual void computeError() override
      {
          // ...
          _error = _measurement - Something;
      }    
  
      virtual void linearizeOplus() override  // 求误差对优化变量的偏导数，雅克比矩阵
      {
          _jacobianOplusXi(pos, pos) = something;
          // ...         
          /*
          _jocobianOplusXj(pos, pos) = something;
          ...
          */
      }      
      private:
      data
  }
```

让我们继续看curveftting这个实例，这里定义的边是简单的一元边：

```cpp
// （误差）边的模型    模板参数：观测值维度，类型，连接顶点类型
class CurveFittingEdge: public g2o::BaseUnaryEdge<1,double,CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge( double x ): BaseUnaryEdge(), _x(x) {}
    // 计算曲线模型误差
    void computeError()
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0,0) = _measurement - std::exp( abc(0,0)*_x*_x + abc(1,0)*_x + abc(2,0) ) ;
    }
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
public:
    double _x;  // x 值， y 值为 _measurement
};
```

上面的例子都比较简单，下面这个是3D-2D点的PnP 问题，也就是**最小化重投影误差问题**，这个问题非常常见，使用最常见的二元边，弄懂了这个基本跟边相关的代码就能懂了：

```cpp
//继承自BaseBinaryEdge类，观测值2维，类型Vector2D,顶点分别是三维点、李群位姿
class G2O_TYPES_SBA_API EdgeProjectXYZ2UV : public  
               BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //1. 默认初始化
    EdgeProjectXYZ2UV();

    //2. 计算误差
    void computeError()  {
      //李群相机位姿v1
      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
      // 顶点v2
      const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
      //相机参数
      const CameraParameters * cam
        = static_cast<const CameraParameters *>(parameter(0));
     //误差计算，测量值减去估计值，也就是重投影误差obs-cam
     //估计值计算方法是T*p,得到相机坐标系下坐标，然后在利用camera2pixel()函数得到像素坐标。
      Vector2D obs(_measurement);
      _error = obs - cam->cam_map(v1->estimate().map(v2->estimate()));
    }

    //3. 线性增量函数，也就是雅克比矩阵J的计算方法
    virtual void linearizeOplus();

    //4. 相机参数
    CameraParameters * _cam; 
    bool read(std::istream& is);
    bool write(std::ostream& os) const;
};
```

这个程序中比较难以理解的地方是：

```cpp
_error = obs - cam->cam_map(v1->estimate().map(v2->estimate()));//误差=观测-投影
```

- cam_map 函数功能是把相机坐标系下三维点（输入）用内参转换为图像坐标（输出）。
- map函数是把世界坐标系下三维点变换到相机坐标系。
- v1->estimate().map(v2->estimate())意思是用V1估计的pose把V2代表的三维点，变换到相机坐标系下。

**\>>>向图中添加边**

和添加点有一点类似，下面是添加一元边：

```cpp
// 往图中增加边
    for ( int i=0; i<N; i++ )
    {
        CurveFittingEdge* edge = new CurveFittingEdge( x_data[i] );
        edge->setId(i);
        edge->setVertex( 0, v );                // 设置连接的顶点
        edge->setMeasurement( y_data[i] );      // 观测数值
        edge->setInformation( Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma) ); // 信息矩阵：协方差矩阵之逆
        optimizer.addEdge( edge );
    }
```

**但在SLAM中我们经常要使用的二元边**（前后两个位姿），那么此时：

```cpp
index = 1;
for ( const Point2f p:points_2d )
{
    g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
    edge->setId ( index );  // 边的b编号
    edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
    edge->setVertex ( 1, pose );
    edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );  // 设置观测的特征点图像坐标
    edge->setParameterId ( 0,0 );
    edge->setInformation ( Eigen::Matrix2d::Identity() );
    optimizer.addEdge ( edge );
    index++;
}
```

—————————————————

至此，就介绍完了g2o中的一些框架和实现，需要提醒的是在SLAM中我们常常用到的是二元边以及对应的点，他们都较为复杂，应当多次学习复习实践。



## Date5.23 - Date6.7周总结 <span id="二"> </span>

### 学习内容：

《视觉SLAM十四讲：从理论到实践》

第七章 视觉里程计一：

 PnP部分+ICP部分

（重点对g2o库进行了学习）

第八章 视觉里程计二：

 LK光流法原理 + 单层LK光流法

![696b99575d2a40d2ddf24c55c22fd14](C:\Users\Li\Documents\WeChat Files\wxid_3j6yc6bus6a732\FileStorage\Temp\696b99575d2a40d2ddf24c55c22fd14.png)

#### 一.（易混淆）2D-3D的PnP求解位姿方法与3D-3D的ICP求解位姿方法的区别：

​	**①2D-3D-PnP方法：**

​		3D指已知一组世界坐标系下的3D坐标，2D指该三维点在当相机处于某个位姿时在相机像素平面（相机坐标系）的一组2D坐标。于是我们的目标就是通过**不断调整由世界坐标系转换到相机坐标系的矩阵R与位移t**，**使得这组三维点能够在相机坐标系的二维像素平面中被更加精准表示**。

​		于是我们通过这个过程得到了**由世界坐标系变换到相机坐标系的旋转矩阵R与平移距离t**。

​		PnP中的n代表已知多少组这样的点，常用三组这样的点即可以解出位姿变换的矩阵R与位移t，故称为P3P。

​		故从上述过程中可知，**本质上PnP求解法即为找到误差的最小值**，故可采用高斯牛顿法等非线性优化方法。

​	**②3D-3D-ICP方法：**

​		此时的3D-3D为既知道一组世界坐标系下的点的3D坐标，也知道相机坐标系下该点的3D坐标。于是我们的目的与2D-3D一样，既**调整旋转矩阵R与平移距离t**使得**这个点从世界坐标系变换到相机坐标系下的后的坐标**与原本就在相机坐标系下的点的**位置相差距离最小**。（也称为点云配准）

​		同理我们也得到了R与t。

​		**实际上，ICP和PnP都有多种解法**，但是**我比较倾向于用非线性优化**解决误差最小问题。

#### **二. g2o非线性优化函数库的学习**

​		故在视觉SLAM中我们经常调用g2o库来对非线性优化过程进行计算。

​		g2o的使用大概包括以下过程：

​		**①在调用g2o解决非线性优化问题的函数之前定义所创建的顶点与边的性质：**

​			1.**顶点（Vertex）：用于存放待优化变量（比如上述所讲到的R、t（统称位姿pose））**

​			比如（下面为自定义定点类型的创建方法（如果使用g2o自带的顶点类型，直接声明即可使用））：

```cpp
class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // 字节对齐

    virtual void setToOriginImpl() // 重置，设定被优化变量的原始值
    {
        _estimate << 0,0,0;
    }

    virtual void oplusImpl( const double* update ) // 更新
    {
        _estimate += Eigen::Vector3d(update);   //update强制类型转换为Vector3d
    }
    // 存盘和读盘：留空
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
};
```

​	其中四个virtual为模板，我们创建时仅需更改：

​		函数头:	

​	BaseVertex<优化变量维度，优化数据变量类型>

​		SetToOriginImpl： 	

​	_estimate = 初始化估计值（如=Sophus：：SE3d -- 代表位姿类型（矩阵中同时包含了旋转矩阵R与平移距离t））

​		oplusImpl：	

​	_estimate += 更新值（你想对估计值如何进行更新（如=Sophus::SE3d::exp(update_eigen) * _estimate））

​	

​		**2.边（Edge）：指向一个顶点（优化变量），边的本质是误差项。几元边即连接几个顶点（优化变量），即对几个优化变量求导数or梯度。**

​		自定义边方法实例如下：

```cpp
//继承自BaseBinaryEdge类，观测值2维，类型Vector2D,顶点分别是三维点、李群位姿
class G2O_TYPES_SBA_API EdgeProjectXYZ2UV : public  
               BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //1. 默认初始化
    EdgeProjectXYZ2UV();

    //2. 计算误差
    void computeError()  {
      //李群相机位姿v1
      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
      // 顶点v2
      const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
      //相机参数
      const CameraParameters * cam
        = static_cast<const CameraParameters *>(parameter(0));
     //误差计算，测量值减去估计值，也就是重投影误差obs-cam
     //估计值计算方法是T*p,得到相机坐标系下坐标，然后在利用camera2pixel()函数得到像素坐标。
      Vector2D obs(_measurement); 
      _error = obs - cam->cam_map(v1->estimate().map(v2->estimate()));
    }

    //3. 线性增量函数，也就是雅克比矩阵J的计算方法
    virtual void linearizeOplus();

    //4. 声明相机参数
    CameraParameters * _cam; 
    //5.读取与存盘
    bool read(std::istream& is);
    bool write(std::ostream& os) const;
};
```

​	一般来说，**边用于计算误差项(_error)，与雅可比矩阵，并将结果回传至g2o优化函数中，可以根据误差大小来判断是否已经达到极值点**，即是否可以停止优化。

​	其中(_estimate为已知量（创建边时会传入，即数据的真实值）)

​	本质上还是bool computeError() + virtual void linearizeOplus() + bool read(wtd::istream& is) + bool write(std::ostream& os) const 四个模板填参数



#### 三.光流法学习

​	光流法是第二种视觉里程计，原理相较于第一种视觉里程计简单很多。

​	**LK光流法常用于跟踪角点的运动。**

​	**LK光流法**的使用**基于两个假设**：

​	**①灰度不变假设：同一个空间点的像素灰度值，在各个图像中是固定不变的。**

​		即有：

​		![image-20250608235849917](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250608235849917.png)

​		等式左边泰特展开：![image-20250608235924431](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250608235924431.png)

​		上式联立有：

![image-20250609000018914](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250609000018914.png)

​		两边同除以dt则有：

​	![image-20250609000053265](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250609000053265.png)

​	**②某一个窗口内的像素具有相同的运动。**

​		则将dx/dt计为u，dy/dt计为v。由于在同一个窗口内的像素具有相同的运动，则每个像素点的u、v都是相同的，即是一个关于像素点位置的二元方程，写成矩阵形式有（考虑其是有w*w个像素的窗口）：

![image-20250609000345218](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250609000345218.png)

​		则：

![image-20250609000519154](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250609000519154.png)

​	即最后能将**像素的水平移动速度u与数值移动速度v解出**，印证其能**跟踪角点运动**的性质。



#### 四.LK光流法代码实现：

​	**①OpenCV实现：**

​		直接调用OpenCV中的calcOpticalFlowPyrLK函数，并传入第一张图像、第二张图像、第一张图特征点（角点）坐标信息、用储存第二张图中角点位置信息的容器、状态数组（标记第一张图的某个坐标信息是否合法）、误差error。

​	实现效果：

​	![image-20250609002308421](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250609002308421.png)

​		（自己拍摄的照片，没有数据集拍的那么标准，两次照片之间位姿变化过大会导致角点难以跟踪）

​	**②单层LK光流法实现：**

​		使用**高斯-牛顿法**（核心：**找到误差函数**）：

![2edb0662f7c56175cc0629c3c0cd5e9](C:\Users\Li\Documents\WeChat Files\wxid_3j6yc6bus6a732\FileStorage\Temp\2edb0662f7c56175cc0629c3c0cd5e9.png)

​	实现效果：

​	![image-20250609002629452](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250609002629452.png)

​	（由于位姿变化过大导致角点无法跟踪）

​	**③多层LK光流法实现：**

​	**解决位姿变化过大导致角点无法跟踪的问题。**（后续学习）

### 学习心得

​	①后续要备考期末，可能书籍阅读方面推进会变缓慢，但仍会保持前进。

​	②暂时侧重点放在SLAM相关文献的阅读上，综述+最新技术成果，关注前沿研究方向。



## Date7.5 重温复习视觉里程计一与视觉里程计二LK单层光流法

![3bed249e65d2f856b9f73ad24e99093](D:\Desktop\SLAM\Pict\3bed249e65d2f856b9f73ad24e99093.png)

**对极几何：**

![image-20250705103737381](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705103737381.png)

![image-20250705103753831](D:\Desktop\SLAM\Pict\image-20250705103753831.png)

​				（O1，O2，P三点所成平面为极平面、e1与e2为极点、l1与l2为极线）

​	O1与O2为相机中心点，为已知点。p1为特征点匹配中第一帧时确定的特征点，p2为特征点匹配中第二帧时确定的与p1相似的特征点，故至此为止空间点P可以求得（O1p1与O2p2连线交点）。

​	进而有：

![image-20250705104545997](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705104545997.png)

​	可见，根据所确定的P点位置，可以还原出两个不同帧之间的位姿变换关系。

​	**注意：什么是齐次坐标？什么是尺度下相等？**

​	**①齐次坐标：**

![image-20250705154945126](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705154945126.png)

![image-20250705154217716](D:\Desktop\SLAM\Pict\image-20250705154217716.png)

​	**②尺度意义下相等**

![image-20250705155126810](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705155126810.png)

​	**不同的齐次坐标之间只相差一个非0的比例因子，即两个齐次坐标之间除了深度不同以外其余坐标实际上都是一致的**

![image-20250705155342022](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705155342022.png)

​								***下面对相机模型部分进行复习***

**针孔相机模型：**

![image-20250705110131649](D:\Desktop\SLAM\Pict\image-20250705110131649.png)

![image-20250705110347380](D:\Desktop\SLAM\Pict\image-20250705110347380.png)

![image-20250705110408765](D:\Desktop\SLAM\Pict\image-20250705110408765.png)

![image-20250705110829111](D:\Desktop\SLAM\Pict\image-20250705110829111.png)

​	注意：此处的O-xyz坐标系相当于为以相机镜头为原点建立的坐标系，而O'-x'y'z'则相当于以CMOS为原点建立的坐标系。（X,Y,Z）与(X',Y',Z')分别代表空间中的点P与像素平面中的点P'的坐标（像素平面中P'不具有Z轴坐标）。(5.2)式满足三角形相似的推论。***此处以CMOS为原点建立的坐标系≠像素坐标系！！！***

​	对5.2进行变换进而可以得到**空间点在CMOS平面上的X'，Y'坐标**（所有单位都是米（m））：

​								![image-20250705111422028](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705111422028.png)![image-20250705112245381](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705112245381.png)

​	**像素坐标系：**

![image-20250705112100056](D:\Desktop\SLAM\Pict\image-20250705112100056.png)

​	总而言之，像素坐标系与CMOS坐标系虽然所在平面一样，但是原点位置不一样。像素坐标系以CMOS左上角为原点，而CMOS坐标系以相机光心（镜头）在CMOS平面的投影点作为原点。

​	进而我们可以得到像素坐标系与CMOS坐标系坐标之间的变换关系：

![image-20250705112235698](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705112235698.png)

​	5.4与5.3融合，将X'与Y'用含α与β的式子推导出点在像素平面坐标系与在相机光心坐标系的直接关系：

![image-20250705112658245](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705112658245.png)

​	则可以写成矩阵形式，过程如下：

![81fec5b4132cddcd0dbb3c4bc08beec](D:\Desktop\SLAM\Pict\81fec5b4132cddcd0dbb3c4bc08beec.jpg)

即：

![image-20250705114212800](D:\Desktop\SLAM\Pict\image-20250705114212800.png)

**相机的内参**

![image-20250705114730603](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705114730603.png)

​	注意：**内参K**是不会变化的，由为相机自身的固定性质。

**相机的外参**

![image-20250705151953954](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705151953954.png)

![image-20250705152030920](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705152030920.png)

​	总而言之，相机的外参数用于将把空间点在世界坐标系中的坐标变换到相机坐标系下。因此，外参数会随着相机的运动而发生变化。（注意：**世界坐标系与相机坐标系是相对而言的**，比如研究2d-2d时，可以将相机第一帧所在的坐标系计为世界坐标系，而后续的帧均为不同的相机坐标系，故只用找到第一帧与后面帧相机的位姿变换关系，就找到了相机的外参数R、t）

**归一化平面**

![image-20250705152307975](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705152307975.png)

​	归一化平面本质上与像素平面一致，只不过归一化平面上点的坐标为（u，v）且Z轴固定为1，故单目视觉中会使点的深度值丢失。

**畸变模型：**

![image-20250705152700125](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705152700125.png)

![image-20250705152623388](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705152623388.png)

​	直接将点在归一化平面上的坐标找到，再利用公式计算出去畸变后归一平面上的点。

​									***下面回到视觉里程计一***

![image-20250705155435835](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705155435835.png)

![image-20250705155750208](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705155750208.png)

​	注意：p1~KP与p2~K(RP+t)可用 尺度意义下相等 解释，即将p1、p2写成齐次坐标（x,y,1）与KP（P本身就是（X，Y，Z））二者相似。其中x1与x2就是归一化平面上的点，只有x、y两个轴上的坐标。

![image-20250705160149319](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705160149319.png)

![image-20250705160201467](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705160201467.png)

![image-20250705160212657](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705160212657.png)

![image-20250705160223439](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705160223439.png)

​	此处就得到了2d-2d位姿变换的核心式（7.8、7.10）。

​	故其实解决2d-2d的位姿变换问题核心是通过两帧图片所已知的两个2d点来得到本质矩阵E，进而解出位姿变换R、t。

​	**解法为SVD分解，略！！！**

​	**编程中直接套用现有算法即可。**

​	 [pose_estimation_2d2d.cpp](SLAM\slambook2-master\slambook2-master\ch7\pose_estimation_2d2d.cpp) 

​	![image-20250705163424537](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705163424537.png)

![image-20250705164645520](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705164645520.png)

​	**解法：**

![image-20250705164701044](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705164701044.png)

​	代码中直接调用OpenCV函数即可直接使用三角测量：

![image-20250705165216009](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705165216009.png)

​	传参传入第一、二帧图片中的FAST关键点与第一、二帧图片之间的匹配点，并用points收集第一帧图片的深度信息，进而：

![image-20250705165453420](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705165453420.png)

​	深度信息s2与s1之间也满足位姿变换关系s2=Rs1+t，因此将s1进行位姿变换后收集其（2,0）（第三行第一列）信息（即z轴信息）即可。

![image-20250705165632982](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705165632982.png)

![image-20250705165743232](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705165743232.png)

![image-20250705165828945](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705165828945.png)

​	如之前所提到的，**坐标系之间的变换是相对的**，2d-2d以第一帧时坐标系作为世界坐标系，而3d-2d由于本身就存在一个3d点，故直接以该3d点所在坐标系为世界坐标系，目的即是将另一帧图片中的2d点所在的相机坐标系相较于第一帧图片世界坐标系间的位姿变换确定下来。

![image-20250705172642135](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705172642135.png)

​	通过解出2d点在相机坐标系下的3d坐标，我们就能够利用ICP的方法解出世界坐标系与相机坐标系之间的位姿变换关系。

​	**解出2d点在相机坐标系下3d坐标的方法：**

​	**EPnP or 非线性优化Bundle Adjustment光束法平差(BA)**

​	代码实现：

​	**OpenCV的EPnP求解方法：**

![image-20250705174801904](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705174801904.png)

![image-20250705174843424](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705174843424.png)

​	**非线性优化**

​	 [pose_estimation_3d2d.cpp](SLAM\slambook2-master\slambook2-master\ch7\pose_estimation_3d2d.cpp) 

![image-20250705175148779](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705175148779.png)

![image-20250705175859571](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705175859571.png)

**非线性优化**

![image-20250705182122420](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705182122420.png)

![image-20250705182141615](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705182141615.png)

![image-20250705182404215](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705182404215.png)



**LK单层光流法：**

![image-20250705212648037](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705212648037.png)

基本假设：

![image-20250705212707984](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705212707984.png)

![image-20250705212727783](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705212727783.png)

则有：

![image-20250705212809465](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705212809465.png)

即经典的线性代数中超定方程，使用最**小二乘法解（略）or 非线性优化or OpenCV函数库**

**OpenCV**：
![image-20250705213202610](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705213202610.png)

​	调用函数，传入前后两帧图像（img1、img2）、第一帧图像的角点（pt1），则可输出追踪后的点（pt2），以及各点的状态（status）、误差（error）。

**非线性优化：高斯牛顿法**

![image-20250705213548781](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705213548781.png)

 [optical_flow.cpp](SLAM\slambook2-master\slambook2-master\ch8\optical_flow.cpp) 

![image-20250705214625972](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705214625972.png)

![image-20250705214651846](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705214651846.png)

​	核心步骤仍为高斯牛顿法，依然是找到误差函数，在最优化误差函数的过程中得到（u，v）（目标追踪点）的最优解。

## Date 7.7 学习记录 <span id="三"> </span>

### 1.C++中的参数注释方法（规范）

```c++
/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
```

### 2.C++中的内联函数 ‘inline’

​	当在定义函数前额外加上‘inline’前缀，则改函数将变为**内联函数**:

​	当程序运行时，内联函数的调用过程不会与常规函数一样进行压栈、入栈、出栈等操作，而是直接将内联函数嵌入进内存中，直接在固定地址对内联函数进行调用，大大节省了效率。

​	（在C++中，**内联函数**是一种用于提高函数执行效率的特性。通过使用*inline*关键字，开发者可以建议编译器在调用点替换函数体，从而减少函数调用的开销。内联函数通常用于执行简短的代码片段，以避免频繁调用小函数时产生的栈空间消耗。）

```C++
// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}
```

### 3.双线性插值函数 bilinear interpolation

​	双线性插值函数输出的浮点数结果在图像处理中具有重要作用，主要用于解决**非整数坐标下的像素值计算问题**

```C++
inline float GetPixelValue(const Mat &img,float x,float y)
{
    if(x < 0) x= 0;
    if(y < 0) y = 0;
    if(x >= img.cols) x = img.cols-1;
    if(y >= img.rows) y = img.rows-1;
    uchar *data = &img.data[int(y) * img.step + int(x)];  
    //img.step 指一行所占的字节数 ； int()用于取整 --> 故*data用于储存（int(x),int(y)）位置的像素
    
    float xx = x - floor(x); //计算x坐标的小数部分
    float yy = y - floor(y); //计算y坐标的小数部分
    //floor()为向负方向取整（不大于输入参数的最大整数）
    
    return float(
        (1-xx) * (1-yy) * data[0] +
        xx * (1 - yy) * data[1] + 
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}
```

**Keypoints！！！**

​	**①uchar *data = &img.data[int(y) * img.step + int(x)];**  

​	指的是用指针data储存img.data的第int(y) * img.step + int(x)个元素的地址，因此data[0] = img.data[int(y) * img.step + int(x)]，以及data[1] = img.data[.... + 1]、data[img.step] = img.data[... + img.step]（下一行的该列）。

​	**②(1-xx) * (1-yy) * data[0] +**

​            **xx * (1 - yy) * data[1] +** 

​            **(1 - xx) * yy * data[img.step] +**

​            **xx * yy * data[img.step + 1]**

​	此处为双线性插值法的核心过程，通过**加权非整数像素点的相邻四个整数点**（左上、左下、右上、右下）来得到该**非整数坐标像素点的像素**。

```text
(3,4)权重0.24   (4,4)权重0.06
       +-----------+
       | · · · · · |
       | · · · · · |  y=4.7
       | · · · · · |
       +-----*-----+  <- 目标点(3.2,4.7)
       | · · · · · |     离(3,5)最近 → 权重最大(0.56)
       | · · · · · |
       +-----------+
     (3,5)权重0.56   (4,5)权重0.14
```

```cpp
return 
(1 - xx) * (1 - yy) * data[0] +  // 左上权重 × 左上像素
xx * (1 - yy) * data[1] +        // 右上权重 × 右上像素
(1 - xx) * yy * data[img.step] + // 左下权重 × 左下像素
xx * yy * data[img.step + 1];    // 右下权重 × 右下像素
```

​	其中**(1-xx)为像素点距离右边界的距离（水平距离权重），(1-yy)为像素点距离下边界的权重（垂直距离权重）**，xx与yy同理（分别也为水平距离权重与垂直距离权重）。

​	（由水平与垂直两方向权重确定像素值-->双线性插值法）

### 4.boost::format + for循环 轮流读取多个文件

- **`"./%06d.png"`**：格式化字符串模板，其中：

  - `%06d`：表示用 6 位数字填充，不足位补零（如 `5` → `000005`）。
  - `./`：文件路径前缀（当前目录）。

- **功能**：后续通过 `fmt_others % 数值` 生成具体的文件名。

  ————

  for循环分步解析

  1. **`fmt_others % i`**：
     - 将整数 `i` 插入到格式化字符串的 `%06d` 位置。
     - 例如 `i = 1` 时，生成 `./000001.png`；`i = 42` 时，生成 `./000042.png`。
  2. **`.str()`**：
     - 将 `boost::format` 对象转换为 `std::string`，得到最终文件名。
  3. **`cv::imread`**：
     - 根据生成的文件名读取图像（`0` 表示以灰度模式读取）。

```C++
boost::format fmt_others("./%06d.png"); // 定义格式化模板
for(int i = 1;i<6;i++)
{
    Mat img = imread((fmt_others % i).str(),0);
}
```

## Date 7.14 ~ 7.19 周总结 <span id="四"> </span>

### 学习内容：

#### 1. 中型ROS2工程代码实践

​	在回顾ROS2基本知识的基础上，对基于ROS2构建的机器人自瞄算法工程进行了修改，利用所学ROS2知识结合EFK（卡尔曼扩展滤波）实现机器人自瞄的可视化（将预瞄点在视觉传感器所摄图片中标出并在节点中发布）。

​	详细思路与过程：

​	①找到云台控制信息发出的yaw-pitch角，找到yaw-pitch角所指向的目标点在云台坐标系下的坐标。

​	②考虑在装甲板信息中额外记录当前装甲板由相机坐标系到云台坐标系之间的变换矩阵。

​	③将云台所指向的目标点通过变换矩阵转换为相机坐标系下的目标点，即找到相机坐标系下的目标点。

​	④将相机坐标系下的目标点利用内参矩阵K+畸变系数p投影到像素平面上，得到目标点在像素平面上的投影点。

​	⑤最后利用OpenCV画图将像素坐标系下的目标点在result图像中标注出来，并在图像发布器上发布。

​	实现效果：

​	节点关系图：

![6e202aae0a592381b14fb33e58131c9](C:\Users\Li\Documents\WeChat Files\wxid_3j6yc6bus6a732\FileStorage\Temp\6e202aae0a592381b14fb33e58131c9.jpg)

​	视频展示：

<video src="C:\Users\Li\Documents\WeChat Files\wxid_3j6yc6bus6a732\FileStorage\Video\2025-07\3e3c101c744ffb57b2a754c07e6b9e98.mp4"></video>

​	本次实践中，我积累了处理中型CPP工程项目的经验，比如如果要像原有的代码中加入新功能，在什么地方需要额外注意（如.cpp文件中新增的变量，需要在.hpp中提前声明，否则会产生编译错误、如.cpp中多进程可能产生冲突，需要添加进程锁防止进程崩溃、如如何跨.cpp文件得到不同.cpp文件中定义的变量值（在目标.cpp文件中创建该命名空间的函数，函数的返回值即为想要得到的值，则最后只需要在实例下调用该函数即可得到目标cpp文件中的变量值）。

![image-20250719211125497](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250719211125497.png)

![image-20250719211155398](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250719211155398.png)

​	此外，本次实践也让我对OpenCV有了更多的认识，比如利用OpenCV灰度处理，二值化处理后得到仅有发光灯条的图片，便于进行灯条检测等。此外，如OpenCV中的projectPoints()函数，在视觉SLAM中曾有遇到，用与将相机坐标系下的点投影至像素平面坐标系，此次重新温习，增进了我对该函数的认识(比如该函数内部已经包含三维点的归一化过程，且投影效果与传入的内参矩阵、畸变系数准确度息息相关。因此为了得到更好的点投影效果，此次实践中我也对相机标定进行了学习，并掌握了使用Matlab处理标定数据，提炼准确的相机内参)。

![74afe6ce0bb9a846b0740a0d8301310](D:\Desktop\RM算法组\暑假实训\作业\Picture\74afe6ce0bb9a846b0740a0d8301310.png)

![ab6092c00c590a254a44a2529d5c7d8](D:\Desktop\RM算法组\暑假实训\作业\Picture\ab6092c00c590a254a44a2529d5c7d8.png)

#### 2.《视觉SLAM十四讲》 第八讲 视觉里程计二（光流、直接、多层光流\直接法）、第九讲 后端(一)（滤波器（KF、EKF）、非线性优化方法（BA图优化（g2o、ceres）））、第十讲 后端(二) （滑动窗口、位姿图） 

##### ①视觉里程计二--直接法：

###### 	视觉里程计一、二中算法的区别：

​	视觉里程计一 中的算法需要**找到图像中的特征点（角点）**，并**计算特征点的描述子**，在两张图像中进行**特征点匹配**。并使用匹配成功的点对作为基础，通过**最小化重投影误差（对极几何、PnP、ICP算法）**优化相机运动，估计相机位姿；

​	而视觉里程计二 中的算法（光流法、直接法），则甚至**无需提取特征点**，只需在图像上随机选点即可，更是**省略了描述子计算、特征点匹配**的过程，大大节省了计算时间，且在特征点稀疏的场景更有优势。其中，视觉里程计二中的**光流法（稀疏）**通过提取随机n个像素点的灰度值（亮度值），并通过**最小化光度误差**得到**点与点之间的关系**（即x、y方向轴上的速度u、v与前后两张图片中点的位移，**也是一种运动估计**）。此外，视觉里程计二中的**直接法**，则可以通过**最小化光度误差**直接对**相机位姿进行优化**。

###### 	视觉里程计二中光流法与直接法之间的区别：

​	光流法与直接法在运动估计的原理部分都是一样的，均采用最小化光度误差来优化相机运动，估计最佳相机运动。然而，二者所估计的相机运动有所不同，**光流法**为估计**相机的平面运动**（x、y方向上的运动速度与位移）；而**直接法**则为估计**相机的空间位姿运动**。

###### 	直接法计算过程：

​	![image-20250721110146452](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721110146452.png)

​	如图所示，我们可以写出投影方程：

![image-20250721110220074](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721110220074.png)

​	p1与p2为两图像中像素点的坐标，然而，直接法的原理是**最小化光度误差进行相机位姿估计**，估我们通过p1、p2得到两像素平面上的光度值，作差即为误差e：	

![image-20250721110701135](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721110701135.png)

​	由于我们要根据优化误差来得到最佳的位姿变换T（即关心误差e是如何随着相机位姿T变化的），估推导由像素位置p到位姿变换T的关系式：

![image-20250721111138887](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721111138887.png)

​	此时即建立起了位姿变换T与第二个像素平面坐标u之间的函数关系：

![image-20250721111459626](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721111459626.png)

​	则对误差e求导，即相当于对I(p2)求偏导，即为：

![image-20250721111527395](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721111527395.png)

​	各个偏导详细求解过程：

![image-20250721111819491](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721111819491.png)

###### 	单层直接法代码实现流程：

​	**1. 初始化阶段**

- **输入数据**：
  - 参考图像 `img1` 和当前图像 `img2`（灰度图）。
  - 参考图像中的像素点集合 `px_ref` 及其对应的深度值 `depth_ref`。
  - 初始相机位姿 `T21`（从参考图像到当前图像的变换矩阵）。
- **参数设置**：
  - 迭代次数 `iterations = 10`。
  - 创建 `JacobianAccumulator` 对象 `jaco_accu`，用于计算雅可比矩阵、海森矩阵和误差。

------

**2. 迭代优化位姿**

**每次迭代的核心步骤**：

1. **重置累加器**：
   - 调用 `jaco_accu.reset()` 清空海森矩阵 `H`、偏置项 `b` 和误差 `cost`。
2. **并行计算雅可比矩阵**：
   - 使用 `cv::parallel_for_` 并行处理所有像素点，调用 `accumulate_jacobian` 计算每个点的贡献：
     - **投影3D点**：将参考图像的像素点根据深度和相机内参转换为3D点，再通过当前位姿 `T21` 投影到当前图像。
     - **光度误差计算**：比较参考图像和当前图像中对应像素块的灰度值差异。
     - **雅可比矩阵计算**：
       - 图像梯度（`J_img_pixel`）：通过中心差分计算当前图像的灰度梯度。
       - 像素对位姿的导数（`J_pixel_xi`）：根据投影几何推导的2×6矩阵。
       - 合成总雅可比矩阵 `J = -(J_img_pixel^T * J_pixel_xi)^T`。
     - **更新海森矩阵和偏置**：
       - `H += J * J^T`（高斯牛顿法）。
       - `b += -error * J`。
3. **求解位姿更新量**：
   - 解线性方程 `H * δξ = b`，得到位姿增量 `update`。
   - 通过指数映射 `Sophus::SE3d::exp(update)` 更新当前位姿 `T21`（左乘扰动）。
4. **收敛判断**：
   - 如果误差 `cost` 上升或更新量 `update.norm()` 小于阈值 `1e-3`，提前终止迭代。

------

**3. 输出与可视化**

- **输出结果**：
  - 打印优化后的位姿 `T21` 和总耗时。
- **可视化匹配点**：
  - 在当前图像上绘制参考图像像素点（绿色圆圈）及其投影位置（绿色连线）。

------

**关键函数说明**

**`JacobianAccumulator::accumulate_jacobian`**

1. **投影3D点**：
   - 将像素坐标 `(x,y)` 转换为归一化坐标，乘以深度得到3D点 `point_ref`。
   - 通过 `T21` 变换到当前相机坐标系 `point_cur`，再投影到当前像素坐标 `(u,v)`。
2. **光度误差**：
   - 对每个像素周围的 `3x3` 邻域，计算参考图像和当前图像的灰度差值 `error`。
3. **雅可比矩阵**：
   - **图像梯度**：通过双线性插值计算当前图像的 `x` 和 `y` 方向梯度。
   - **投影几何导数**：推导像素坐标对李代数位姿的导数（2×6矩阵）。
   - 合成总雅可比矩阵 `J`。
4. **并行累加**：
   - 使用互斥锁保护全局变量 `H`、`b` 和 `cost` 的更新。

------

**4. 辅助函数**

- **`GetPixelValue`**：双线性插值获取图像像素值，处理边界情况。



###### 	多层直接法代码实现：

**1. 流程结构对比**

| **步骤**           | **单层直接法**                               | **多层直接法**                                               |
| :----------------- | :------------------------------------------- | :----------------------------------------------------------- |
| **图像预处理**     | 直接使用原始分辨率图像。                     | 构建图像金字塔（如4层，缩放比例0.5）。                       |
| **参数初始化**     | 固定相机内参 `(fx, fy, cx, cy)`。            | 每层缩放内参（如第`level`层内参为 `fx*scale^level`）。       |
| **位姿优化入口**   | 直接调用 `DirectPoseEstimationSingleLayer`。 | 从最粗层（顶层）开始，逐层调用 `DirectPoseEstimationSingleLayer`。 |
| **位姿传递方式**   | 无跨层传递，单次优化完成。                   | 上一层的优化结果 `T21` 作为下一层的初始值。                  |
| **像素点坐标处理** | 直接使用原始像素坐标 `px_ref`。              | 每层按比例缩放像素坐标 `px_ref * scale^level`。              |

**2. 关键操作差异**

**(1) 图像金字塔构建**

- **多层法**：

  ```cpp
  vector<cv::Mat> pyr1, pyr2;  // 图像金字塔
  for (int i = 0; i < pyramids; i++) {
      if (i == 0) {
          pyr1.push_back(img1); 
          pyr2.push_back(img2); // 原始图像
      } else {
          cv::resize(pyr1[i-1], img1_pyr, Size(rows*scale, cols*scale)); // 降采样 ***即在图像所想展示的内容不变前提下，将图片分辨率降低（如原始图像：640x480 → 第1层：320x240 → 第2层：160x120）***
          pyr1.push_back(img1_pyr);
      }
  }
  ```

- **单层法**：无需此步骤，直接使用 `img1` 和 `img2`。

**(2) 内参和像素坐标缩放**

- **多层法**：

  ```cpp
  for (int level = pyramids-1; level >= 0; level--) {
      fx = fxG * scales[level];  // 缩放内参
      VecVector2d px_ref_pyr = px_ref * scales[level]; // 缩放像素坐标
      DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
  }
  ```

- **单层法**：内参和像素坐标保持不变。

**(3) 位姿优化顺序**

- **多层法**：
  从最粗层（低分辨率）到最细层（高分辨率）**逐层优化**，位姿结果依次传递：

  ```cpp
  for (int level = pyramids-1; level >= 0; level--) {
      // 每次调用单层优化，T21会被更新并传递到下一层
      DirectPoseEstimationSingleLayer(..., T21); 
  }
  ```

- **单层法**：仅在原始分辨率进行一次优化。

**3. 核心代码差异示例**

**(1) 单层法调用**

```cpp
// 直接使用原始图像和参数
DirectPoseEstimationSingleLayer(img1, img2, px_ref, depth_ref, T21);
```

**(2) 多层法调用**

```cpp
// 构建金字塔后，从顶层到底层逐层优化
for (int level = pyramids-1; level >= 0; level--) {
    fx = fxG * scales[level];  // 动态调整内参
    VecVector2d px_ref_pyr = px_ref * scales[level]; // 缩放像素坐标
    DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
}
```

​	详情： [direct_method.cpp](SLAM\slambook2-master\slambook2-master\ch8\direct_method.cpp) 

##### ②后端(一)

###### 	什么是后端？

​	后端站在全局的角度，对机器人的运动进行最优估计。

![image-20250721152840468](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721152840468.png)

​	SLAM过程可以由运动方程与观测方程来描述，即：
![image-20250721153110178](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721153110178.png)

​	其中Xk为机器人在k时刻状态（位姿），Uk为k时刻的运动输入，Zk为k时刻的观测值，Yj为路标点，Wk、Vk分别为运动方程的噪声与观测方程的噪声。	![image-20250721153657582](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721153657582.png)

​	根据对k时刻状态Xk的相关性假设，后端的处理方法分为滤波器方法（k时刻只与k-1时刻状态相关）与非线性优化方法（k时刻与k时刻之前所有状态相关）两种方法。

![image-20250721154447352](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721154447352.png)

###### 	后端中的滤波器方法（线性系统与KF）卡尔曼滤波推导方法

​	由于滤波器方法假设了马尔科夫性（k时刻只与前k-1时刻状态相关），则可以将（9.6）式中的两式转写为：

![image-20250721160117797](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721160117797.png)

​	也就是说滤波器方法只需要维护一个时刻的状态量（位姿）即可，如果该状态量满足高斯分布，则只需要维护该状态量的均值与协方差即可，于是滤波器便能够推导下一时刻的机器人状态：

![image-20250721161929165](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721161929165.png)

​	由线性高斯系统将（9.5）式中似然与先验概率转写成高斯分布 -- **预测**：

![image-20250721161328993](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721161328993.png)

![image-20250721163459713](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721163459713.png)

​	则根据预测步得到的（9.11）与（9.13）两个方程，能够对后验进行计算（后验=似然*先验），即 -- **更新**：

![image-20250721163754433](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721163754433.png)

​	将高斯分布展开，对应项对齐，最终解出后验与先验的关系：

![image-20250721164200353](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721164200353.png)

###### 	非线性系统与扩展卡尔曼滤波（EKF）

![image-20250721164406089](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721164406089.png)

​	扩展卡尔曼滤波滤波利用泰勒展开将非线性变为线性，后续过程与KF推导过程一致，则最终有：

![image-20250721164507091](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721164507091.png)

###### 	后端中的非线性优化方法（BA与图优化）

​	BA（光束法平差）：

![image-20250721165944952](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721165944952.png)

​	投影过程：

![image-20250721170020106](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170020106.png)

![image-20250721170056572](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170056572.png)

​	BA求解：

![image-20250721170133461](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170133461.png)

​	即使用高斯牛顿法or列文伯格法求解该式：

![image-20250721170343509](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170343509.png)

​	利用稀疏性性质，采用边缘化路标点变量δXp（Schur消元）：

![image-20250721170542461](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170542461.png)

![image-20250721170635554](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170635554.png)

![image-20250721170712915](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170712915.png)

​	避免误匹配产生的误差影响整体后端估计--鲁棒核函数：
![image-20250721172335424](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721172335424.png)

![image-20250721172348319](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721172348319.png)

###### 使用Ceres库计算BA（代码实现）

​	![image-20250721193322325](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721193322325.png)

​	核心代码部分：

![image-20250721174933472](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721174933472.png)

​	首先，调用BAL数据集，初始化数据集数据（标准化、设置噪声强度）。然后，调用SolveBA函数对数据进行求解运算，得到优化后的数据（点云数据（位姿））。最后，调用BAL数据集导出数据（名为“finnal.ply”）。

​	SolveBA()函数部分：

​	核心为定义代价函数，计算与求雅可比矩阵部分交给ceres库自动计算，最终直接输出结果即可。

![image-20250721175338703](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721175338703.png)

​	着重详细讲述代价函数定义部分

```cpp
  cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);
```

​	通过调用**SnavelyReprojectionError::Create()函数**创建自动微分代价函数：

![image-20250721175608479](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721175608479.png)

1. `ceres::AutoDiffCostFunction` 解析

```cpp
ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>
```

这是Ceres Solver提供的自动微分模板类，其核心参数如下：

| 模板参数                   | 说明                                   |
| -------------------------- | -------------------------------------- |
| `SnavelyReprojectionError` | 用户自定义的仿函数类，包含误差计算逻辑 |
| `2`                        | 输出残差的维度（2D像素坐标误差）       |
| `9`                        | 第一个优化变量的维度（相机参数）       |
| `3`                        | 第二个优化变量的维度（3D点坐标）       |

关键特点：
- **自动微分**：自动计算雅可比矩阵，避免手动推导复杂导数
- **多参数支持**：可支持多个参数块（这里处理相机和点两个参数块）
- **类型泛化**：通过模板参数`T`同时支持双精度和Jet类型（用于自动微分）

2. `SnavelyReprojectionError` 类实现

```cpp
class SnavelyReprojectionError {
public:
    // 构造函数保存观测值
    SnavelyReprojectionError(double observation_x, double observation_y) 
        : observed_x(observation_x), observed_y(observation_y) {}

    // 核心误差计算函数
    template<typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        // 1. 将3D点投影到相机坐标系
        T p[3];
        AngleAxisRotatePoint(camera, point, p);  // 旋转
        p[0] += camera[3];  // 平移X
        p[1] += camera[4];  // 平移Y
        p[2] += camera[5];  // 平移Z

        // 2. 归一化平面投影
        T xp = -p[0]/p[2];
        T yp = -p[1]/p[2];

        // 3. 应用径向畸变
        const T& l1 = camera[7];  // 二阶畸变系数
        const T& l2 = camera[8];  // 四阶畸变系数
        T r2 = xp*xp + yp*yp;
        T distortion = T(1.0) + r2*(l1 + l2*r2);

        // 4. 焦距缩放
        const T& focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;

        // 5. 计算残差
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        return true;
    }

    // 工厂方法创建CostFunction
    static ceres::CostFunction* Create(double observed_x, double observed_y) {
        return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
            new SnavelyReprojectionError(observed_x, observed_y));
    }

private:
    const double observed_x;  // 观测到的x坐标
    const double observed_y;  // 观测到的y坐标
};
```

3. 参数映射说明

相机参数 `camera[9]` 的结构：

| 索引 | 参数        | 类型   | 说明                  |
| ---- | ----------- | ------ | --------------------- |
| 0-2  | rotation    | double | 旋转向量（角轴表示）  |
| 3-5  | translation | double | 平移向量 (tx, ty, tz) |
| 6    | focal       | double | 焦距 f                |
| 7    | k1          | double | 二阶径向畸变系数      |
| 8    | k2          | double | 四阶径向畸变系数      |

3D点参数 `point[3]` 的结构：

| 索引 | 参数 | 类型   | 说明    |
| ---- | ---- | ------ | ------- |
| 0    | X    | double | 点X坐标 |
| 1    | Y    | double | 点Y坐标 |
| 2    | Z    | double | 点Z坐标 |

4. 自动微分工作原理(在ceres内部调用，用于计算残差的雅可比矩阵，并求解Hδx=g方程)

1. **第一次调用**：使用实际类型 `double` 计算残差
2. **第二次调用**：使用Jet类型计算偏导数
   - 对相机参数求导：9个偏导
   - 对点坐标求导：3个偏导
3. Ceres自动组合这些偏导数形成雅可比矩阵

###### 使用g2o库计算BA（代码实现）

​	略！ -- 定义顶点 + 定义二元边 + 添加顶点、二元边 + 优化（见代码）

​	 [bundle_adjustment_g2o.cpp](SLAM\slambook2-master\slambook2-master\ch9\bundle_adjustment_g2o.cpp) 



## Date7.21 ~ 7.27 周总结 <span id = "五"> </span>

### 	学习内容:

#### 	1.《视觉》第11讲 -- 回环检测

![image-20250728105956735](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728105956735.png)

​	在后端中，回环检测是一十分重要的部分。

​	回环检测的基础 -- 词袋模型。词袋模型通过将图片中的特征点向下划分（类似树结构），归类得到含多个单词的字典。该字典即可以用于对任意一张新图片进行描述（将图片转换为词袋向量），从而能够实现图像之间的相似度检测，进而能够得到回环的一对照片。

​	当检测到回环，并得到回环的一对照片后，在这一对照片的序号之内进行回环的后端优化（比如序号1-10之内），实现对**地图路标点+相机位姿**的同时优化，最终在地图中得到更加准确的相机位姿与路标点位置。

> [!note]
>
> Jetson 嵌入式设备无法使用较大的字典对图像进行分类（Maybe CPU性能不足），只能使用较小的字典，估判断图像相似不那么好。

#### 	2.《视觉》第12讲 -- 建图

![image-20250728110014759](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728110014759.png)

​	稠密地图是相对于稀疏地图而言的，**稀疏地图只建模感兴趣的部分（即特征点）**，而**稠密地图建模所有看到的点（每一个像素点都建模）**。稠密地图建模过程中运算量大，但得到的**稠密地图相较于稀疏地图**，能够辅助机器人实现**导航、避障**或是环境的完整**三维重建**。

​	**①单目稠密重建**，由于使用单目相机进行稠密重建，而单目相机不能直接得到像素点深度，需要使用对极几何对像素点进行**三角化（极限搜索+块匹配）**，故虽然能够实现稠密重建，但是实现效果不佳（在特征稀疏平面中，块匹配会由于一个区域的像素都长得差不多而失效）。

（下图为：原始图像image 、 真实的深度图像depth_truth 、 估计的深度图像与误差depth_estimate、depth_error）

![image-20250728113609720](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728113609720.png)

​	实际上**此时已经计算出了图像中所有像素点的3d位置**，并且数据集中已经记录了各个图像相机的位姿，如果将每次计算的**像素点3d位置**结合**相机位姿**进行点云拼接，**就能得到稠密地图**。

​	**②RGB-D稠密建图**，RGB-D相机拍摄出的图片会附属携带其深度信息，估不再需要想单目or双目相机那样通过计算得到图像的深度信息，既节省了计算量，同时也避免了计算带来的误差。

​	RGB-D的稠密建图与单目相机稠密建图在最后一步完全一致（即得到了图像的深度信息后（即相当于得到所有像素的相机坐标系下3d点）），只需要**结合相机位姿信息将相机坐标系下各个像素的3d坐标转换为世界坐标系下的3d坐标**，并将**点云进行加和**，使用**滤波器（统计滤波器+体素滤波器）消除不需要的点**，最终即可**得到多幅图像拼接得到的稠密点云地图**。

![image-20250728120554665](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728120554665.png)

​	然而单纯的稠密点云地图并不能用于定位（点云并没有进行位姿上的优化)，也不能用于导航与避障（不能说单纯几个点云是否被占据，需要将点云划分为占据网格)。因此需要从点云重建网格（先计算点云中每个点的法线（拟合平面找法线），再根据法线计算网格)。

![image-20250728160659995](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728160659995.png)

​	此图即为**点云数据+占据网格后得到的稠密地图**（白色线是我加的法线可视化，也可隐藏）。

​	然而该占据网格地图也有弊端，一是.pcd文件所占空间大（其中有很多我们不需要的细节），二是无法处理运动物体，点云地图靠的是点云间的不断拼接而成，而没有删除点云的做法。为解决这两个问题，我们可以使用八叉树地图（octomap）。

![image-20250728161340953](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728161340953.png)	八叉树可以自定义展开的层数，展开的层数越多那么细节也就越多，相反展开层数越少细节则越少，且占据信息使用[0,1]中的的概率（概率对数值）表示（初始时占据信息为0.5，若后续不断检测到该占据信息那么概率则增加，相反概率减少)，既能够实现占据信息的实时更新，也能够节约大部分空间消耗。

![image-20250728161745752](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728161745752.png)

​						（⬆️⬆️⬆️概率对数值y与概率x的换算关系⬆️⬆️⬆️）

![image-20250728163202396](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728163202396.png)

​										（八叉树展开八层）

![image-20250728163430832](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728163430832.png)

​								（八叉树展开十五层并按照深度值上色）

#### 	3.《视觉》第13讲 -- MySLAM工程实践（代码理解+增加回环检测功能）

![image-20250728110026005](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728110026005.png)

​	该视觉SLAM工程使用**双目相机**作为视觉传感器，构建**稀疏地图**。

​	视觉SLAM工程主要由四个部分构成，**前端、后端、地图、回环检测**。

​	**前端**的工作是根据视觉传感器所采集的图像做初始化的位姿估计与特征点检测、匹配工作，根据特征点匹配的情况（如果前后两帧图像匹配成功的内点（排除有误差的外点后剩的点）较少）决定是否更新地图中的路标点、设置关键帧与是否进入后端进行位姿优化（对前端设置的关键帧位姿进行优化）。

​	**后端**的工作是根据前端传回的信息，对目前活跃的关键帧进行优化（目前活跃关键帧为只在全部关键帧中使用滑动窗口挑选最新的几帧关键帧，加快了后端优化速度），由于后端负责对地图的规模进行控制，如果不使用滑动窗口挑选最新关键帧，那么会导致后端优化十分缓慢，进而导致可视化不及时甚至卡死（不过理应后端不影响实时可视化？？猜测后端长期占用地图线程导致前端无法进入地图线程中对信息进行更新）。

​	**地图**的工作是标记视觉传感器所采集到的路标点，与显示关键图像帧（可选是否显示），主要用于给用户端的可视化展示，构建稀疏地图。

​	**回环检测**的工作在先前提过，当检测到回环，并得到回环的一对照片后，在这一对照片的序号之内进行回环的后端优化（比如序号1-10之内），实现对**地图路标点+相机位姿**的同时优化，最终在地图中得到更加准确的相机位姿与路标点位置。

​	值得注意的是，该SLAM工程在前端、后端、回环检测中都使用的是借助**BA原理使用g2o**图优化，然而，**前端的g2o仅为单元边**的优化（仅针对相机位姿），而**后端与回环检测都是双元边**的优化（相机位姿 + 路标点），但是**后端采用的是滑动窗口**，故只对最近的几帧关键帧进行优化，而**回环检测是对整个回环包含的关键帧进行优化**，范围更大。并且，工程**特征匹配**所用的方法都是**光流法**，而**特征点检测**使用的是**GFTT**方法。

![image-20250728181904618](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728181904618.png)

​									（使用KITT数据集进行视觉SLAM建图的结果）

> [!note]
>
> ​	如果采用滑动窗口，MySLAM窗口中就不能显示全局的特征点，Why？
>
> ​	--- 滑动窗口只显示被观测的路标点，未被观测的路标点会被删除。
>
> ​	目前虽然能打开全局路标点显示（将全局路标点用新变量储存并可视化该新变量），但是由于回环检测模块有bug，导致全局位姿偏差较大（建出来的图十分丑陋）。

### 	未来规划：

#### 	1.下一周对目前视觉SLAM优秀开源代码进行学习、理解、复现

​	VSLAM要求：（稀疏or半稠密 -- 能够实时建图）

​	**ORB-SLAM** -- 特征点SLAM中的巅峰，十分经典的视觉SLAM，能够实现实时的视觉SLAM建图，并支持单目、双目、RGB-D多种模式。缺点是由于建立的是稀疏地图，估不能支持避障、导航功能。

​	深入SLAM研究需要一定的深度学习基础，后续考虑学习深度学习？（若有时间）

#### 	2.实践：使用视觉传感器实现实时建图！

​	使用ORB-SLAM尝试实现手持相机建图！

​	复现可参考该GitHub -- [LegendLeoChen/LeoDrone: ubuntu22.04 + ROS2 humble 环境下的无人机基本运动控制和视觉SLAM方案](https://github.com/LegendLeoChen/LeoDrone?tab=readme-ov-file)

#### 3.实践过程中问题记录 --（Successfully）

##### 	①使用ORB-SLAM数据集能跑，ROS2版本功能包能成功编译，但是ros2 run之后报错：

```apl
jetson@snake:~/ORB_SLAM_ROS2_WS$ ros2 run orbslam3 mono /home/jetson/ORB_SLAM_ROS2_WS/src/ORB_SLAM3_ROS2-humble/vocabulary/ORBvoc.txt /home/jetson/ORB_SLAM3-master/Examples/Monocular/TUM1.yaml

ORB-SLAM3 Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
ORB-SLAM2 Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
This program comes with ABSOLUTELY NO WARRANTY;
This is free software, and you are welcome to redistribute it
under certain conditions. See LICENSE.txt.

Input sensor was set to: Monocular
Loading settings from /home/jetson/ORB_SLAM3-master/Examples/Monocular/TUM1.yaml
	-Loaded camera 1
Camera.newHeight optional parameter does not exist...
Camera.newWidth optional parameter does not exist...
	-Loaded image info
	-Loaded ORB settings
Viewer.imageViewScale optional parameter does not exist...
	-Loaded viewer settings
System.LoadAtlasFromFile optional parameter does not exist...
System.SaveAtlasToFile optional parameter does not exist...
	-Loaded Atlas settings
System.thFarPoints optional parameter does not exist...
	-Loaded misc parameters
----------------------------------
SLAM settings: 
	-Camera 1 parameters (Pinhole): [ 517.306 516.469 318.643 255.314 ]
	-Camera 1 distortion parameters: [  0.262383 -0.953104 -0.005358 0.002628 1.16331 ]
	-Original image size: [ 640 , 480 ]
	-Current image size: [ 640 , 480 ]
	-Sequence FPS: 30
	-Features per image: 1000
	-ORB scale factor: 1.2
	-ORB number of scales: 8
	-Initial FAST threshold: 20
	-Min FAST threshold: 7


Loading ORB Vocabulary. This could take a while...
Vocabulary loaded!

Initialization of Atlas from scratch 
Creation of new map with id: 0
Creation of new map with last KF id: 0
Seq. Name: 
There are 1 cameras in the atlas
Camera 0 is pinhole
slam changed
============================ 
Starting the Viewer
[ros2run]: Segmentation fault
```

​	解决方案（不知是否可行）-- [Segmentation fault when running ros2 humble (mono) · Issue #20 · zang09/ORB_SLAM3_ROS2](https://github.com/zang09/ORB_SLAM3_ROS2/issues/20)

​	Jetson orin nano自带的OpenCV库为4.1.0，而ORB_SLAM_ROS2功能包中要求4.2.0，清除4.1.0的OpenCV下载4.2.0的？

​	-- 解决方案可行！！！

---

##### 	②连接上USB相机后使用yahboom自带的相机驱动无法启动相机！

​	解决方案

​	在camera_usb.py中将下面代码

```C++
self.cap = cv2.VideoCapture(0)
```

​	改为

```C++
self.cap = cv2.VideoCapture(1)
```

​	意为初始读取/dev/video0设备改为读取/dev/video1（嵌入式相机为video0，而USB相机为video1（USB相机接入后一般有video1与video2两个dev，但是只有video1有相机输出画面））

---

##### 	③如何使用无线传输将手机相机传入Ubuntu系统中（实现Ubuntu系统能够识别出/dev/videox的效果）

​	解决方案

​	首先，在https://www.dev47apps.com/中下载适用于arm设备的源码（Jetson orin nano为arm64架构，只能通过源码编译下载该虚拟投屏软件）。得到DroidCam源码后，阅读README.md文件安装Ubuntu所需要依赖，依赖安装完成后sudo ./install-client 与 sudo ./install-video，完成DroidCam软件的完整安装。

​	然后，手机中下载DroidCam OBS（随便下，我下的是破解版）。在Ubuntu终端输入DroidCam打开DroidCam，手机打开DroidCam OBS，将手机中的设备IP号输入到Ubuntu的DroidCam中，点击start即可完成虚拟图像传输！！！（Tips:实验室网不好，建议开热点，让两设备处于同一热点下）（此时Ubuntu系统已经能够成功识别设备，显示为 /dev/video3与/dev/video4（或2与3））

​	最后，在yahboom自带的ROS2相机驱动包中将camera_usb.py中下面代码

```C++
self.cap = cv2.VideoCapture(1)
```

​	改为

```C++
self.cap = cv2.VideoCapture(4)
```

​	colcon build ， source ，ros2 run三件套，即可将传输进的虚拟图像在ROS2中发布为名称为“/image_raw"的消息！！！（供ORB_SLAM订阅）

​	或者！！可以直接在手机上发布ROS2消息，但是目前只在网上找到ROS1消息的软件，且仅支持到安卓7.1系统。

---

##### 	**成功复现！！！**

​	**使用USB接口摄像头建图：**

![image-20250801165018029](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250801165018029.png)

​	**使用无线设备图像传输建图：**

![image-20250801165052438](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250801165052438.png)



## Date7.28 ~ 8.2 周总结 <span id="六"> </span>

### 学习内容：

#### 	ORB_SLAM3经典视觉SLAM算法功能复现与代码初步理解。

1. ##### ORB_SLAM3原生仅支持Ubuntu18.04 ROS1，而我们需要在Ubuntu22.04的arm架构上运行，并实现实时建图、无线通信，需要ORB_SLAM_ROS2工程辅助。

​	遇到的详细问题在上一周的未来规划中有所记录！

2. ##### ORB_SLAM3算法特性。

   目前仅对代码中主函数延伸的各个函数代码进行了逐行理解，并没有进行宏观代码理解。

   不过从目前感受来看，工程项目果然还是十分稳健。很多地方加了互斥锁，并且函数分块，一个函数能尽可能在更多的地方被调用。

   在特征点提取方面，ORB_SLAM3采用ORB特征点提取，通过计算描述子与词袋向量，将不同图像帧之间的特征点利用词袋单词相同进行匹配，大大提高了相较于暴力匹配的匹配效率。

   位姿估计方面，分为三层位姿估计 -- 初始化位姿估计（通过上一帧计算当前帧预测位姿） -- 局部地图位姿估计（利用局部地图的关键帧对当前图像帧相机位姿进行估计，使用g2o图优化，同时优化路标点） -- 全局地图位姿估计（估计出现在回环检测部分，但是目前还没有看到回环检测（可能是漏看了？））。

   在适用性方面，ORB_SLAM3支持单、双目、RGB-D以及其带IMU的传感器形式。且似乎带IMU时算法处理会比较宽松（比较相信IMU数据能够给位姿估计提供更优帮助），不过IMU与相机传感器数据融合在《视觉十四讲》中没有专门讲过，所以对于原理目前还十分朦胧，需要后续学习了解。

### 未来规划：

1. 8.4 - 8.15放假约10天，调整放松。
2. 放假回来之后整体理解ORB_SLAM3算法，撰写算法大致流程，与 前端 -- 后端 -- 地图 -- 回环检测 四个线程的工作流程。
3. 学习IMU数据与相机传感器数据融合原理，尝试使用带IMU的视觉传感器检验建图效果？
4. 完成ORB_SLAM3算法学习的所有工作后开启激光SLAM的学习！

## Date 8.15 ~ 8.30

### 学习内容：

#### 1.ORB_SALM3算法流程概览

![Untitled diagram _ Mermaid Chart-2025-08-16-114103](D:\Desktop\SLAM\Pict\Untitled diagram _ Mermaid Chart-2025-08-16-114103.png)

![deepseek_mermaid_20250816_d7c566](D:\Desktop\SLAM\Pict\deepseek_mermaid_20250816_d7c566.png)

##### ①**前端（Tracking 线程）**

![Untitled diagram _ Mermaid Chart-2025-08-16-093831](D:\Desktop\SLAM\Pict\Untitled diagram _ Mermaid Chart-2025-08-16-093831.png)

**目标**：实时处理每一帧图像，估计相机位姿并决定关键帧插入
**工作流程**：

1. **初始化**（单目特有）：

   - 通过 **对极几何** 或 **单应矩阵** 计算前两帧的相对位姿
   - 三角化生成初始地图点（如 `Initializer::Initialize()`）
   - IMU单目模式会同时初始化IMU参数（重力方向、偏置等）

2. **帧处理**：

   ```C++
   // ORB-SLAM3/src/Tracking.cc
   cv::Mat Tracking::GrabImageMonocular(...) {
       // 1. 特征提取
       ExtractORB(im, 0); 
       // 2. 位姿估计（三种模式）
       if (mState == OK) {
           if (mVelocity.empty()) 
               TrackReferenceKeyFrame();  // 参考关键帧跟踪
           else 
               TrackWithMotionModel();    // 运动模型跟踪
       }
       else 
           Relocalization();              // 重定位
       // 3. 局部地图跟踪
       TrackLocalMap();
       // 4. 关键帧决策
       if (NeedNewKeyFrame())
           CreateNewKeyFrame();
   }
   ```

   - **特征提取**：提取ORB特征点（`ExtractorORB`）

   - **位姿估计**：

     - **运动模型**：基于恒定速度假设估计位姿

     - **参考关键帧**：通过**词袋匹配（BoW）**估计位姿（避免暴力匹配，提高特征点之间匹配效率）
     - **重定位**：当丢失时，通过DBoW2检索候选关键帧

   - **局部地图跟踪**：将当前帧与局部地图点匹配，优化位姿

   - **关键帧决策**（`NeedNewKeyFrame()`）：

     - 时间间隔（>15帧）
     - 跟踪点数量下降（<参考关键帧的90%）
     - 局部地图点观测不足

##### ②**后端（LocalMapping 线程）**

![Untitled diagram _ Mermaid Chart-2025-08-16-093934](D:\Desktop\SLAM\Pict\Untitled diagram _ Mermaid Chart-2025-08-16-093934.png)

**目标**：优化局部地图结构，维护地图一致性
**工作流程**：

```C++
// ORB-SLAM3/src/LocalMapping.cc
void LocalMapping::Run() {
    while (1) {
        // 1. 新关键帧处理
        ProcessNewKeyFrame();
        // 2. 地图点剔除
        MapPointCulling();
        // 3. 新地图点创建（三角化）
        CreateNewMapPoints();
        // 4. 局部BA优化
        Optimizer::LocalBundleAdjustment(...);
        // 5. 冗余关键帧剔除
        KeyFrameCulling();
    }
}
```

- **关键帧插入**：
  - 更新共视图（`covisibility graph`）
  - 更新生成树（`spanning tree`）
- **地图点管理**：
  - **三角化**：通过相邻关键帧创建新地图点（`CreateNewMapPoints()`）
  - **剔除劣质点**：观测不足/重投影误差过大
- **局部BA**：
  - 优化当前关键帧 + 共视关键帧 + 地图点
  - 使用 `g2o` 实现（`Optimizer::LocalBundleAdjustment`）
- **关键帧剔除**：删除冗余关键帧（>90%点被其他关键帧观测）

##### ③**回环检测（LoopClosing 线程）**

![Untitled diagram _ Mermaid Chart-2025-08-16-094100](D:\Desktop\SLAM\Pict\Untitled diagram _ Mermaid Chart-2025-08-16-094100.png)

**目标**：识别场景回环，校正累积误差
**工作流程**：

```C++
// ORB-SLAM3/src/LoopClosing.cc
void LoopClosing::Run() {
    while (1) {
        // 1. 检测回环候选帧
        if (DetectLoop()) {
            // 2. 计算Sim3变换（单目需估计尺度）
            if (ComputeSim3()) {
                // 3. 闭环校正
                CorrectLoop();
            }
        }
    }
}
```

1. **检测回环**（`DetectLoop()`）：

   - 基于DBoW2词袋模型检索相似关键帧
   - 连续性检测（连续3帧匹配成功）

2. **计算Sim3**（`ComputeSim3()`）：

   - 单目模式下需估计**尺度因子**（7自由度变换）
   - RANSAC求解相似变换矩阵

3. **闭环校正**（`CorrectLoop()`）：

   - **位姿图优化**：融合Sim3约束，优化Essential Graph

   ```C++
   Optimizer::OptimizeEssentialGraph(...);
   ```

   - **地图点融合**：合并重复地图点
   - **全局BA**（可选）：在独立线程执行全局优化

##### ④**可视化地图（Viewer 线程）**

![deepseek_mermaid_20250816_de4995](D:\Desktop\SLAM\Pict\deepseek_mermaid_20250816_de4995.png)

**目标**：实时展示SLAM状态和地图
**工作流程**：

```C++
// ORB-SLAM3/src/Viewer.cc
void Viewer::Run() {
    while (1) {
        // 1. 获取当前帧/地图数据
        GetCurrentState();
        // 2. 绘制相机轨迹
        DrawCameraTrajectory();
        // 3. 渲染地图点云
        DrawMapPoints();
        // 4. 显示关键帧位姿
        DrawKeyFrames();
    }
}
```

- **显示内容**：
  - **相机轨迹**：当前帧位姿 + 历史轨迹
  - **地图点云**：激活点（绿色）| 非激活点（黑色）
  - **关键帧**：相机坐标系 + 共视关系线
  - **状态信息**：跟踪状态/关键帧数/地图点数
- **多地图支持**（ORB-SLAM3特有）：
  - 同时可视化多个子地图（Atlas系统）
  - 用不同颜色区分激活地图和非激活地图

------

**单目模式特有机制**

1. **尺度漂移处理**：
   - 回环检测中通过Sim3估计尺度因子
   - 全局BA优化尺度一致性
2. **多地图系统**（Atlas）：
   - 跟踪丢失时创建新地图（`Tracking::Reset()`）
   - 重定位到旧地图时执行地图融合（`LoopClosing::MergeMaps()`）
3. **IMU融合**（IMU-MONOCULAR）：
   - **前端**：IMU预积分约束位姿估计
   - **后端**：视觉-惯性联合优化（VI-BA）



#### 2.ORB_SLAM3如何实现相机与IMU数据融合

##### 	①什么是IMU预积分？预积分的作用是什么？

​	长久以来一个问题一直困扰着我，视觉传感器与IMU数据融合，到底是哪方面进行了融合？视觉SLAM进行图像帧位姿估计的核心就是通过特征点之间的重投影找到最优变换矩阵，从而得到最优位姿估计，那再加上IMU能为这一过程贡献一份力量吗？

​	实际上，**IMU在ORB_SLAM3中只为整个运动系统测量了三个值——加速度、角速度、时间**。而IMU预积分的过程，则是基于这三个值，计算出JPa、JPg、JVa、JVg，用于计算相对**位移与速度的预积分量更新结果** + A、B、Q，用于计算**预积分量更新结果的协方差矩阵**。

![image-20250819202920492](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250819202920492.png)

  **`JPa -- 位置预积分关于加速度零偏的雅可比矩阵`**

  **`JPg -- 位置预积分关于陀螺仪零偏的雅可比矩阵`**

  **`JVa -- 速度预积分量关于加速度计零偏（bias_a）的雅可比矩阵`**

  **`JVg -- 速度预积分量关于陀螺仪零偏（bias_g）的雅可比矩阵`**

  **`δba -- 加速度的偏差值`**

s  **`δbg -- 陀螺仪的偏差值`**

![image-20250819202947689](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250819202947689.png)

**`C_{k+1}`: 更新后的协方差矩阵，包含了预积分量在当前时刻的不确定性。**

**`C_k`: 上一个时间步长的不确定性。**

**`A`: 状态转移矩阵。它描述了预积分量从 k 时刻到 k+1 时刻的线性演化。代码中的 `A.block<...>` 就是在填充这个矩阵。**

**`B`: 噪声输入矩阵。它描述了IMU的噪声如何影响预积分量的变化。代码中的 `B.block<...>` 就是在填充这个矩阵。**

**`Q`: IMU的噪声协方差矩阵。它包含了陀螺仪和加速度计的测量噪声（`Nga`）。**

 **`B、C的计算过程比较繁琐，要用到旋转矩阵dR、加速度白噪声Wacc之类的数据。`**

​	综上所述，根据IMU测量的测量值，我们最终可以得到三个数据P + V + C，其中**C（协方差矩阵）可以用来度量P、V的可置信度**，而**P、V**则代表**机器人在相邻两帧之间的相对位置与相对速度信息**。纯视觉SLAM机器人只能通过对不同帧特征点之间的运动进行位姿估计来估计自身的运动轨迹，而当遇到稀疏特征或是追踪丢失等情况时，机器人便会迷失自己在全局地图中的位置。因此，当视觉传感器与IMU数据进行融合后，IMU数据为全局地图**提供了现实世界中的绝对尺度**，即使在跟踪丢失等情况下，机器人也能凭借IMU数据精准认知自身在**全局地图乃至世界中的精准位置（纠正漂移问题）**。除此之外，**IMU数据的预积分**为SLAM的**后端优化提供了新的约束条件**，使得**后端位姿估计更加准确**；且在后端优化的过程中，**基于预积分中计算的一系列雅可比矩阵**，能够**快速计算出陀螺仪与加速度数据有些许偏差后的P、V期望值**，便于误差函数的快速计算，提高求解效率。

​	总之，IMU的预积分在进行图像追踪时直接进行，避免了将数据堆在后端优化时一个一个处理（计算雅可比矩阵十分麻烦），极大提高了后端优化的效率；同时，IMU的预积分储存了相邻两图像帧之间的相对运动信息，使得机器人的运动变得十分明确，为机器人自身定位提供了极大帮助。

##### 	②IMU数据如何作为条件约束后端优化？即P、V如何影响SLAM的后端位姿估计？

​	IMU数据融合部分位于LocalMapping.cc中Run()函数中。该函数在LocalMapping（后端）线程启动时同步开始循环执行。

​	IMU测量值数据将被设置为g2o的顶点，并设置IMU、陀螺仪、加速度约束边（连接由IMU测量值数据设置而成的顶点），利用g2o进行图优化。详细位于Optimizer::LocalInertialBA()函数中！

```C++
// 检查连续关键帧是否都有IMU数据且已完成预积分
if (pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated) {
    // 初始化预积分噪声参数
    pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
    
    // 获取两个关键帧的优化顶点（位姿、速度、陀螺仪零偏、加速度计零偏）
    VertexPose* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
    VertexVelocity* VV1 = optimizer.vertex(maxKFid + 3*(pKFi->mPrevKF->mnId) + 1);
    VertexGyroBias* VG1 = optimizer.vertex(maxKFid + 3*(pKFi->mPrevKF->mnId) + 2);
    VertexAccBias* VA1 = optimizer.vertex(maxKFid + 3*(pKFi->mPrevKF->mnId) + 3);
    
    VertexPose* VP2 = optimizer.vertex(pKFi->mnId);
    VertexVelocity* VV2 = optimizer.vertex(maxKFid + 3*(pKFi->mnId) + 1);
    VertexGyroBias* VG2 = optimizer.vertex(maxKFid + 3*(pKFi->mnId) + 2);
    VertexAccBias* VA2 = optimizer.vertex(maxKFid + 3*(pKFi->mnId) + 3);
    
    // 创建并设置IMU预积分约束边
    EdgeInertial* ei = new EdgeInertial(pKFi->mpImuPreintegrated);
    ei->setVertex(0, VP1);  // 上一关键帧位姿
    ei->setVertex(1, VV1);  // 上一关键帧速度
    ei->setVertex(2, VG1);  // 上一关键帧陀螺仪零偏
    ei->setVertex(3, VA1);  // 上一关键帧加速度计零偏
    ei->setVertex(4, VP2);  // 当前关键帧位姿
    ei->setVertex(5, VV2);  // 当前关键帧速度
    optimizer.addEdge(ei);
    
    // 创建并设置陀螺仪零偏随机游走约束边
    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0, VG1);
    egr->setVertex(1, VG2);
    egr->setInformation(pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse());
    optimizer.addEdge(egr);
    
    // 创建并设置加速度计零偏随机游走约束边
    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0, VA1);
    ear->setVertex(1, VA2);
    ear->setInformation(pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse());
    optimizer.addEdge(ear);
}
```

​	**g2o中设置的误差函数（其中之一）**：

```C++
 _error = obs - VPose->estimate().Project(VPoint->estimate(),cam_idx);
```

​	可见，尽管有IMU数据约束，但是后端优化的**本质仍是计算重投影误差（BA）**来**进行位姿估计与优化**。

> [!tip]
>
> ​	**此处理解稍有片面！实际上IMU与视觉传感器进行数据融合，确实本质仍是使误差函数最优，但是误差函数的表达式并非该g2o设置的误差函数！（详细往后面看就明白了）**

​	实际上！IMU数据和视觉数据的融合不是在边的内部计算中完成的，而是在更高层的**优化框架**中实现的。在EdgeInertial类中，也存在computeError()函数，实际上一般的边类定义中都定义了computeError()函数，但是在代码中不会主动调用。那么IMU数据是在何处起作用的？

​	这就要牵涉到g2o的精妙之处了。g2o是一种图优化求解非线性优化的方式，其由多个**顶点**与**边**构成，顶点代表着一种类型的数据，而边则相当于约束，边会连接一定数量个顶点，根据边中定义的误差计算函数，会同时优化边所连接的所有顶点，从而使顶点的数据达到理想状态。IMU与视觉传感器数据融合就是利用这一原理，将**同步优化所有状态变量**，IMU与视觉传感器通过**不同的边约束同一组状态顶点**，**同步优化**所有**状态变量**。

​	**论文中提到的视觉-惯性联合优化误差函数：**

![image-20250826120550149](../Typora Picture/image-20250826120550149.png)

![image-20250826121424188](../Typora Picture/image-20250826121424188.png)

​	**实际上**，可以看作**论文中给出的误差函数为总式**，而实际上代码中将**总误差函数分解为多个g2o边**进行分别求解。

![image-20250826123349973](../Typora Picture/image-20250826123349973.png)

##### ③偏置b的计算过程与作用

###### 一、偏置 `b` 是什么？

IMU偏置 `b` 是一个6维向量，通常分为两部分：

- **加速度计偏置** `b_a = [b_ax, b_ay, b_az]^T`
- **陀螺仪偏置** `b_g = [b_gx, b_gy, b_gz]^T`

它不是固定不变的常数，而是会随着温度、电路状态等缓慢漂移的变量。它的核心作用是**修正IMU的原始测量值**，得到更接近真实的角速度和加速度：

```
真实值 ≈ 原始测量值 - 偏置
```

###### 二、偏置 `b` 的核心作用

1. **修正测量，减少积分漂移**：这是最直接的作用。准确的偏置估计是获得可信的IMU积分结果（旋转、速度、位置）的前提。一个微小的偏置误差会在积分过程中被指数级放大。
2. **作为优化状态量，连接IMU与视觉**：在VIO中，偏置 `b` 与相机位姿、地图点一样，是**被优化器估算的状态变量**。通过将 `b` 纳入优化问题，IMU信息和视觉信息得以在一个统一的概率框架内共同修正系统状态。
3. **实现IMU的在线标定**：系统不需要依赖出厂时粗糙的标定数据。它可以在运行过程中**自动地、实时地**估计出当前环境下IMU的精确偏置，极大地提升了对低成本IMU的利用效率。

###### 三、偏置 `b` 的计算过程（优化过程）

偏置 `b` 不是被“计算”出来的，而是被“**优化**”出来的。整个过程是一个**不断迭代的期望最大化**过程：

1. **初始猜测**：通常假设偏置为0，即 `b = [0,0,0,0,0,0]^T`。基于这个初始值，进行IMU预积分，得到初始的预积分量 `(ΔR, Δv, Δp)` 和**雅可比矩阵 `J`**（如 `JRg`, `JVg`, `JVa` 等）。
2. **构建优化问题**：将以下变量设为顶点，构建图优化模型：
   - **待优化变量**：关键帧的位姿 `T`、速度 `v`、**IMU偏置 `b`**、地图点 `X`。
   - **约束边**：
     - `EdgeInertial`：连接连续两帧的 `T`, `v` 和 `b`，其误差由预积分量计算。
     - `EdgeReprojection`：连接帧的 `T` 和地图点 `X`，其误差为重投影误差。
3. **优化求解**：g2o等优化器开始工作，尝试微调所有顶点的值（包括给 `b` 一个变化量 `δb`）来最小化总误差。
4. **更新预积分量**：**这是最关键的一步**。当优化器改变 `b` 时，我们**不会**用新的 `b` 去重新积分所有IMU数据。而是使用**一阶近似**，利用步骤1中计算好的雅可比矩阵 `J` 来快速更新预积分量。
   `Δ量_{new} ≈ Δ量_{old} ○ Exp(J * δb)` （`○` 代表相应的更新操作）
   这个过程非常高效，是VIO算法实时性的保证。
5. **迭代收敛**：用更新后的预积分量重新计算误差，优化器继续迭代，直到总误差收敛到最小。此时得到的偏置 `b` 就是在当前数据下的**最优估计值**。
6. **持续跟踪**：偏置是缓慢变化的，因此这个优化过程会持续在整个SLAM运行过程中进行，不断跟踪和修正偏置的最新值。

------

###### 代码实例剖析

以您提供的代码为例，这是上述第4步（更新预积分量）中针对**旋转预积分量**的具体实现：

```C++
Eigen::Matrix3f Preintegrated::GetDeltaRotation(const Bias &b_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    // 1. 计算偏置的变化量 δb_g
    Eigen::Vector3f dbg;
    dbg << b_.bwx - b.bwx, b_.bwy - b.bwy, b_.bwz - b.bwz;

    // 2. 使用一阶近似更新旋转预积分量
    return NormalizeRotation(dR * Sophus::SO3f::exp(JRg * dbg).matrix());
}
```

- **`b_`**：优化器提议的**新偏置值**。
- **`b`**：预积分时所用的**原始偏置值**（存储在类成员变量中）。
- **`dbg`**：这就是偏置的变化量 **`δb_g`**。它是优化过程的**输出**，是优化器为了最小化误差而尝试做出的调整。
- **`JRg`**：这是预积分过程中计算并保存下来的**雅可比矩阵**。它编码了“旋转预积分量对陀螺仪偏置的敏感度”。
- **`JRg \* dbg`**：根据偏置的变化量 `δb_g`，估算出旋转预积分量需要相应做出的调整量（一个李代数向量）。
- **`Sophus::SO3f::exp(JRg \* dbg)`**：通过指数映射，将调整量（李代数）转换为一个旋转矩阵（李群），代表所需的**校正旋转**。
- **`dR \* ...`**：将原始的旋转预积分量 `dR` 右乘这个校正旋转矩阵，得到**更新后**的、更接近用新偏置 `b_` 积分所能得到的旋转预积分量。
- **`NormalizeRotation`**：保证输出矩阵的正交性，消除计算过程中的数值误差。

**这段代码的核心作用就是：当优化器想要试探一个新的偏置值 `b_` 时，它能极其高效地返回对应的旋转预积分量，从而让优化器能够评估在这个新偏置下总误差是大还是小，进而决定下一步的优化方向。** 它是连接“偏置优化”和“IMU测量”的桥梁，是整个联合优化得以实现的技术基石。



#### 3.ORB_SLAM3工程深入挖掘

##### 	①ORB_SLAM3工程原生支持单目、双目、RBG-D + IMU，似乎也原生支持相机数据实时传入（不用借助ROS2，直接读取USB相机数据）

![image-20250818103839801](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250818103839801.png)

​	首先，标定相机内参，将相机内参与ORB特征点参数按格式写进yaml文件。（参考/Monocular/RealSense_D435i.yaml）

​	其次，根据相机驱动文件（参考/Monocular/mono_realsense_D435i.cc），传入所需要参数（若使用不同相机，驱动文件需要微调？or重新写一份？）

​	最后，传入所需参数（词典路径、SLAM参数（即第一步的yaml文件）、SLAM轨迹文件保存位置（可选传入）），即可实现实时SLAM稀疏建图。

> [!note]
>
> ​	**仅原生支持D435i与T265相机实时驱动！！！**（否则需要自己写代码！！！过于复杂，不如直接ROS2通信！！！）



#### 4.ORB_SLAM3代码阅读感想

​	①**代码量巨大**！！！随便一个文件就是一两千行的代码，核心线程更是四五千不止，而这样的文件有二十几个！！！阅读下来非常非常非常费力，需要足够的耐心与专注度完成注释与阅读。

​	②**牵涉过多C++特性的语法（偏工程化）**，如互斥锁与大量的面向对象编程。尽管这样的编程方式对新手阅读起来非常不友好，但是使得实现不同功能的代码之间既能够互相调用，又能够保证代码管理十分方便（层次清晰）。这样的编程方式是我后续需要加强的！

​	③ORB_SLAM虽然说算是SLAM最经典，也是较为古老的一批视觉SLAM算法，但是其在四大线程 -- 前端（Track）、后端（LocalMapping）、回环检测（LoopClothing）、地图（Map）中的代码实现细节，都**相较于书上讲的要复杂的多**（为工程项目的稳健性考量？难以想象这是如何写出来的）。目前位置虽然说基本看完了ORB_SLAM3的整个工作流程，但还是有些一头雾水（让我动手优化or写代码仍然无从下手），待看完ORB_SALM3衍生出的相关论文或许能好一些？

​	④**头晕....**

### 未来规划：

​	① 集中准备数学建模大赛！ 

​	② 完成ORB_SALM3衍生的论文阅读，学习ORB_SLAM3各线程的优化策略！

​	③ 根据论文所述回头看代码加深理解！ （一定要弄清楚四大线程的作用与具体实现过程！）

​	④ 复习线性代数 + 学习概率论知识，回头重新复习《视觉十四讲》理论部分，自行推导并理解ORB_SLAM3算法论文与代码中的后端优化过程！**（Important）**

## Date 9.8 ~ 

### 学习内容：

#### 0.基础知识普及计划 -- Git

[简介 - Git教程 - 廖雪峰的官方网站](https://liaoxuefeng.com/books/git/introduction/index.html)

##### 0.1.创建版本库

###### 〇 初始化当前文件夹为Git库

![image-20250912103455413](../Typora Picture/image-20250912103455413.png)

###### ①将文件添加进git库中

![image-20250911211036626](../Typora Picture/image-20250911211036626.png)

###### 	②查看仓库中所有文件的状态

![image-20250911211716540](../Typora Picture/image-20250911211716540.png)

###### 	③查看被修改文件的具体修改内容

![image-20250911211742515](../Typora Picture/image-20250911211742515.png)

##### 0.2版本回退与还原

###### 	①版本回退

​	**查看当前文件的所有版本**

![image-20250911211911068](../Typora Picture/image-20250911211911068.png)

​	**选择一个状态进行回退**

![image-20250911212050041](../Typora Picture/image-20250911212050041.png)

​	**回到上一个版本后重新回退到当前版本**（回退到上一版本后当前版本会自动消失，需要使用识别码索引）

​	**法一：**

![image-20250911212345311](../Typora Picture/image-20250911212345311.png)

​	**法二：**
![image-20250911212529758](../Typora Picture/image-20250911212529758.png)

###### 	②撤销修改 

![image-20250911215810435](../Typora Picture/image-20250911215810435.png)

​	撤销修改分为两种情况，**一为修改后未使用add添加至缓存区**，**二为修改后已使用add添加至缓存区。**

![image-20250911215230138](../Typora Picture/image-20250911215230138.png)

​	如果想**将已添加至缓存区的修改撤回至添加至缓存区之前（将缓存区中的文件拿出）**，则：

![image-20250911215411754](../Typora Picture/image-20250911215411754.png)

###### ③ 删除文件

![image-20250911220249568](../Typora Picture/image-20250911220249568.png)
	git add添加文件后删除，使用git status会显示**缓存区与当前工作区的区别**（工作区已不存在文件)。

​	**Choice1：从版本库中删除该文件**

![image-20250911220427609](../Typora Picture/image-20250911220427609.png)

​	**Choice2：删错文件，恢复文件**

![image-20250911220537569](../Typora Picture/image-20250911220537569.png)

##### 0.3远程仓库

###### ① GitHub远程仓库创建SSH链接

​	**本地链接创建**

![image-20250912075002964](../Typora Picture/image-20250912075002964.png)

​	创建成功的**Key保存在主目录下（C:/User/Li/.ssh）**

​	**GitHub创建链接**

![image-20250912080019274](../Typora Picture/image-20250912080019274.png)

###### ② 将本地代码添加到远程库GitHub中

​	**在GitHub中创建新Git库**

![image-20250912081748958](../Typora Picture/image-20250912081748958.png)

​	**建立与远程Git库的链接**

![image-20250912082001229](../Typora Picture/image-20250912082001229.png)

```shell
git remote add [远程库名字] [git仓库链接]
```

![image-20250912082054265](../Typora Picture/image-20250912082054265.png)

​	**将本地文件上传至远程Git库中**

![image-20250912082211965](../Typora Picture/image-20250912082211965.png)

###### ③ 删除远程Git库

![image-20250912082309025](../Typora Picture/image-20250912082309025.png)

​	实际上只是解除了**本地和远程的绑定关系**！

###### ④ 从远程库克隆

![image-20250912094214340](../Typora Picture/image-20250912094214340.png)

​	与建立远程Git库连接类似，此处也是直接**使用SSH链接对远程Git库进行克隆**即可。

##### 0.4 分支管理

![image-20250912094757686](../Typora Picture/image-20250912094757686.png)

###### ①创建、合并与删除分支

![image-20250912095043482](../Typora Picture/image-20250912095043482.png)

​	**创建分支，名字可以自定义。**

![image-20250912095206735](../Typora Picture/image-20250912095206735.png)

​	查看当前分支，**当前所在分支前会标记*号**。

![image-20250912095600995](../Typora Picture/image-20250912095600995.png)

![image-20250912101203086](../Typora Picture/image-20250912101203086.png)

​	**切换回主分支。**（也可以修改分支名字切换到其他分支）

![image-20250912095739881](../Typora Picture/image-20250912095739881.png)

​	在**dev分支进行修改的文件，在master主分支中并没有同步**。

![image-20250912095704016](../Typora Picture/image-20250912095704016.png)

​	因此，我们需要**对master分支与dev分支进行合并**。

![image-20250912095905846](../Typora Picture/image-20250912095905846.png)

​	**对dev分支进行删除**

![image-20250912102915145](../Typora Picture/image-20250912102915145.png)

​	如果在**mater主分支与dev分支同时对文件进行了修改**，merge时会发生冲突，**打开编辑器会提示你选择一项进行修改。**

​	修改完成后即可再次手动提交。

![image-20250912103051490](../Typora Picture/image-20250912103051490.png)

​	**查看分支合并日志。**

###### 	② 分支管理策略

![image-20250912110750586](../Typora Picture/image-20250912110750586.png)

​	**合并分支时加上 --no-ff 可以使用普通模式进行合并**，合并后的历史有分支，能看出来曾经做过合并，而**fast forward合并则看不出来曾经有合并**。

###### 	③Bug管理

![image-20250913092330217](../Typora Picture/image-20250913092330217.png)

![image-20250912120952469](../Typora Picture/image-20250912120952469.png)

​	当**遇到突发事件，需要创建一个新分支**，并且**需要保存当前未完成工作的分支**，则**使用stash功能将工作现场暂存**。

![image-20250912121138958](../Typora Picture/image-20250912121138958.png)

​	当**处理完突发事件后，回到将工作现场暂存的分支**，使用**git stash list 查看当前有多少个工作现场被储存**。

![image-20250912121301088](../Typora Picture/image-20250912121301088.png)

![image-20250912121422438](../Typora Picture/image-20250912121422438.png)

​	选择一种方式来恢复工作现场—— **不删除工作现场的暂存并恢复 or 删除工作现场的暂存并恢复**。

![image-20250912174521820](../Typora Picture/image-20250912174521820.png)

​	然而**当修复bug回到暂存工作区后，工作中的bug仍未被修改，此时则可以使用cherr-pick快速同步修改**。

​	即：**继续使用git stash将当前工作区暂存**；再**回到master主分支查看文件提交记录，找到修改bug的commit并复制部分代号**；**使用cherry-pick命令将两部分修改合并，即完成bug修改同步**！

![image-20250913100210786](../Typora Picture/image-20250913100210786.png)

###### ④ Feature管理

![image-20250913092304252](../Typora Picture/image-20250913092304252.png)

![image-20250913092820531](../Typora Picture/image-20250913092820531.png)

​	实际上就是**开了个新分支feature**，在**新分支feature上修改然后在dev分支中与新分支feature进行合并**。

![image-20250913093104838](../Typora Picture/image-20250913093104838.png)

​	但是，**如果新内容不需要进行合并**了，**删除feature分支时只需要将-d写为-D**即可**强制删除有内容的分支**。

![image-20250913100146088](../Typora Picture/image-20250913100146088.png)

###### ⑤ 多人协作

![image-20250913094045963](../Typora Picture/image-20250913094045963.png)

![image-20250913094124373](../Typora Picture/image-20250913094124373.png)

​	**查看远程仓库详细信息。**

![image-20250913094241603](../Typora Picture/image-20250913094241603.png)

​	推送时，可以**选择分支进行推送**！一般而言都是在**dev分支上进行开发**工作，**bug分支用于修复bug**，**feature分支用于添加新功能**。是否推送至远程仓库，**取决于是否有人需要与你的工作同步**！

![image-20250913094623941](../Typora Picture/image-20250913094623941.png)

![image-20250913095111575](../Typora Picture/image-20250913095111575.png)

​	单纯**从远程仓库origin中克隆仓库，无法克隆dev分支**，需要**手动创建远程仓库origin的dev分支到本地**，从而在本地进行开发！

![image-20250913095634347](../Typora Picture/image-20250913095634347.png)

​	当**两个人向dev分支中提交了一份有冲突的文件**，git会报错。此时需要**使用pull将已上传的文件下拉**，在**本地解决冲突后再重新上传**。

![image-20250913095846625](../Typora Picture/image-20250913095846625.png)

​	然而，**在pull之前，远程仓库需要知道pull哪一部分**，因此需要**指定本地分支与远程分支的链接**！（如图）

**总结！！！**

![image-20250913100110905](../Typora Picture/image-20250913100110905.png)

##### 0.5 标签管理

![image-20250913101132196](../Typora Picture/image-20250913101132196.png)

###### 	① 创建标签

![image-20250913101846935](../Typora Picture/image-20250913101846935.png)

![image-20250913103459526](../Typora Picture/image-20250913103459526.png)

​	实际上，**打标签是为上一次commit打标签**（而**不是为即将commit打标签**）。此外，还能**使用-a与-m指定当前标签名与其说明文字**。

![image-20250913103135918](../Typora Picture/image-20250913103135918.png)

​	使用**git tag可以查看所打标签的情况**。

![image-20250913103212735](../Typora Picture/image-20250913103212735.png)

​	使用**log找到未打标签的版本commit**，再使用 **git tag [tag] [commit_id]** 可以对该版本**补打标签**。

![image-20250913104022755](../Typora Picture/image-20250913104022755.png)

​	使用 **git show [tag] 能够看到当前标签的说明文字**。

![image-20250913104159576](../Typora Picture/image-20250913104159576.png)

###### ② 操作标签

![image-20250913104850035](../Typora Picture/image-20250913104850035.png)

​	**将标签推送到远程**仓库（创建的标签默认**不会自动推送到远程仓库**！！）

​	**删除本地标签：**

![image-20250913104641639](../Typora Picture/image-20250913104641639.png)

​	**删除远程仓库标签：**

![image-20250913105139526](../Typora Picture/image-20250913105139526.png)

![image-20250913105345934](../Typora Picture/image-20250913105345934.png)

#### 1.当务之急速通概率论与数理统计。

#### 2.概率论与数理统计与SLAM基础数学知识混学。

#### 3.混合学习后重新读论文！！！SLAM代码重读计划延后！！！

### 未来规划：
=======


## 目录

### 	一.Date 5.16 -5.23 周总结  [Click](#一)

### 	二.Date5.23 - Date6.7周总结 [Click](#二)

### 	三.Date 7.7 学习记录 [Click](#三)

### 	四.Date 7.14 ~ 7.19 周总结 [Click](#四)

### 	五.Date7.21 ~ 7.27 周总结 [Click](#五)

### 六.Date7.28 ~ 8.2 周总结 [Click](#六)

---

## Date 5.16 -5.23 周总结  <span id="一"> </span>

### 	**学习内容：**

#### 	**1.《视觉SLAM十四讲：从理论到实践》**

​	第二部分（实践应用）-- 第七讲（视觉里程计）7.1--7.6

![image-20250524004028069](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524004028069.png)

​	通过对该讲目前的学习

​	①掌握了从图像中提取特征点的方法--寻找关键点与描述子。SLAM方案中，为了质量与性能兼得，我们采用提取ORB特征点的方式。主要步骤为，找关键点，对每个关键点计算描述子。

![image-20250524102850225](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524102850225.png)

​	当我们得到两张由于相机的位姿发生变换而拍摄得到的图片后，我们可以通过计算两张图片描述子之间的汉明距离，给两张图片中的特征点进行匹配。

​	![image-20250524102914113](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524102914113.png)

​	然而，为了获取质量更高的特征匹配，我们通常会对一对匹配点间的汉明距离与设定的阈值进行比较，当汉明距离超过三十 且 不大于两倍的所有匹配点间最小汉明距离 时对该对点进行保留。

​	![image-20250524102944750](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524102944750.png)

​	且均已在Ubuntu20.04环境下成功使用代码复现。

​	②对对极几何知识进行了基本学习。对极几何用于对相机运动的运动进行计算（主要解决2D-2D（已知一堆2D点）求解位姿问题），得到的结果为**相机由1处运动到2处发生的旋转**（得到旋转矩阵R）与由1处运动到2处发生的平移（得到平移量t）。**（注意！！！2D-2D通常以第一帧相机的坐标系为参考系，故解出来的R、t均为相对于第一帧相机的相对变化关系。 而3D-2D则不一样，3D-2D以世界坐标系为参考系，解出来的R、t是将世界坐标系变换到相机坐标系的变换关系。）**

​	该部分知识牵涉大量线性代数知识，大量推导过程。要得到R，t，首先需要根据对极约束计算出本质矩阵E或基本矩阵F，再利用E或F求解R与t。其中需要辨析的是，要求得本质矩阵E需要知道三维空间中特征点在两次相机不同位姿下的归一化坐标x1与x2；而要求得基本矩阵则需要知道三维空间中特征点在两次相机不同位姿下的像素坐标p1与p2。要解出R与t，（我倾向于使用本质矩阵E解R、t），需要使用奇异值分解（SVD）方法，从而解出两对不同的R与t，根据特征点相对于相机位置深度为正，可以排除掉三组情况，得到正确的旋转矩阵R与平移量t。

​	当两特征点处于同一平面上时（特征点共面or相机发生纯旋转），此时无法用本质矩阵与基本矩阵解出相机位姿R、t，需要引入单应矩阵H。然后可利用数值法、解析法解R、t（书中未详细提及）。

![image-20250524002523255](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524002523255.png)

​	实际上，上述复杂的数学推导在代码中直接调用OpenCV函数即可。比如求解本质矩阵E，经需要传入特征点的像素坐标p1、p2与相机的内参（焦距与光心距），findEssentialMat函数会自动将p1、p2转换为归一化坐标x1、x2进行计算，得到本质矩阵E。基本矩阵F需要传入特征点的像素坐标p1、p2与八点法（FM_8POINT），findFundamentalMat函数会自动计算出F。求解单应矩阵也是如此，传入特征点p1、p2后传入随机采样一致性（RANSAC）方法求解（RANSAC书中未详细提及，仅仅说优于最小二乘解法，适用于很多带错误数据的情况，能够处理带有错误匹配的数据）。求解R、t也只用调用recoverPose并传入本质矩阵E，特征点坐标p1、p2，光心距、焦距即可。

![image-20250524002544633](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524002544633.png)

​	通过本质矩阵解出R、t后，可以带回对极化约束之中检验x1.t * E * x2 与0的误差（误差＜1e-3则说明解出R、t较为精准）。（E = t^R）

​		![image-20250524002617816](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524002617816.png)

​		均已在Ubuntu中代码复现。

​		③单目相机通过R、t 估计特征点深度 -- 三角测量。得到深度，共分为四步。首先，又对极几何约束进行列式。其次，做成x2的反对称矩阵(x2^)（等价于x2叉乘......）。然后，通过等式左侧为0，且R、t，x1、x2已知解出相机在第一个位姿下特征点的深度s1。最后通过s1，解出相机在第二个位姿下特征点的深度s2。从而该对点由2D坐标通过计算得到3D空间坐标。

![481bdec463c2c256922b6f8c514b6a1](C:\Users\Li\Documents\WeChat Files\wxid_3j6yc6bus6a732\FileStorage\Temp\481bdec463c2c256922b6f8c514b6a1.png)

​		与求解R、t一样，代码实现上无需考虑那么多数学过程，直接调用triangulation函数并传入特征点p1、p2、匹配点对、R、t与用于接受三维点结果的vector<Point3d>容器即可。

​	![image-20250524103119093](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524103119093.png)

​	![image-20250524103151198](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524103151198.png)			

​		均在Ubuntu复现。

​		④了解了PnP与ICP的概念与相关知识。PnP用于利用已知一对2D-3D的点对求解R、t的问题（2D指像素平面上特征点的投影点为2D，而3D指特征点为3D点）。相较于2D-2D，2D-3D对与所需要的点对数由8个及以上锐减至最少3个点对。且最后将2D-3D的点对转化为3D-3D的点对，坐标系也由世界坐标系转换到相机坐标系上。ICP概念利用迭代最近点思想，在3D-3D点对之间，通过迭代的方式最小化两组点云间距离。（具体如何如何运用目前还不清楚。）

​		PnP问题有很多解法，大致分为直接求解、非线性优化求解两类型。其中直接求解包含P3P，直接线性变换(DLT)，EPnP，UPnP。非线性优化求解即为构造最小二乘问题并迭代求解（光束法平差(BA)）。

​		P3P即为其中用三对点对求解2D-3D下R、t的方法。由于2D-3D的特征点坐标系为世界坐标系，而非相机坐标系中坐标，最终通过计算可以得到空间中特征点在相机坐标系下的坐标，同时也得到像素平面上投影点的3D坐标，从而变为3D-3D问题。（后续过程解出R、t还未学到）

​		DLT也是用于处理3D-2D问题，通过一组已知世界坐标系下的特征点P与其投影在像素平面上的坐标p解出由世界坐标系变换到相机坐标系的变换关系R、t。即p = MR。（此处的M为旋转矩阵R与平移量p的增广矩阵），将M解出后，即可通过系列数学方法解出R、t。

​		由于该知识的实践部分还没学到，故没有代码成功展示。

#### 	2.《自动驾驶与机器人中的SLAM技术：从理论到实践》

​	主要对二章进行了学习，而第二章大多为上一本书前六章知识的回顾。

​	![image-20250524014007708](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250524014007708.png)

### 	学习感想：

​	1.**前期学习规划错误**，过于想赶进度进入激光SLAM学习。但在经过调研与思考后，其实先完成对视觉SLAM的学习对新手来说会更好，能够为学好激光SLAM打好基础。且后续研究方向包含多传感器融合，故对于视觉SLAM的学习也是不可或缺的。故目前从激光SLAM调头继续学习视觉SLAM部分。

​	2.**数学知识欠缺严重。**大多数数学知识相当于对于我来说完全是新知识，边看书边学，尽管每个数学公式都跟着推了，但对知识的印象仍不深。后续或考虑快速过一遍书目内容？不死磕一个地方的推理与证明，侧重实际代码实现？（代码实现上基本都是调用库函数，不牵扯这么多的数学推导证明）但数学知识也不能直接放弃，快速过完一遍，对视觉SLAM算法有足够了解后或许对众多不熟悉数学知识之间能有重点与非重点之分？然后再返回学习？

​	3.对Eigen3、Pangolin、OpenCV**函数库的调用仍然不熟**，仍然停留在代码模仿的层面。是否要专门学习OpenCV等重要的库函数使用？还是说在实践中慢慢增加对其掌握程度？



### g2o 学习：（参考[SLAM从0到1——6. 图优化g2o：从看懂代码到动手编写（长文） - 知乎](https://zhuanlan.zhihu.com/p/121628349)）

1.图优化中的点是相机位姿，即优化变量（状态变量）。

2.图优化中的边是指位姿之间的变换关系，通常表示误差项。

官方文档中经典的g2o框架：

![img](https://pic3.zhimg.com/v2-7f06dfa0db13584f48d6c56712d94b50_1440w.jpg)

对这个结构框图做一个简单介绍（注意图中三种箭头的含义（右上角注解））：

（1）整个g2o框架可以分为上下两部分，两部分中间的连接点：**SparseOptimizer 就是整个g2o的核心部分。**

（2）往上看，SparseOpyimizer其实是一个[Optimizable Graph](https://zhida.zhihu.com/search?content_id=115158590&content_type=Article&match_order=1&q=Optimizable+Graph&zhida_source=entity)，从而也是一个**超图（HyperGraph）**。

（3）**超图有很多顶点和边**。顶点继承自 [Base Vertex](https://zhida.zhihu.com/search?content_id=115158590&content_type=Article&match_order=1&q=Base+Vertex&zhida_source=entity)，也即OptimizableGraph::Vertex；而边可以继承自[ BaseUnaryEdge](https://zhida.zhihu.com/search?content_id=115158590&content_type=Article&match_order=1&q=+BaseUnaryEdge&zhida_source=entity)（单边）,[ BaseBinaryEdge](https://zhida.zhihu.com/search?content_id=115158590&content_type=Article&match_order=1&q=+BaseBinaryEdge&zhida_source=entity)（双边）或[BaseMultiEdge](https://zhida.zhihu.com/search?content_id=115158590&content_type=Article&match_order=1&q=BaseMultiEdge&zhida_source=entity)（多边），它们都叫做OptimizableGraph::Edge。

（4）往下看，SparseOptimizer包含一个**优化算法部分OptimizationAlgorithm**，它是通过OptimizationWithHessian 来实现的。其中迭代策略可以从**[Gauss-Newton](https://zhida.zhihu.com/search?content_id=115158590&content_type=Article&match_order=1&q=Gauss-Newton&zhida_source=entity)（高斯牛顿法，简称GN）、 Levernberg-Marquardt（简称LM法）、Powell's dogleg 三者中间选择一个**（常用的是GN和LM）。(列文伯格法与高斯牛顿法差别在于H是否+ λI)

（5）对优化算法部分进行求解的时**求解器solver，它实际由BlockSolver组成**。BlockSolver由两部分组成：**一个是SparseBlockMatrix**，它由于求解稀疏矩阵(雅克比和海塞)；**另一个部分是LinearSolver**，它用来求解线性方程 HΔx=−b 得到待求增量，因此这一部分是非常重要的，它可以从PCG/CSparse/Choldmod选择求解方法。



**框架搭建步骤：**

**整体按照结构框图从下到上逐渐搭建（从底层到顶层），共分为六步：**

```cpp
typedef g2o::BlockSolver< g2o::BlockSolverTraits<3,1> > Block;  // 每个误差项优化变量维度为3，误差值维度为1
//另一种写法（更直接，直接创建底层Block求解器）
typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> BlockSolverType;
```

**①创建线性求解器Linear Solver（此处若要解雅可比矩阵or海塞矩阵则选择创建SparseBlockMatrix）**

```cpp
/*************** 第1步：创建一个线性求解器LinearSolver*************************/
Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); 

//另一种写法（更直接，直接创建底层线性求解器）
typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
```

**②创建BlockSolver，传入①中定义的Linear Solver求解器初始化。**

```cpp
/*************** 第2步：创建BlockSolver。并用上面定义的线性求解器初始化**********/
Block* solver_ptr = new Block( linearSolver )
```

**③创建总求解器Solver，并从GN（GaussNewton）/LM(Lervernberg-Marquardt）/DogeLeg选一个作为迭代策略，在传入②中定义的块求解器BlcokSolver初始化。**

```cpp
/*************** 第3步：创建总求解器solver。并从GN, LM, DogLeg 中选一个，再用上述块求解器BlockSolver初始化****/
g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );

//另一种写法（更直接，直接创建总求解器solver并选择优化方法，并传入包含线性求解器LinearSolverType的block求解器BlockSolverType，分别使用独享指针也使其更安全）(②、③步合一)
auto solver = new g2o::OptimizationAlgorithmLevenberg(
	g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
);
```

**④创建图优化的核心：稀疏优化器SparseOptimizer，并传入总求解器Solver初始化，打开调试输出（即在优化过程中输出信息）。**

```cpp
/*************** 第4步：创建图优化的核心：稀疏优化器（SparseOptimizer）**********/
g2o::SparseOptimizer optimizer;     // 图模型
optimizer.setAlgorithm( solver );   // 设置求解器
optimizer.setVerbose( true );       // 打开调试输出
```

**⑤定义图的定点和边，并添加到稀疏优化器SparseOptimizer中（SparseOptimizer.addVertex()  SparseOptimizer.addEdge() ）。**

```cpp
/*************** 第5步：定义图的顶点和边。并添加到SparseOptimizer中**********/
CurveFittingVertex* v = new CurveFittingVertex(); //往图中增加顶点
v->setEstimate( Eigen::Vector3d(0,0,0) );
v->setId(0);
optimizer.addVertex( v );
for ( int i=0; i<N; i++ )    // 往图中增加边
    {
  CurveFittingEdge* edge = new CurveFittingEdge( x_data[i] );
  edge->setId(i);
  edge->setVertex( 0, v );                // 设置连接的顶点
  edge->setMeasurement( y_data[i] );      // 观测数值
  edge->setInformation( Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma) ); // 信息矩阵：协方差矩阵之逆
  optimizer.addEdge( edge );
	}
```

**⑥设置优化参数，开始执行优化（设置迭代次数）**

```cpp
/*************** 第6步：设置优化参数，开始执行优化**********/
optimizer.initializeOptimization();
optimizer.optimize(100);    //设置迭代次数
```



**六步的详细解释：**

**（1）创建一个线性求解器LinearSolver**

这一步中我们可以选择不同的求解方式来求解线性方程 HΔx=−b ，g2o中提供的求解方式主要有：

- LinearSolverCholmod ：使用sparse cholesky分解法，继承自LinearSolverCCS。
- LinearSolverCSparse：使用CSparse法，继承自LinearSolverCCS。
- LinearSolverPCG ：使用preconditioned conjugate gradient 法，继承自LinearSolver。
- LinearSolverDense ：使用dense cholesky分解法，继承自LinearSolver。
- LinearSolverEigen： 依赖项只有eigen，使用eigen中sparse Cholesky 求解，因此编译好后可以方便的在其他地方使用，性能和CSparse差不多，继承自LinearSolver。

可以对照上面程序的代码去看求解方式在哪里设置。

**（2）创建BlockSolver，并用定义的线性求解器初始化**

BlockSolver有两种定义方式：

```cpp
// 固定变量的solver。 p代表pose的维度（是流形manifold下的最小表示），l表示landmark的维度
using BlockSolverPL = BlockSolver< BlockSolverTraits<p, l> >;

// 可变尺寸的solver。Pose和Landmark在程序开始时并不能确定，所有参数都在中间过程中被确定。
using BlockSolverX = BlockSolverPL<Eigen::Dynamic, Eigen::Dynamic>;
```

此外g2o还预定义了以下几种常用类型：

- BlockSolver_6_3 ：表示pose 是6维，观测点是3维，用于3D SLAM中的BA。
- BlockSolver_7_3：在BlockSolver_6_3 的基础上多了一个scale。
- BlockSolver_3_2：表示pose 是3维，观测点是2维。

**（3）创建总求解器solver**

注意看程序中只使用了一行代码进行创建：右侧是初始化；左侧含有我们选择的迭代策略，在这一部分，我们有三迭代策略可以选择：

- g2o::OptimizationAlgorithmGaussNewton
- g2o::OptimizationAlgorithmLevenberg
- g2o::OptimizationAlgorithmDogleg

**（4）创建图优化的核心：稀疏优化器**

根据程序中的代码示例，创建稀疏优化器：

```cpp
g2o::SparseOptimizer  optimizer;
```

设置求解方法：

```cpp
SparseOptimizer::setAlgorithm(OptimizationAlgorithm* algorithm)
```

设置优化过程输出信息：

```cpp
SparseOptimizer::setVerbose(bool verbose)
```

**（5）定义图的顶点和边，并添加到SparseOptimizer中**

看下面的具体讲解。

**（6）设置优化参数，开始执行优化**

设置SparseOptimizer的初始化、迭代次数、保存结果等。

初始化：

```cpp
SparseOptimizer::initializeOptimization(HyperGraph::EdgeSet& eset)
```

设置迭代次数：

```cpp
SparseOptimizer::optimize(int iterations,bool online)
```

————————————————————————

下面专门讲讲第5步：**定义图的顶点和边**。这一部分使比较重要且比较难的部分，但是如果要入门g2o，这又是必不可少的一部分

#### 1. 点 Vertex

在g2o中定义Vertex有一个通用的类模板：BaseVertex。在结构框图中可以看到它的位置就是HyperGraph继承的根源。

同时在图中我们注意到BaseVertex具有两个参数D/T，**这两个参数非常重要**，我们来看一下：

- D 是int 类型，表示vertex的最小维度，例如3D空间中旋转是3维的，则 D = 3（为BaseVertex<,>第一空所填）
- T 是待估计vertex的数据类型，例如用四元数表达三维旋转，则 T 就是Quaternion 类型（为BaseVertex<,>第二空所填）

```cpp
static const int Dimension = D; ///< dimension of the estimate (minimal) in the manifold space

typedef T EstimateType;
EstimateType _estimate;
```

特别注意的是这个D不是顶点(状态变量)的维度，而是**其在流形空间(manifold)的最小表示。**

**>>>如何自己定义Vertex**

在我们动手定义自己的Vertex之前，可以先看下g2o本身已经定义了一些常用的顶点类型：

```cpp
ertexSE2 : public BaseVertex<3, SE2>  
//2D pose Vertex, (x,y,theta)

VertexSE3 : public BaseVertex<6, Isometry3> //Isometry3使欧式变换矩阵T，实质是4*4矩阵
//6d vector (x,y,z,qx,qy,qz) (note that we leave out the w part of the quaternion)

VertexPointXY : public BaseVertex<2, Vector2>
VertexPointXYZ : public BaseVertex<3, Vector3>
VertexSBAPointXYZ : public BaseVertex<3, Vector3>

// SE3 Vertex parameterized internally with a transformation matrix and externally with its exponential map
VertexSE3Expmap : public BaseVertex<6, SE3Quat>

// SBACam Vertex, (x,y,z,qw,qx,qy,qz),(x,y,z,qx,qy,qz) (note that we leave out the w part of the quaternion.
// qw is assumed to be positive, otherwise there is an ambiguity in qx,qy,qz as a rotation
VertexCam : public BaseVertex<6, SBACam>

// Sim3 Vertex, (x,y,z,qw,qx,qy,qz),7d vector,(x,y,z,qx,qy,qz) (note that we leave out the w part of the quaternion.
VertexSim3Expmap : public BaseVertex<7, Sim3>
```

但是！如果在使用中发现没有我们可以直接使用的Vertex，那就需要自己来定义了。一般来说定义Vertex需要重写这几个函数（注意注释）：

```cpp
virtual bool read(std::istream& is);
virtual bool write(std::ostream& os) const;
// 分别是读盘、存盘函数，一般情况下不需要进行读/写操作的话，仅仅声明一下就可以

virtual void oplusImpl(const number_t* update);
//顶点更新函数

virtual void setToOriginImpl();
//顶点重置函数，设定被优化变量的原始值。
```

请注意里面的**oplusImpl函数，是非常重要的函数**，主要用于优化过程中增量△x 的计算。根据增量方程计算出增量后，**通过这个函数对估计值进行调整**，因此该函数的内容要重视。

根据上面四个函数可以得到定义顶点的基本格式：

```cpp
class myVertex: public g2o::BaseVertex<Dim, Type> //初始化g2o库中的自定义顶点BaseVertex，并传入参数优化变量维数(Dimension)与数据类型(Type)
  {
      public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      myVertex(){}

      virtual void read(std::istream& is) {}
      virtual void write(std::ostream& os) const {}

      virtual void setOriginImpl()
      {
          _estimate = Type();
      }
      virtual void oplusImpl(const double* update) override
      {
          _estimate += update;
      }
  };
```

如果还不太明白，那么继续看下面的实例：

```cpp
class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // 字节对齐

    virtual void setToOriginImpl() // 重置，设定被优化变量的原始值
    {
        _estimate << 0,0,0;
    }

    virtual void oplusImpl( const double* update ) // 更新
    {
        _estimate += Eigen::Vector3d(update);   //update强制类型转换为Vector3d
    }
    // 存盘和读盘：留空
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
};
```

另外值得注意的是，优化变量更新**并不是所有时候都可以像上面两个一样直接 += 就可以**，这要**看优化变量使用的类型（是否对加法封闭）**。

**>>> 向图中添加顶点**

接着上面定义完的顶点，我们把它添加到图中：

```cpp
CurveFittingVertex* v = new CurveFittingVertex(); // 创建顶点实例
v->setEstimate( Eigen::Vector3d(0,0,0) )；  // 设定初始值
v->setId(0);                               // 定义节点编号
optimizer.addVertex( v );                  // 把节点添加到图中
```

三个步骤对应三行代码，注释已经解释了作用。

#### 2.边 Edge

图优化中的边：BaseUnaryEdge，BaseBinaryEdge，BaseMultiEdge 分别表示一元边，两元边，多元边。

顾名思义，一元边可以理解为一条边只连接一个顶点，两元边理解为一条边连接两个顶点（常见），多元边理解为一条边可以连接多个（3个以上）顶点。

以最常见的二元边为例分析一下他们的参数：D, E, VertexXi, VertexXj：

- D 是 int 型，表示测量值的维度 （dimension）
- E 表示测量值的数据类型
- VertexXi，VertexXj 分别表示不同顶点的类型

```cpp
BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>
```

上面这行代码表示二元边，参数1是说测量值是2维的；参数2对应测量值的类型是Vector2D，参数3和4表示两个顶点也就是优化变量分别是三维点 VertexSBAPointXYZ，和李群位姿VertexSE3Expmap。

**>>> 如何定义一个边**

除了上面那行定义语句，还要复写一些重要的成员函数：

```cpp
virtual bool read(std::istream& is);
virtual bool write(std::ostream& os) const;
// 分别是读盘、存盘函数，一般情况下不需要进行读/写操作的话，仅仅声明一下就可以

virtual void computeError();
// 非常重要，是使用当前顶点值计算的测量值与真实测量值之间的误差

virtual void linearizeOplus();
// 非常重要，是在当前顶点的值下，该误差对优化变量的偏导数，也就是Jacobian矩阵
```

除了上面四个函数，还有几个重要的成员变量以及函数：

```cpp
_measurement； // 存储观测值
_error;  // 存储computeError() 函数计算的误差
_vertices[]; // 存储顶点信息，比如二元边，_vertices[]大小为2
//存储顺序和调用setVertex(int, vertex) 和设定的int有关（0或1）

setId(int);  // 定义边的编号（决定了在H矩阵中的位置）
setMeasurement(type);  // 定义观测值
setVertex(int, vertex);  // 定义顶点
setInformation();  // 定义协方差矩阵的逆
```

有了上面那些重要的成员变量和成员函数，就可以用来定义一条边了：

```cpp
class myEdge: public g2o::BaseBinaryEdge<errorDim, errorType, Vertex1Type, Vertex2Type>
  {
      public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW      

      myEdge(){}     
      virtual bool read(istream& in) {}
      virtual bool write(ostream& out) const {}      
      virtual void computeError() override
      {
          // ...
          _error = _measurement - Something;
      }    
  
      virtual void linearizeOplus() override  // 求误差对优化变量的偏导数，雅克比矩阵
      {
          _jacobianOplusXi(pos, pos) = something;
          // ...         
          /*
          _jocobianOplusXj(pos, pos) = something;
          ...
          */
      }      
      private:
      data
  }
```

让我们继续看curveftting这个实例，这里定义的边是简单的一元边：

```cpp
// （误差）边的模型    模板参数：观测值维度，类型，连接顶点类型
class CurveFittingEdge: public g2o::BaseUnaryEdge<1,double,CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge( double x ): BaseUnaryEdge(), _x(x) {}
    // 计算曲线模型误差
    void computeError()
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0,0) = _measurement - std::exp( abc(0,0)*_x*_x + abc(1,0)*_x + abc(2,0) ) ;
    }
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
public:
    double _x;  // x 值， y 值为 _measurement
};
```

上面的例子都比较简单，下面这个是3D-2D点的PnP 问题，也就是**最小化重投影误差问题**，这个问题非常常见，使用最常见的二元边，弄懂了这个基本跟边相关的代码就能懂了：

```cpp
//继承自BaseBinaryEdge类，观测值2维，类型Vector2D,顶点分别是三维点、李群位姿
class G2O_TYPES_SBA_API EdgeProjectXYZ2UV : public  
               BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //1. 默认初始化
    EdgeProjectXYZ2UV();

    //2. 计算误差
    void computeError()  {
      //李群相机位姿v1
      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
      // 顶点v2
      const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
      //相机参数
      const CameraParameters * cam
        = static_cast<const CameraParameters *>(parameter(0));
     //误差计算，测量值减去估计值，也就是重投影误差obs-cam
     //估计值计算方法是T*p,得到相机坐标系下坐标，然后在利用camera2pixel()函数得到像素坐标。
      Vector2D obs(_measurement);
      _error = obs - cam->cam_map(v1->estimate().map(v2->estimate()));
    }

    //3. 线性增量函数，也就是雅克比矩阵J的计算方法
    virtual void linearizeOplus();

    //4. 相机参数
    CameraParameters * _cam; 
    bool read(std::istream& is);
    bool write(std::ostream& os) const;
};
```

这个程序中比较难以理解的地方是：

```cpp
_error = obs - cam->cam_map(v1->estimate().map(v2->estimate()));//误差=观测-投影
```

- cam_map 函数功能是把相机坐标系下三维点（输入）用内参转换为图像坐标（输出）。
- map函数是把世界坐标系下三维点变换到相机坐标系。
- v1->estimate().map(v2->estimate())意思是用V1估计的pose把V2代表的三维点，变换到相机坐标系下。

**\>>>向图中添加边**

和添加点有一点类似，下面是添加一元边：

```cpp
// 往图中增加边
    for ( int i=0; i<N; i++ )
    {
        CurveFittingEdge* edge = new CurveFittingEdge( x_data[i] );
        edge->setId(i);
        edge->setVertex( 0, v );                // 设置连接的顶点
        edge->setMeasurement( y_data[i] );      // 观测数值
        edge->setInformation( Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma) ); // 信息矩阵：协方差矩阵之逆
        optimizer.addEdge( edge );
    }
```

**但在SLAM中我们经常要使用的二元边**（前后两个位姿），那么此时：

```cpp
index = 1;
for ( const Point2f p:points_2d )
{
    g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
    edge->setId ( index );  // 边的b编号
    edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
    edge->setVertex ( 1, pose );
    edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );  // 设置观测的特征点图像坐标
    edge->setParameterId ( 0,0 );
    edge->setInformation ( Eigen::Matrix2d::Identity() );
    optimizer.addEdge ( edge );
    index++;
}
```

—————————————————

至此，就介绍完了g2o中的一些框架和实现，需要提醒的是在SLAM中我们常常用到的是二元边以及对应的点，他们都较为复杂，应当多次学习复习实践。



## Date5.23 - Date6.7周总结 <span id="二"> </span>

### 学习内容：

《视觉SLAM十四讲：从理论到实践》

第七章 视觉里程计一：

 PnP部分+ICP部分

（重点对g2o库进行了学习）

第八章 视觉里程计二：

 LK光流法原理 + 单层LK光流法

![696b99575d2a40d2ddf24c55c22fd14](C:\Users\Li\Documents\WeChat Files\wxid_3j6yc6bus6a732\FileStorage\Temp\696b99575d2a40d2ddf24c55c22fd14.png)

#### 一.（易混淆）2D-3D的PnP求解位姿方法与3D-3D的ICP求解位姿方法的区别：

​	**①2D-3D-PnP方法：**

​		3D指已知一组世界坐标系下的3D坐标，2D指该三维点在当相机处于某个位姿时在相机像素平面（相机坐标系）的一组2D坐标。于是我们的目标就是通过**不断调整由世界坐标系转换到相机坐标系的矩阵R与位移t**，**使得这组三维点能够在相机坐标系的二维像素平面中被更加精准表示**。

​		于是我们通过这个过程得到了**由世界坐标系变换到相机坐标系的旋转矩阵R与平移距离t**。

​		PnP中的n代表已知多少组这样的点，常用三组这样的点即可以解出位姿变换的矩阵R与位移t，故称为P3P。

​		故从上述过程中可知，**本质上PnP求解法即为找到误差的最小值**，故可采用高斯牛顿法等非线性优化方法。

​	**②3D-3D-ICP方法：**

​		此时的3D-3D为既知道一组世界坐标系下的点的3D坐标，也知道相机坐标系下该点的3D坐标。于是我们的目的与2D-3D一样，既**调整旋转矩阵R与平移距离t**使得**这个点从世界坐标系变换到相机坐标系下的后的坐标**与原本就在相机坐标系下的点的**位置相差距离最小**。（也称为点云配准）

​		同理我们也得到了R与t。

​		**实际上，ICP和PnP都有多种解法**，但是**我比较倾向于用非线性优化**解决误差最小问题。

#### **二. g2o非线性优化函数库的学习**

​		故在视觉SLAM中我们经常调用g2o库来对非线性优化过程进行计算。

​		g2o的使用大概包括以下过程：

​		**①在调用g2o解决非线性优化问题的函数之前定义所创建的顶点与边的性质：**

​			1.**顶点（Vertex）：用于存放待优化变量（比如上述所讲到的R、t（统称位姿pose））**

​			比如（下面为自定义定点类型的创建方法（如果使用g2o自带的顶点类型，直接声明即可使用））：

```cpp
class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // 字节对齐

    virtual void setToOriginImpl() // 重置，设定被优化变量的原始值
    {
        _estimate << 0,0,0;
    }

    virtual void oplusImpl( const double* update ) // 更新
    {
        _estimate += Eigen::Vector3d(update);   //update强制类型转换为Vector3d
    }
    // 存盘和读盘：留空
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
};
```

​	其中四个virtual为模板，我们创建时仅需更改：

​		函数头:	

​	BaseVertex<优化变量维度，优化数据变量类型>

​		SetToOriginImpl： 	

​	_estimate = 初始化估计值（如=Sophus：：SE3d -- 代表位姿类型（矩阵中同时包含了旋转矩阵R与平移距离t））

​		oplusImpl：	

​	_estimate += 更新值（你想对估计值如何进行更新（如=Sophus::SE3d::exp(update_eigen) * _estimate））

​	

​		**2.边（Edge）：指向一个顶点（优化变量），边的本质是误差项。几元边即连接几个顶点（优化变量），即对几个优化变量求导数or梯度。**

​		自定义边方法实例如下：

```cpp
//继承自BaseBinaryEdge类，观测值2维，类型Vector2D,顶点分别是三维点、李群位姿
class G2O_TYPES_SBA_API EdgeProjectXYZ2UV : public  
               BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //1. 默认初始化
    EdgeProjectXYZ2UV();

    //2. 计算误差
    void computeError()  {
      //李群相机位姿v1
      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
      // 顶点v2
      const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
      //相机参数
      const CameraParameters * cam
        = static_cast<const CameraParameters *>(parameter(0));
     //误差计算，测量值减去估计值，也就是重投影误差obs-cam
     //估计值计算方法是T*p,得到相机坐标系下坐标，然后在利用camera2pixel()函数得到像素坐标。
      Vector2D obs(_measurement); 
      _error = obs - cam->cam_map(v1->estimate().map(v2->estimate()));
    }

    //3. 线性增量函数，也就是雅克比矩阵J的计算方法
    virtual void linearizeOplus();

    //4. 声明相机参数
    CameraParameters * _cam; 
    //5.读取与存盘
    bool read(std::istream& is);
    bool write(std::ostream& os) const;
};
```

​	一般来说，**边用于计算误差项(_error)，与雅可比矩阵，并将结果回传至g2o优化函数中，可以根据误差大小来判断是否已经达到极值点**，即是否可以停止优化。

​	其中(_estimate为已知量（创建边时会传入，即数据的真实值）)

​	本质上还是bool computeError() + virtual void linearizeOplus() + bool read(wtd::istream& is) + bool write(std::ostream& os) const 四个模板填参数



#### 三.光流法学习

​	光流法是第二种视觉里程计，原理相较于第一种视觉里程计简单很多。

​	**LK光流法常用于跟踪角点的运动。**

​	**LK光流法**的使用**基于两个假设**：

​	**①灰度不变假设：同一个空间点的像素灰度值，在各个图像中是固定不变的。**

​		即有：

​		![image-20250608235849917](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250608235849917.png)

​		等式左边泰特展开：![image-20250608235924431](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250608235924431.png)

​		上式联立有：

![image-20250609000018914](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250609000018914.png)

​		两边同除以dt则有：

​	![image-20250609000053265](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250609000053265.png)

​	**②某一个窗口内的像素具有相同的运动。**

​		则将dx/dt计为u，dy/dt计为v。由于在同一个窗口内的像素具有相同的运动，则每个像素点的u、v都是相同的，即是一个关于像素点位置的二元方程，写成矩阵形式有（考虑其是有w*w个像素的窗口）：

![image-20250609000345218](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250609000345218.png)

​		则：

![image-20250609000519154](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250609000519154.png)

​	即最后能将**像素的水平移动速度u与数值移动速度v解出**，印证其能**跟踪角点运动**的性质。



#### 四.LK光流法代码实现：

​	**①OpenCV实现：**

​		直接调用OpenCV中的calcOpticalFlowPyrLK函数，并传入第一张图像、第二张图像、第一张图特征点（角点）坐标信息、用储存第二张图中角点位置信息的容器、状态数组（标记第一张图的某个坐标信息是否合法）、误差error。

​	实现效果：

​	![image-20250609002308421](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250609002308421.png)

​		（自己拍摄的照片，没有数据集拍的那么标准，两次照片之间位姿变化过大会导致角点难以跟踪）

​	**②单层LK光流法实现：**

​		使用**高斯-牛顿法**（核心：**找到误差函数**）：

![2edb0662f7c56175cc0629c3c0cd5e9](C:\Users\Li\Documents\WeChat Files\wxid_3j6yc6bus6a732\FileStorage\Temp\2edb0662f7c56175cc0629c3c0cd5e9.png)

​	实现效果：

​	![image-20250609002629452](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250609002629452.png)

​	（由于位姿变化过大导致角点无法跟踪）

​	**③多层LK光流法实现：**

​	**解决位姿变化过大导致角点无法跟踪的问题。**（后续学习）

### 学习心得

​	①后续要备考期末，可能书籍阅读方面推进会变缓慢，但仍会保持前进。

​	②暂时侧重点放在SLAM相关文献的阅读上，综述+最新技术成果，关注前沿研究方向。



## Date7.5 重温复习视觉里程计一与视觉里程计二LK单层光流法

![3bed249e65d2f856b9f73ad24e99093](D:\Desktop\SLAM\Pict\3bed249e65d2f856b9f73ad24e99093.png)

**对极几何：**

![image-20250705103737381](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705103737381.png)

![image-20250705103753831](D:\Desktop\SLAM\Pict\image-20250705103753831.png)

​				（O1，O2，P三点所成平面为极平面、e1与e2为极点、l1与l2为极线）

​	O1与O2为相机中心点，为已知点。p1为特征点匹配中第一帧时确定的特征点，p2为特征点匹配中第二帧时确定的与p1相似的特征点，故至此为止空间点P可以求得（O1p1与O2p2连线交点）。

​	进而有：

![image-20250705104545997](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705104545997.png)

​	可见，根据所确定的P点位置，可以还原出两个不同帧之间的位姿变换关系。

​	**注意：什么是齐次坐标？什么是尺度下相等？**

​	**①齐次坐标：**

![image-20250705154945126](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705154945126.png)

![image-20250705154217716](D:\Desktop\SLAM\Pict\image-20250705154217716.png)

​	**②尺度意义下相等**

![image-20250705155126810](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705155126810.png)

​	**不同的齐次坐标之间只相差一个非0的比例因子，即两个齐次坐标之间除了深度不同以外其余坐标实际上都是一致的**

![image-20250705155342022](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705155342022.png)

​								***下面对相机模型部分进行复习***

**针孔相机模型：**

![image-20250705110131649](D:\Desktop\SLAM\Pict\image-20250705110131649.png)

![image-20250705110347380](D:\Desktop\SLAM\Pict\image-20250705110347380.png)

![image-20250705110408765](D:\Desktop\SLAM\Pict\image-20250705110408765.png)

![image-20250705110829111](D:\Desktop\SLAM\Pict\image-20250705110829111.png)

​	注意：此处的O-xyz坐标系相当于为以相机镜头为原点建立的坐标系，而O'-x'y'z'则相当于以CMOS为原点建立的坐标系。（X,Y,Z）与(X',Y',Z')分别代表空间中的点P与像素平面中的点P'的坐标（像素平面中P'不具有Z轴坐标）。(5.2)式满足三角形相似的推论。***此处以CMOS为原点建立的坐标系≠像素坐标系！！！***

​	对5.2进行变换进而可以得到**空间点在CMOS平面上的X'，Y'坐标**（所有单位都是米（m））：

​								![image-20250705111422028](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705111422028.png)![image-20250705112245381](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705112245381.png)

​	**像素坐标系：**

![image-20250705112100056](D:\Desktop\SLAM\Pict\image-20250705112100056.png)

​	总而言之，像素坐标系与CMOS坐标系虽然所在平面一样，但是原点位置不一样。像素坐标系以CMOS左上角为原点，而CMOS坐标系以相机光心（镜头）在CMOS平面的投影点作为原点。

​	进而我们可以得到像素坐标系与CMOS坐标系坐标之间的变换关系：

![image-20250705112235698](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705112235698.png)

​	5.4与5.3融合，将X'与Y'用含α与β的式子推导出点在像素平面坐标系与在相机光心坐标系的直接关系：

![image-20250705112658245](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705112658245.png)

​	则可以写成矩阵形式，过程如下：

![81fec5b4132cddcd0dbb3c4bc08beec](D:\Desktop\SLAM\Pict\81fec5b4132cddcd0dbb3c4bc08beec.jpg)

即：

![image-20250705114212800](D:\Desktop\SLAM\Pict\image-20250705114212800.png)

**相机的内参**

![image-20250705114730603](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705114730603.png)

​	注意：**内参K**是不会变化的，由为相机自身的固定性质。

**相机的外参**

![image-20250705151953954](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705151953954.png)

![image-20250705152030920](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705152030920.png)

​	总而言之，相机的外参数用于将把空间点在世界坐标系中的坐标变换到相机坐标系下。因此，外参数会随着相机的运动而发生变化。（注意：**世界坐标系与相机坐标系是相对而言的**，比如研究2d-2d时，可以将相机第一帧所在的坐标系计为世界坐标系，而后续的帧均为不同的相机坐标系，故只用找到第一帧与后面帧相机的位姿变换关系，就找到了相机的外参数R、t）

**归一化平面**

![image-20250705152307975](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705152307975.png)

​	归一化平面本质上与像素平面一致，只不过归一化平面上点的坐标为（u，v）且Z轴固定为1，故单目视觉中会使点的深度值丢失。

**畸变模型：**

![image-20250705152700125](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705152700125.png)

![image-20250705152623388](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705152623388.png)

​	直接将点在归一化平面上的坐标找到，再利用公式计算出去畸变后归一平面上的点。

​									***下面回到视觉里程计一***

![image-20250705155435835](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705155435835.png)

![image-20250705155750208](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705155750208.png)

​	注意：p1~KP与p2~K(RP+t)可用 尺度意义下相等 解释，即将p1、p2写成齐次坐标（x,y,1）与KP（P本身就是（X，Y，Z））二者相似。其中x1与x2就是归一化平面上的点，只有x、y两个轴上的坐标。

![image-20250705160149319](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705160149319.png)

![image-20250705160201467](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705160201467.png)

![image-20250705160212657](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705160212657.png)

![image-20250705160223439](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705160223439.png)

​	此处就得到了2d-2d位姿变换的核心式（7.8、7.10）。

​	故其实解决2d-2d的位姿变换问题核心是通过两帧图片所已知的两个2d点来得到本质矩阵E，进而解出位姿变换R、t。

​	**解法为SVD分解，略！！！**

​	**编程中直接套用现有算法即可。**

​	 [pose_estimation_2d2d.cpp](SLAM\slambook2-master\slambook2-master\ch7\pose_estimation_2d2d.cpp) 

​	![image-20250705163424537](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705163424537.png)

![image-20250705164645520](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705164645520.png)

​	**解法：**

![image-20250705164701044](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705164701044.png)

​	代码中直接调用OpenCV函数即可直接使用三角测量：

![image-20250705165216009](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705165216009.png)

​	传参传入第一、二帧图片中的FAST关键点与第一、二帧图片之间的匹配点，并用points收集第一帧图片的深度信息，进而：

![image-20250705165453420](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705165453420.png)

​	深度信息s2与s1之间也满足位姿变换关系s2=Rs1+t，因此将s1进行位姿变换后收集其（2,0）（第三行第一列）信息（即z轴信息）即可。

![image-20250705165632982](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705165632982.png)

![image-20250705165743232](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705165743232.png)

![image-20250705165828945](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705165828945.png)

​	如之前所提到的，**坐标系之间的变换是相对的**，2d-2d以第一帧时坐标系作为世界坐标系，而3d-2d由于本身就存在一个3d点，故直接以该3d点所在坐标系为世界坐标系，目的即是将另一帧图片中的2d点所在的相机坐标系相较于第一帧图片世界坐标系间的位姿变换确定下来。

![image-20250705172642135](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705172642135.png)

​	通过解出2d点在相机坐标系下的3d坐标，我们就能够利用ICP的方法解出世界坐标系与相机坐标系之间的位姿变换关系。

​	**解出2d点在相机坐标系下3d坐标的方法：**

​	**EPnP or 非线性优化Bundle Adjustment光束法平差(BA)**

​	代码实现：

​	**OpenCV的EPnP求解方法：**

![image-20250705174801904](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705174801904.png)

![image-20250705174843424](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705174843424.png)

​	**非线性优化**

​	 [pose_estimation_3d2d.cpp](SLAM\slambook2-master\slambook2-master\ch7\pose_estimation_3d2d.cpp) 

![image-20250705175148779](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705175148779.png)

![image-20250705175859571](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705175859571.png)

**非线性优化**

![image-20250705182122420](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705182122420.png)

![image-20250705182141615](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705182141615.png)

![image-20250705182404215](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705182404215.png)



**LK单层光流法：**

![image-20250705212648037](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705212648037.png)

基本假设：

![image-20250705212707984](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705212707984.png)

![image-20250705212727783](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705212727783.png)

则有：

![image-20250705212809465](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705212809465.png)

即经典的线性代数中超定方程，使用最**小二乘法解（略）or 非线性优化or OpenCV函数库**

**OpenCV**：
![image-20250705213202610](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705213202610.png)

​	调用函数，传入前后两帧图像（img1、img2）、第一帧图像的角点（pt1），则可输出追踪后的点（pt2），以及各点的状态（status）、误差（error）。

**非线性优化：高斯牛顿法**

![image-20250705213548781](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705213548781.png)

 [optical_flow.cpp](SLAM\slambook2-master\slambook2-master\ch8\optical_flow.cpp) 

![image-20250705214625972](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705214625972.png)

![image-20250705214651846](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250705214651846.png)

​	核心步骤仍为高斯牛顿法，依然是找到误差函数，在最优化误差函数的过程中得到（u，v）（目标追踪点）的最优解。

## Date 7.7 学习记录 <span id="三"> </span>

### 1.C++中的参数注释方法（规范）

```c++
/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
```

### 2.C++中的内联函数 ‘inline’

​	当在定义函数前额外加上‘inline’前缀，则改函数将变为**内联函数**:

​	当程序运行时，内联函数的调用过程不会与常规函数一样进行压栈、入栈、出栈等操作，而是直接将内联函数嵌入进内存中，直接在固定地址对内联函数进行调用，大大节省了效率。

​	（在C++中，**内联函数**是一种用于提高函数执行效率的特性。通过使用*inline*关键字，开发者可以建议编译器在调用点替换函数体，从而减少函数调用的开销。内联函数通常用于执行简短的代码片段，以避免频繁调用小函数时产生的栈空间消耗。）

```C++
// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}
```

### 3.双线性插值函数 bilinear interpolation

​	双线性插值函数输出的浮点数结果在图像处理中具有重要作用，主要用于解决**非整数坐标下的像素值计算问题**

```C++
inline float GetPixelValue(const Mat &img,float x,float y)
{
    if(x < 0) x= 0;
    if(y < 0) y = 0;
    if(x >= img.cols) x = img.cols-1;
    if(y >= img.rows) y = img.rows-1;
    uchar *data = &img.data[int(y) * img.step + int(x)];  
    //img.step 指一行所占的字节数 ； int()用于取整 --> 故*data用于储存（int(x),int(y)）位置的像素
    
    float xx = x - floor(x); //计算x坐标的小数部分
    float yy = y - floor(y); //计算y坐标的小数部分
    //floor()为向负方向取整（不大于输入参数的最大整数）
    
    return float(
        (1-xx) * (1-yy) * data[0] +
        xx * (1 - yy) * data[1] + 
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}
```

**Keypoints！！！**

​	**①uchar *data = &img.data[int(y) * img.step + int(x)];**  

​	指的是用指针data储存img.data的第int(y) * img.step + int(x)个元素的地址，因此data[0] = img.data[int(y) * img.step + int(x)]，以及data[1] = img.data[.... + 1]、data[img.step] = img.data[... + img.step]（下一行的该列）。

​	**②(1-xx) * (1-yy) * data[0] +**

​            **xx * (1 - yy) * data[1] +** 

​            **(1 - xx) * yy * data[img.step] +**

​            **xx * yy * data[img.step + 1]**

​	此处为双线性插值法的核心过程，通过**加权非整数像素点的相邻四个整数点**（左上、左下、右上、右下）来得到该**非整数坐标像素点的像素**。

```text
(3,4)权重0.24   (4,4)权重0.06
       +-----------+
       | · · · · · |
       | · · · · · |  y=4.7
       | · · · · · |
       +-----*-----+  <- 目标点(3.2,4.7)
       | · · · · · |     离(3,5)最近 → 权重最大(0.56)
       | · · · · · |
       +-----------+
     (3,5)权重0.56   (4,5)权重0.14
```

```cpp
return 
(1 - xx) * (1 - yy) * data[0] +  // 左上权重 × 左上像素
xx * (1 - yy) * data[1] +        // 右上权重 × 右上像素
(1 - xx) * yy * data[img.step] + // 左下权重 × 左下像素
xx * yy * data[img.step + 1];    // 右下权重 × 右下像素
```

​	其中**(1-xx)为像素点距离右边界的距离（水平距离权重），(1-yy)为像素点距离下边界的权重（垂直距离权重）**，xx与yy同理（分别也为水平距离权重与垂直距离权重）。

​	（由水平与垂直两方向权重确定像素值-->双线性插值法）

### 4.boost::format + for循环 轮流读取多个文件

- **`"./%06d.png"`**：格式化字符串模板，其中：

  - `%06d`：表示用 6 位数字填充，不足位补零（如 `5` → `000005`）。
  - `./`：文件路径前缀（当前目录）。

- **功能**：后续通过 `fmt_others % 数值` 生成具体的文件名。

  ————

  for循环分步解析

  1. **`fmt_others % i`**：
     - 将整数 `i` 插入到格式化字符串的 `%06d` 位置。
     - 例如 `i = 1` 时，生成 `./000001.png`；`i = 42` 时，生成 `./000042.png`。
  2. **`.str()`**：
     - 将 `boost::format` 对象转换为 `std::string`，得到最终文件名。
  3. **`cv::imread`**：
     - 根据生成的文件名读取图像（`0` 表示以灰度模式读取）。

```C++
boost::format fmt_others("./%06d.png"); // 定义格式化模板
for(int i = 1;i<6;i++)
{
    Mat img = imread((fmt_others % i).str(),0);
}
```

## Date 7.14 ~ 7.19 周总结 <span id="四"> </span>

### 学习内容：

#### 1. 中型ROS2工程代码实践

​	在回顾ROS2基本知识的基础上，对基于ROS2构建的机器人自瞄算法工程进行了修改，利用所学ROS2知识结合EFK（卡尔曼扩展滤波）实现机器人自瞄的可视化（将预瞄点在视觉传感器所摄图片中标出并在节点中发布）。

​	详细思路与过程：

​	①找到云台控制信息发出的yaw-pitch角，找到yaw-pitch角所指向的目标点在云台坐标系下的坐标。

​	②考虑在装甲板信息中额外记录当前装甲板由相机坐标系到云台坐标系之间的变换矩阵。

​	③将云台所指向的目标点通过变换矩阵转换为相机坐标系下的目标点，即找到相机坐标系下的目标点。

​	④将相机坐标系下的目标点利用内参矩阵K+畸变系数p投影到像素平面上，得到目标点在像素平面上的投影点。

​	⑤最后利用OpenCV画图将像素坐标系下的目标点在result图像中标注出来，并在图像发布器上发布。

​	实现效果：

​	节点关系图：

![6e202aae0a592381b14fb33e58131c9](C:\Users\Li\Documents\WeChat Files\wxid_3j6yc6bus6a732\FileStorage\Temp\6e202aae0a592381b14fb33e58131c9.jpg)

​	视频展示：

<video src="C:\Users\Li\Documents\WeChat Files\wxid_3j6yc6bus6a732\FileStorage\Video\2025-07\3e3c101c744ffb57b2a754c07e6b9e98.mp4"></video>

​	本次实践中，我积累了处理中型CPP工程项目的经验，比如如果要像原有的代码中加入新功能，在什么地方需要额外注意（如.cpp文件中新增的变量，需要在.hpp中提前声明，否则会产生编译错误、如.cpp中多进程可能产生冲突，需要添加进程锁防止进程崩溃、如如何跨.cpp文件得到不同.cpp文件中定义的变量值（在目标.cpp文件中创建该命名空间的函数，函数的返回值即为想要得到的值，则最后只需要在实例下调用该函数即可得到目标cpp文件中的变量值）。

![image-20250719211125497](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250719211125497.png)

![image-20250719211155398](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250719211155398.png)

​	此外，本次实践也让我对OpenCV有了更多的认识，比如利用OpenCV灰度处理，二值化处理后得到仅有发光灯条的图片，便于进行灯条检测等。此外，如OpenCV中的projectPoints()函数，在视觉SLAM中曾有遇到，用与将相机坐标系下的点投影至像素平面坐标系，此次重新温习，增进了我对该函数的认识(比如该函数内部已经包含三维点的归一化过程，且投影效果与传入的内参矩阵、畸变系数准确度息息相关。因此为了得到更好的点投影效果，此次实践中我也对相机标定进行了学习，并掌握了使用Matlab处理标定数据，提炼准确的相机内参)。

![74afe6ce0bb9a846b0740a0d8301310](D:\Desktop\RM算法组\暑假实训\作业\Picture\74afe6ce0bb9a846b0740a0d8301310.png)

![ab6092c00c590a254a44a2529d5c7d8](D:\Desktop\RM算法组\暑假实训\作业\Picture\ab6092c00c590a254a44a2529d5c7d8.png)

#### 2.《视觉SLAM十四讲》 第八讲 视觉里程计二（光流、直接、多层光流\直接法）、第九讲 后端(一)（滤波器（KF、EKF）、非线性优化方法（BA图优化（g2o、ceres）））、第十讲 后端(二) （滑动窗口、位姿图） 

##### ①视觉里程计二--直接法：

###### 	视觉里程计一、二中算法的区别：

​	视觉里程计一 中的算法需要**找到图像中的特征点（角点）**，并**计算特征点的描述子**，在两张图像中进行**特征点匹配**。并使用匹配成功的点对作为基础，通过**最小化重投影误差（对极几何、PnP、ICP算法）**优化相机运动，估计相机位姿；

​	而视觉里程计二 中的算法（光流法、直接法），则甚至**无需提取特征点**，只需在图像上随机选点即可，更是**省略了描述子计算、特征点匹配**的过程，大大节省了计算时间，且在特征点稀疏的场景更有优势。其中，视觉里程计二中的**光流法（稀疏）**通过提取随机n个像素点的灰度值（亮度值），并通过**最小化光度误差**得到**点与点之间的关系**（即x、y方向轴上的速度u、v与前后两张图片中点的位移，**也是一种运动估计**）。此外，视觉里程计二中的**直接法**，则可以通过**最小化光度误差**直接对**相机位姿进行优化**。

###### 	视觉里程计二中光流法与直接法之间的区别：

​	光流法与直接法在运动估计的原理部分都是一样的，均采用最小化光度误差来优化相机运动，估计最佳相机运动。然而，二者所估计的相机运动有所不同，**光流法**为估计**相机的平面运动**（x、y方向上的运动速度与位移）；而**直接法**则为估计**相机的空间位姿运动**。

###### 	直接法计算过程：

​	![image-20250721110146452](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721110146452.png)

​	如图所示，我们可以写出投影方程：

![image-20250721110220074](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721110220074.png)

​	p1与p2为两图像中像素点的坐标，然而，直接法的原理是**最小化光度误差进行相机位姿估计**，估我们通过p1、p2得到两像素平面上的光度值，作差即为误差e：	

![image-20250721110701135](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721110701135.png)

​	由于我们要根据优化误差来得到最佳的位姿变换T（即关心误差e是如何随着相机位姿T变化的），估推导由像素位置p到位姿变换T的关系式：

![image-20250721111138887](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721111138887.png)

​	此时即建立起了位姿变换T与第二个像素平面坐标u之间的函数关系：

![image-20250721111459626](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721111459626.png)

​	则对误差e求导，即相当于对I(p2)求偏导，即为：

![image-20250721111527395](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721111527395.png)

​	各个偏导详细求解过程：

![image-20250721111819491](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721111819491.png)

###### 	单层直接法代码实现流程：

​	**1. 初始化阶段**

- **输入数据**：
  - 参考图像 `img1` 和当前图像 `img2`（灰度图）。
  - 参考图像中的像素点集合 `px_ref` 及其对应的深度值 `depth_ref`。
  - 初始相机位姿 `T21`（从参考图像到当前图像的变换矩阵）。
- **参数设置**：
  - 迭代次数 `iterations = 10`。
  - 创建 `JacobianAccumulator` 对象 `jaco_accu`，用于计算雅可比矩阵、海森矩阵和误差。

------

**2. 迭代优化位姿**

**每次迭代的核心步骤**：

1. **重置累加器**：
   - 调用 `jaco_accu.reset()` 清空海森矩阵 `H`、偏置项 `b` 和误差 `cost`。
2. **并行计算雅可比矩阵**：
   - 使用 `cv::parallel_for_` 并行处理所有像素点，调用 `accumulate_jacobian` 计算每个点的贡献：
     - **投影3D点**：将参考图像的像素点根据深度和相机内参转换为3D点，再通过当前位姿 `T21` 投影到当前图像。
     - **光度误差计算**：比较参考图像和当前图像中对应像素块的灰度值差异。
     - **雅可比矩阵计算**：
       - 图像梯度（`J_img_pixel`）：通过中心差分计算当前图像的灰度梯度。
       - 像素对位姿的导数（`J_pixel_xi`）：根据投影几何推导的2×6矩阵。
       - 合成总雅可比矩阵 `J = -(J_img_pixel^T * J_pixel_xi)^T`。
     - **更新海森矩阵和偏置**：
       - `H += J * J^T`（高斯牛顿法）。
       - `b += -error * J`。
3. **求解位姿更新量**：
   - 解线性方程 `H * δξ = b`，得到位姿增量 `update`。
   - 通过指数映射 `Sophus::SE3d::exp(update)` 更新当前位姿 `T21`（左乘扰动）。
4. **收敛判断**：
   - 如果误差 `cost` 上升或更新量 `update.norm()` 小于阈值 `1e-3`，提前终止迭代。

------

**3. 输出与可视化**

- **输出结果**：
  - 打印优化后的位姿 `T21` 和总耗时。
- **可视化匹配点**：
  - 在当前图像上绘制参考图像像素点（绿色圆圈）及其投影位置（绿色连线）。

------

**关键函数说明**

**`JacobianAccumulator::accumulate_jacobian`**

1. **投影3D点**：
   - 将像素坐标 `(x,y)` 转换为归一化坐标，乘以深度得到3D点 `point_ref`。
   - 通过 `T21` 变换到当前相机坐标系 `point_cur`，再投影到当前像素坐标 `(u,v)`。
2. **光度误差**：
   - 对每个像素周围的 `3x3` 邻域，计算参考图像和当前图像的灰度差值 `error`。
3. **雅可比矩阵**：
   - **图像梯度**：通过双线性插值计算当前图像的 `x` 和 `y` 方向梯度。
   - **投影几何导数**：推导像素坐标对李代数位姿的导数（2×6矩阵）。
   - 合成总雅可比矩阵 `J`。
4. **并行累加**：
   - 使用互斥锁保护全局变量 `H`、`b` 和 `cost` 的更新。

------

**4. 辅助函数**

- **`GetPixelValue`**：双线性插值获取图像像素值，处理边界情况。



###### 	多层直接法代码实现：

**1. 流程结构对比**

| **步骤**           | **单层直接法**                               | **多层直接法**                                               |
| :----------------- | :------------------------------------------- | :----------------------------------------------------------- |
| **图像预处理**     | 直接使用原始分辨率图像。                     | 构建图像金字塔（如4层，缩放比例0.5）。                       |
| **参数初始化**     | 固定相机内参 `(fx, fy, cx, cy)`。            | 每层缩放内参（如第`level`层内参为 `fx*scale^level`）。       |
| **位姿优化入口**   | 直接调用 `DirectPoseEstimationSingleLayer`。 | 从最粗层（顶层）开始，逐层调用 `DirectPoseEstimationSingleLayer`。 |
| **位姿传递方式**   | 无跨层传递，单次优化完成。                   | 上一层的优化结果 `T21` 作为下一层的初始值。                  |
| **像素点坐标处理** | 直接使用原始像素坐标 `px_ref`。              | 每层按比例缩放像素坐标 `px_ref * scale^level`。              |

**2. 关键操作差异**

**(1) 图像金字塔构建**

- **多层法**：

  ```cpp
  vector<cv::Mat> pyr1, pyr2;  // 图像金字塔
  for (int i = 0; i < pyramids; i++) {
      if (i == 0) {
          pyr1.push_back(img1); 
          pyr2.push_back(img2); // 原始图像
      } else {
          cv::resize(pyr1[i-1], img1_pyr, Size(rows*scale, cols*scale)); // 降采样 ***即在图像所想展示的内容不变前提下，将图片分辨率降低（如原始图像：640x480 → 第1层：320x240 → 第2层：160x120）***
          pyr1.push_back(img1_pyr);
      }
  }
  ```

- **单层法**：无需此步骤，直接使用 `img1` 和 `img2`。

**(2) 内参和像素坐标缩放**

- **多层法**：

  ```cpp
  for (int level = pyramids-1; level >= 0; level--) {
      fx = fxG * scales[level];  // 缩放内参
      VecVector2d px_ref_pyr = px_ref * scales[level]; // 缩放像素坐标
      DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
  }
  ```

- **单层法**：内参和像素坐标保持不变。

**(3) 位姿优化顺序**

- **多层法**：
  从最粗层（低分辨率）到最细层（高分辨率）**逐层优化**，位姿结果依次传递：

  ```cpp
  for (int level = pyramids-1; level >= 0; level--) {
      // 每次调用单层优化，T21会被更新并传递到下一层
      DirectPoseEstimationSingleLayer(..., T21); 
  }
  ```

- **单层法**：仅在原始分辨率进行一次优化。

**3. 核心代码差异示例**

**(1) 单层法调用**

```cpp
// 直接使用原始图像和参数
DirectPoseEstimationSingleLayer(img1, img2, px_ref, depth_ref, T21);
```

**(2) 多层法调用**

```cpp
// 构建金字塔后，从顶层到底层逐层优化
for (int level = pyramids-1; level >= 0; level--) {
    fx = fxG * scales[level];  // 动态调整内参
    VecVector2d px_ref_pyr = px_ref * scales[level]; // 缩放像素坐标
    DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
}
```

​	详情： [direct_method.cpp](SLAM\slambook2-master\slambook2-master\ch8\direct_method.cpp) 

##### ②后端(一)

###### 	什么是后端？

​	后端站在全局的角度，对机器人的运动进行最优估计。

![image-20250721152840468](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721152840468.png)

​	SLAM过程可以由运动方程与观测方程来描述，即：
![image-20250721153110178](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721153110178.png)

​	其中Xk为机器人在k时刻状态（位姿），Uk为k时刻的运动输入，Zk为k时刻的观测值，Yj为路标点，Wk、Vk分别为运动方程的噪声与观测方程的噪声。	![image-20250721153657582](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721153657582.png)

​	根据对k时刻状态Xk的相关性假设，后端的处理方法分为滤波器方法（k时刻只与k-1时刻状态相关）与非线性优化方法（k时刻与k时刻之前所有状态相关）两种方法。

![image-20250721154447352](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721154447352.png)

###### 	后端中的滤波器方法（线性系统与KF）卡尔曼滤波推导方法

​	由于滤波器方法假设了马尔科夫性（k时刻只与前k-1时刻状态相关），则可以将（9.6）式中的两式转写为：

![image-20250721160117797](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721160117797.png)

​	也就是说滤波器方法只需要维护一个时刻的状态量（位姿）即可，如果该状态量满足高斯分布，则只需要维护该状态量的均值与协方差即可，于是滤波器便能够推导下一时刻的机器人状态：

![image-20250721161929165](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721161929165.png)

​	由线性高斯系统将（9.5）式中似然与先验概率转写成高斯分布 -- **预测**：

![image-20250721161328993](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721161328993.png)

![image-20250721163459713](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721163459713.png)

​	则根据预测步得到的（9.11）与（9.13）两个方程，能够对后验进行计算（后验=似然*先验），即 -- **更新**：

![image-20250721163754433](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721163754433.png)

​	将高斯分布展开，对应项对齐，最终解出后验与先验的关系：

![image-20250721164200353](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721164200353.png)

###### 	非线性系统与扩展卡尔曼滤波（EKF）

![image-20250721164406089](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721164406089.png)

​	扩展卡尔曼滤波滤波利用泰勒展开将非线性变为线性，后续过程与KF推导过程一致，则最终有：

![image-20250721164507091](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721164507091.png)

###### 	后端中的非线性优化方法（BA与图优化）

​	BA（光束法平差）：

![image-20250721165944952](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721165944952.png)

​	投影过程：

![image-20250721170020106](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170020106.png)

![image-20250721170056572](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170056572.png)

​	BA求解：

![image-20250721170133461](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170133461.png)

​	即使用高斯牛顿法or列文伯格法求解该式：

![image-20250721170343509](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170343509.png)

​	利用稀疏性性质，采用边缘化路标点变量δXp（Schur消元）：

![image-20250721170542461](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170542461.png)

![image-20250721170635554](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170635554.png)

![image-20250721170712915](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721170712915.png)

​	避免误匹配产生的误差影响整体后端估计--鲁棒核函数：
![image-20250721172335424](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721172335424.png)

![image-20250721172348319](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721172348319.png)

###### 使用Ceres库计算BA（代码实现）

​	![image-20250721193322325](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721193322325.png)

​	核心代码部分：

![image-20250721174933472](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721174933472.png)

​	首先，调用BAL数据集，初始化数据集数据（标准化、设置噪声强度）。然后，调用SolveBA函数对数据进行求解运算，得到优化后的数据（点云数据（位姿））。最后，调用BAL数据集导出数据（名为“finnal.ply”）。

​	SolveBA()函数部分：

​	核心为定义代价函数，计算与求雅可比矩阵部分交给ceres库自动计算，最终直接输出结果即可。

![image-20250721175338703](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721175338703.png)

​	着重详细讲述代价函数定义部分

```cpp
  cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);
```

​	通过调用**SnavelyReprojectionError::Create()函数**创建自动微分代价函数：

![image-20250721175608479](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250721175608479.png)

1. `ceres::AutoDiffCostFunction` 解析

```cpp
ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>
```

这是Ceres Solver提供的自动微分模板类，其核心参数如下：

| 模板参数                   | 说明                                   |
| -------------------------- | -------------------------------------- |
| `SnavelyReprojectionError` | 用户自定义的仿函数类，包含误差计算逻辑 |
| `2`                        | 输出残差的维度（2D像素坐标误差）       |
| `9`                        | 第一个优化变量的维度（相机参数）       |
| `3`                        | 第二个优化变量的维度（3D点坐标）       |

关键特点：
- **自动微分**：自动计算雅可比矩阵，避免手动推导复杂导数
- **多参数支持**：可支持多个参数块（这里处理相机和点两个参数块）
- **类型泛化**：通过模板参数`T`同时支持双精度和Jet类型（用于自动微分）

2. `SnavelyReprojectionError` 类实现

```cpp
class SnavelyReprojectionError {
public:
    // 构造函数保存观测值
    SnavelyReprojectionError(double observation_x, double observation_y) 
        : observed_x(observation_x), observed_y(observation_y) {}

    // 核心误差计算函数
    template<typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        // 1. 将3D点投影到相机坐标系
        T p[3];
        AngleAxisRotatePoint(camera, point, p);  // 旋转
        p[0] += camera[3];  // 平移X
        p[1] += camera[4];  // 平移Y
        p[2] += camera[5];  // 平移Z

        // 2. 归一化平面投影
        T xp = -p[0]/p[2];
        T yp = -p[1]/p[2];

        // 3. 应用径向畸变
        const T& l1 = camera[7];  // 二阶畸变系数
        const T& l2 = camera[8];  // 四阶畸变系数
        T r2 = xp*xp + yp*yp;
        T distortion = T(1.0) + r2*(l1 + l2*r2);

        // 4. 焦距缩放
        const T& focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;

        // 5. 计算残差
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        return true;
    }

    // 工厂方法创建CostFunction
    static ceres::CostFunction* Create(double observed_x, double observed_y) {
        return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
            new SnavelyReprojectionError(observed_x, observed_y));
    }

private:
    const double observed_x;  // 观测到的x坐标
    const double observed_y;  // 观测到的y坐标
};
```

3. 参数映射说明

相机参数 `camera[9]` 的结构：

| 索引 | 参数        | 类型   | 说明                  |
| ---- | ----------- | ------ | --------------------- |
| 0-2  | rotation    | double | 旋转向量（角轴表示）  |
| 3-5  | translation | double | 平移向量 (tx, ty, tz) |
| 6    | focal       | double | 焦距 f                |
| 7    | k1          | double | 二阶径向畸变系数      |
| 8    | k2          | double | 四阶径向畸变系数      |

3D点参数 `point[3]` 的结构：

| 索引 | 参数 | 类型   | 说明    |
| ---- | ---- | ------ | ------- |
| 0    | X    | double | 点X坐标 |
| 1    | Y    | double | 点Y坐标 |
| 2    | Z    | double | 点Z坐标 |

4. 自动微分工作原理(在ceres内部调用，用于计算残差的雅可比矩阵，并求解Hδx=g方程)

1. **第一次调用**：使用实际类型 `double` 计算残差
2. **第二次调用**：使用Jet类型计算偏导数
   - 对相机参数求导：9个偏导
   - 对点坐标求导：3个偏导
3. Ceres自动组合这些偏导数形成雅可比矩阵

###### 使用g2o库计算BA（代码实现）

​	略！ -- 定义顶点 + 定义二元边 + 添加顶点、二元边 + 优化（见代码）

​	 [bundle_adjustment_g2o.cpp](SLAM\slambook2-master\slambook2-master\ch9\bundle_adjustment_g2o.cpp) 



## Date7.21 ~ 7.27 周总结 <span id = "五"> </span>

### 	学习内容:

#### 	1.《视觉》第11讲 -- 回环检测

![image-20250728105956735](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728105956735.png)

​	在后端中，回环检测是一十分重要的部分。

​	回环检测的基础 -- 词袋模型。词袋模型通过将图片中的特征点向下划分（类似树结构），归类得到含多个单词的字典。该字典即可以用于对任意一张新图片进行描述（将图片转换为词袋向量），从而能够实现图像之间的相似度检测，进而能够得到回环的一对照片。

​	当检测到回环，并得到回环的一对照片后，在这一对照片的序号之内进行回环的后端优化（比如序号1-10之内），实现对**地图路标点+相机位姿**的同时优化，最终在地图中得到更加准确的相机位姿与路标点位置。

> [!note]
>
> Jetson 嵌入式设备无法使用较大的字典对图像进行分类（Maybe CPU性能不足），只能使用较小的字典，估判断图像相似不那么好。

#### 	2.《视觉》第12讲 -- 建图

![image-20250728110014759](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728110014759.png)

​	稠密地图是相对于稀疏地图而言的，**稀疏地图只建模感兴趣的部分（即特征点）**，而**稠密地图建模所有看到的点（每一个像素点都建模）**。稠密地图建模过程中运算量大，但得到的**稠密地图相较于稀疏地图**，能够辅助机器人实现**导航、避障**或是环境的完整**三维重建**。

​	**①单目稠密重建**，由于使用单目相机进行稠密重建，而单目相机不能直接得到像素点深度，需要使用对极几何对像素点进行**三角化（极限搜索+块匹配）**，故虽然能够实现稠密重建，但是实现效果不佳（在特征稀疏平面中，块匹配会由于一个区域的像素都长得差不多而失效）。

（下图为：原始图像image 、 真实的深度图像depth_truth 、 估计的深度图像与误差depth_estimate、depth_error）

![image-20250728113609720](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728113609720.png)

​	实际上**此时已经计算出了图像中所有像素点的3d位置**，并且数据集中已经记录了各个图像相机的位姿，如果将每次计算的**像素点3d位置**结合**相机位姿**进行点云拼接，**就能得到稠密地图**。

​	**②RGB-D稠密建图**，RGB-D相机拍摄出的图片会附属携带其深度信息，估不再需要想单目or双目相机那样通过计算得到图像的深度信息，既节省了计算量，同时也避免了计算带来的误差。

​	RGB-D的稠密建图与单目相机稠密建图在最后一步完全一致（即得到了图像的深度信息后（即相当于得到所有像素的相机坐标系下3d点）），只需要**结合相机位姿信息将相机坐标系下各个像素的3d坐标转换为世界坐标系下的3d坐标**，并将**点云进行加和**，使用**滤波器（统计滤波器+体素滤波器）消除不需要的点**，最终即可**得到多幅图像拼接得到的稠密点云地图**。

![image-20250728120554665](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728120554665.png)

​	然而单纯的稠密点云地图并不能用于定位（点云并没有进行位姿上的优化)，也不能用于导航与避障（不能说单纯几个点云是否被占据，需要将点云划分为占据网格)。因此需要从点云重建网格（先计算点云中每个点的法线（拟合平面找法线），再根据法线计算网格)。

![image-20250728160659995](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728160659995.png)

​	此图即为**点云数据+占据网格后得到的稠密地图**（白色线是我加的法线可视化，也可隐藏）。

​	然而该占据网格地图也有弊端，一是.pcd文件所占空间大（其中有很多我们不需要的细节），二是无法处理运动物体，点云地图靠的是点云间的不断拼接而成，而没有删除点云的做法。为解决这两个问题，我们可以使用八叉树地图（octomap）。

![image-20250728161340953](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728161340953.png)	八叉树可以自定义展开的层数，展开的层数越多那么细节也就越多，相反展开层数越少细节则越少，且占据信息使用[0,1]中的的概率（概率对数值）表示（初始时占据信息为0.5，若后续不断检测到该占据信息那么概率则增加，相反概率减少)，既能够实现占据信息的实时更新，也能够节约大部分空间消耗。

![image-20250728161745752](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728161745752.png)

​						（⬆️⬆️⬆️概率对数值y与概率x的换算关系⬆️⬆️⬆️）

![image-20250728163202396](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728163202396.png)

​										（八叉树展开八层）

![image-20250728163430832](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728163430832.png)

​								（八叉树展开十五层并按照深度值上色）

#### 	3.《视觉》第13讲 -- MySLAM工程实践（代码理解+增加回环检测功能）

![image-20250728110026005](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728110026005.png)

​	该视觉SLAM工程使用**双目相机**作为视觉传感器，构建**稀疏地图**。

​	视觉SLAM工程主要由四个部分构成，**前端、后端、地图、回环检测**。

​	**前端**的工作是根据视觉传感器所采集的图像做初始化的位姿估计与特征点检测、匹配工作，根据特征点匹配的情况（如果前后两帧图像匹配成功的内点（排除有误差的外点后剩的点）较少）决定是否更新地图中的路标点、设置关键帧与是否进入后端进行位姿优化（对前端设置的关键帧位姿进行优化）。

​	**后端**的工作是根据前端传回的信息，对目前活跃的关键帧进行优化（目前活跃关键帧为只在全部关键帧中使用滑动窗口挑选最新的几帧关键帧，加快了后端优化速度），由于后端负责对地图的规模进行控制，如果不使用滑动窗口挑选最新关键帧，那么会导致后端优化十分缓慢，进而导致可视化不及时甚至卡死（不过理应后端不影响实时可视化？？猜测后端长期占用地图线程导致前端无法进入地图线程中对信息进行更新）。

​	**地图**的工作是标记视觉传感器所采集到的路标点，与显示关键图像帧（可选是否显示），主要用于给用户端的可视化展示，构建稀疏地图。

​	**回环检测**的工作在先前提过，当检测到回环，并得到回环的一对照片后，在这一对照片的序号之内进行回环的后端优化（比如序号1-10之内），实现对**地图路标点+相机位姿**的同时优化，最终在地图中得到更加准确的相机位姿与路标点位置。

​	值得注意的是，该SLAM工程在前端、后端、回环检测中都使用的是借助**BA原理使用g2o**图优化，然而，**前端的g2o仅为单元边**的优化（仅针对相机位姿），而**后端与回环检测都是双元边**的优化（相机位姿 + 路标点），但是**后端采用的是滑动窗口**，故只对最近的几帧关键帧进行优化，而**回环检测是对整个回环包含的关键帧进行优化**，范围更大。并且，工程**特征匹配**所用的方法都是**光流法**，而**特征点检测**使用的是**GFTT**方法。

![image-20250728181904618](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250728181904618.png)

​									（使用KITT数据集进行视觉SLAM建图的结果）

> [!note]
>
> ​	如果采用滑动窗口，MySLAM窗口中就不能显示全局的特征点，Why？
>
> ​	--- 滑动窗口只显示被观测的路标点，未被观测的路标点会被删除。
>
> ​	目前虽然能打开全局路标点显示（将全局路标点用新变量储存并可视化该新变量），但是由于回环检测模块有bug，导致全局位姿偏差较大（建出来的图十分丑陋）。

### 	未来规划：

#### 	1.下一周对目前视觉SLAM优秀开源代码进行学习、理解、复现

​	VSLAM要求：（稀疏or半稠密 -- 能够实时建图）

​	**ORB-SLAM** -- 特征点SLAM中的巅峰，十分经典的视觉SLAM，能够实现实时的视觉SLAM建图，并支持单目、双目、RGB-D多种模式。缺点是由于建立的是稀疏地图，估不能支持避障、导航功能。

​	深入SLAM研究需要一定的深度学习基础，后续考虑学习深度学习？（若有时间）

#### 	2.实践：使用视觉传感器实现实时建图！

​	使用ORB-SLAM尝试实现手持相机建图！

​	复现可参考该GitHub -- [LegendLeoChen/LeoDrone: ubuntu22.04 + ROS2 humble 环境下的无人机基本运动控制和视觉SLAM方案](https://github.com/LegendLeoChen/LeoDrone?tab=readme-ov-file)

#### 3.实践过程中问题记录 --（Successfully）

##### 	①使用ORB-SLAM数据集能跑，ROS2版本功能包能成功编译，但是ros2 run之后报错：

```apl
jetson@snake:~/ORB_SLAM_ROS2_WS$ ros2 run orbslam3 mono /home/jetson/ORB_SLAM_ROS2_WS/src/ORB_SLAM3_ROS2-humble/vocabulary/ORBvoc.txt /home/jetson/ORB_SLAM3-master/Examples/Monocular/TUM1.yaml

ORB-SLAM3 Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
ORB-SLAM2 Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
This program comes with ABSOLUTELY NO WARRANTY;
This is free software, and you are welcome to redistribute it
under certain conditions. See LICENSE.txt.

Input sensor was set to: Monocular
Loading settings from /home/jetson/ORB_SLAM3-master/Examples/Monocular/TUM1.yaml
	-Loaded camera 1
Camera.newHeight optional parameter does not exist...
Camera.newWidth optional parameter does not exist...
	-Loaded image info
	-Loaded ORB settings
Viewer.imageViewScale optional parameter does not exist...
	-Loaded viewer settings
System.LoadAtlasFromFile optional parameter does not exist...
System.SaveAtlasToFile optional parameter does not exist...
	-Loaded Atlas settings
System.thFarPoints optional parameter does not exist...
	-Loaded misc parameters
----------------------------------
SLAM settings: 
	-Camera 1 parameters (Pinhole): [ 517.306 516.469 318.643 255.314 ]
	-Camera 1 distortion parameters: [  0.262383 -0.953104 -0.005358 0.002628 1.16331 ]
	-Original image size: [ 640 , 480 ]
	-Current image size: [ 640 , 480 ]
	-Sequence FPS: 30
	-Features per image: 1000
	-ORB scale factor: 1.2
	-ORB number of scales: 8
	-Initial FAST threshold: 20
	-Min FAST threshold: 7


Loading ORB Vocabulary. This could take a while...
Vocabulary loaded!

Initialization of Atlas from scratch 
Creation of new map with id: 0
Creation of new map with last KF id: 0
Seq. Name: 
There are 1 cameras in the atlas
Camera 0 is pinhole
slam changed
============================ 
Starting the Viewer
[ros2run]: Segmentation fault
```

​	解决方案（不知是否可行）-- [Segmentation fault when running ros2 humble (mono) · Issue #20 · zang09/ORB_SLAM3_ROS2](https://github.com/zang09/ORB_SLAM3_ROS2/issues/20)

​	Jetson orin nano自带的OpenCV库为4.1.0，而ORB_SLAM_ROS2功能包中要求4.2.0，清除4.1.0的OpenCV下载4.2.0的？

​	-- 解决方案可行！！！

---

##### 	②连接上USB相机后使用yahboom自带的相机驱动无法启动相机！

​	解决方案

​	在camera_usb.py中将下面代码

```C++
self.cap = cv2.VideoCapture(0)
```

​	改为

```C++
self.cap = cv2.VideoCapture(1)
```

​	意为初始读取/dev/video0设备改为读取/dev/video1（嵌入式相机为video0，而USB相机为video1（USB相机接入后一般有video1与video2两个dev，但是只有video1有相机输出画面））

---

##### 	③如何使用无线传输将手机相机传入Ubuntu系统中（实现Ubuntu系统能够识别出/dev/videox的效果）

​	解决方案

​	首先，在https://www.dev47apps.com/中下载适用于arm设备的源码（Jetson orin nano为arm64架构，只能通过源码编译下载该虚拟投屏软件）。得到DroidCam源码后，阅读README.md文件安装Ubuntu所需要依赖，依赖安装完成后sudo ./install-client 与 sudo ./install-video，完成DroidCam软件的完整安装。

​	然后，手机中下载DroidCam OBS（随便下，我下的是破解版）。在Ubuntu终端输入DroidCam打开DroidCam，手机打开DroidCam OBS，将手机中的设备IP号输入到Ubuntu的DroidCam中，点击start即可完成虚拟图像传输！！！（Tips:实验室网不好，建议开热点，让两设备处于同一热点下）（此时Ubuntu系统已经能够成功识别设备，显示为 /dev/video3与/dev/video4（或2与3））

​	最后，在yahboom自带的ROS2相机驱动包中将camera_usb.py中下面代码

```C++
self.cap = cv2.VideoCapture(1)
```

​	改为

```C++
self.cap = cv2.VideoCapture(4)
```

​	colcon build ， source ，ros2 run三件套，即可将传输进的虚拟图像在ROS2中发布为名称为“/image_raw"的消息！！！（供ORB_SLAM订阅）

​	或者！！可以直接在手机上发布ROS2消息，但是目前只在网上找到ROS1消息的软件，且仅支持到安卓7.1系统。

---

##### 	**成功复现！！！**

​	**使用USB接口摄像头建图：**

![image-20250801165018029](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250801165018029.png)

​	**使用无线设备图像传输建图：**

![image-20250801165052438](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250801165052438.png)



## Date7.28 ~ 8.2 周总结 <span id="六"> </span>

### 学习内容：

#### 	ORB_SLAM3经典视觉SLAM算法功能复现与代码初步理解。

1. ##### ORB_SLAM3原生仅支持Ubuntu18.04 ROS1，而我们需要在Ubuntu22.04的arm架构上运行，并实现实时建图、无线通信，需要ORB_SLAM_ROS2工程辅助。

​	遇到的详细问题在上一周的未来规划中有所记录！

2. ##### ORB_SLAM3算法特性。

   目前仅对代码中主函数延伸的各个函数代码进行了逐行理解，并没有进行宏观代码理解。

   不过从目前感受来看，工程项目果然还是十分稳健。很多地方加了互斥锁，并且函数分块，一个函数能尽可能在更多的地方被调用。

   在特征点提取方面，ORB_SLAM3采用ORB特征点提取，通过计算描述子与词袋向量，将不同图像帧之间的特征点利用词袋单词相同进行匹配，大大提高了相较于暴力匹配的匹配效率。

   位姿估计方面，分为三层位姿估计 -- 初始化位姿估计（通过上一帧计算当前帧预测位姿） -- 局部地图位姿估计（利用局部地图的关键帧对当前图像帧相机位姿进行估计，使用g2o图优化，同时优化路标点） -- 全局地图位姿估计（估计出现在回环检测部分，但是目前还没有看到回环检测（可能是漏看了？））。

   在适用性方面，ORB_SLAM3支持单、双目、RGB-D以及其带IMU的传感器形式。且似乎带IMU时算法处理会比较宽松（比较相信IMU数据能够给位姿估计提供更优帮助），不过IMU与相机传感器数据融合在《视觉十四讲》中没有专门讲过，所以对于原理目前还十分朦胧，需要后续学习了解。

### 未来规划：

1. 8.4 - 8.15放假约10天，调整放松。
2. 放假回来之后整体理解ORB_SLAM3算法，撰写算法大致流程，与 前端 -- 后端 -- 地图 -- 回环检测 四个线程的工作流程。
3. 学习IMU数据与相机传感器数据融合原理，尝试使用带IMU的视觉传感器检验建图效果？
4. 完成ORB_SLAM3算法学习的所有工作后开启激光SLAM的学习！

## Date 8.15 ~ 8.30

### 学习内容：

#### 1.ORB_SALM3算法流程概览

![Untitled diagram _ Mermaid Chart-2025-08-16-114103](D:\Desktop\SLAM\Pict\Untitled diagram _ Mermaid Chart-2025-08-16-114103.png)

![deepseek_mermaid_20250816_d7c566](D:\Desktop\SLAM\Pict\deepseek_mermaid_20250816_d7c566.png)

##### ①**前端（Tracking 线程）**

![Untitled diagram _ Mermaid Chart-2025-08-16-093831](D:\Desktop\SLAM\Pict\Untitled diagram _ Mermaid Chart-2025-08-16-093831.png)

**目标**：实时处理每一帧图像，估计相机位姿并决定关键帧插入
**工作流程**：

1. **初始化**（单目特有）：

   - 通过 **对极几何** 或 **单应矩阵** 计算前两帧的相对位姿
   - 三角化生成初始地图点（如 `Initializer::Initialize()`）
   - IMU单目模式会同时初始化IMU参数（重力方向、偏置等）

2. **帧处理**：

   ```C++
   // ORB-SLAM3/src/Tracking.cc
   cv::Mat Tracking::GrabImageMonocular(...) {
       // 1. 特征提取
       ExtractORB(im, 0); 
       // 2. 位姿估计（三种模式）
       if (mState == OK) {
           if (mVelocity.empty()) 
               TrackReferenceKeyFrame();  // 参考关键帧跟踪
           else 
               TrackWithMotionModel();    // 运动模型跟踪
       }
       else 
           Relocalization();              // 重定位
       // 3. 局部地图跟踪
       TrackLocalMap();
       // 4. 关键帧决策
       if (NeedNewKeyFrame())
           CreateNewKeyFrame();
   }
   ```

   - **特征提取**：提取ORB特征点（`ExtractorORB`）

   - **位姿估计**：

     - **运动模型**：基于恒定速度假设估计位姿

     - **参考关键帧**：通过**词袋匹配（BoW）**估计位姿（避免暴力匹配，提高特征点之间匹配效率）
     - **重定位**：当丢失时，通过DBoW2检索候选关键帧

   - **局部地图跟踪**：将当前帧与局部地图点匹配，优化位姿

   - **关键帧决策**（`NeedNewKeyFrame()`）：

     - 时间间隔（>15帧）
     - 跟踪点数量下降（<参考关键帧的90%）
     - 局部地图点观测不足

##### ②**后端（LocalMapping 线程）**

![Untitled diagram _ Mermaid Chart-2025-08-16-093934](D:\Desktop\SLAM\Pict\Untitled diagram _ Mermaid Chart-2025-08-16-093934.png)

**目标**：优化局部地图结构，维护地图一致性
**工作流程**：

```C++
// ORB-SLAM3/src/LocalMapping.cc
void LocalMapping::Run() {
    while (1) {
        // 1. 新关键帧处理
        ProcessNewKeyFrame();
        // 2. 地图点剔除
        MapPointCulling();
        // 3. 新地图点创建（三角化）
        CreateNewMapPoints();
        // 4. 局部BA优化
        Optimizer::LocalBundleAdjustment(...);
        // 5. 冗余关键帧剔除
        KeyFrameCulling();
    }
}
```

- **关键帧插入**：
  - 更新共视图（`covisibility graph`）
  - 更新生成树（`spanning tree`）
- **地图点管理**：
  - **三角化**：通过相邻关键帧创建新地图点（`CreateNewMapPoints()`）
  - **剔除劣质点**：观测不足/重投影误差过大
- **局部BA**：
  - 优化当前关键帧 + 共视关键帧 + 地图点
  - 使用 `g2o` 实现（`Optimizer::LocalBundleAdjustment`）
- **关键帧剔除**：删除冗余关键帧（>90%点被其他关键帧观测）

##### ③**回环检测（LoopClosing 线程）**

![Untitled diagram _ Mermaid Chart-2025-08-16-094100](D:\Desktop\SLAM\Pict\Untitled diagram _ Mermaid Chart-2025-08-16-094100.png)

**目标**：识别场景回环，校正累积误差
**工作流程**：

```C++
// ORB-SLAM3/src/LoopClosing.cc
void LoopClosing::Run() {
    while (1) {
        // 1. 检测回环候选帧
        if (DetectLoop()) {
            // 2. 计算Sim3变换（单目需估计尺度）
            if (ComputeSim3()) {
                // 3. 闭环校正
                CorrectLoop();
            }
        }
    }
}
```

1. **检测回环**（`DetectLoop()`）：

   - 基于DBoW2词袋模型检索相似关键帧
   - 连续性检测（连续3帧匹配成功）

2. **计算Sim3**（`ComputeSim3()`）：

   - 单目模式下需估计**尺度因子**（7自由度变换）
   - RANSAC求解相似变换矩阵

3. **闭环校正**（`CorrectLoop()`）：

   - **位姿图优化**：融合Sim3约束，优化Essential Graph

   ```C++
   Optimizer::OptimizeEssentialGraph(...);
   ```

   - **地图点融合**：合并重复地图点
   - **全局BA**（可选）：在独立线程执行全局优化

##### ④**可视化地图（Viewer 线程）**

![deepseek_mermaid_20250816_de4995](D:\Desktop\SLAM\Pict\deepseek_mermaid_20250816_de4995.png)

**目标**：实时展示SLAM状态和地图
**工作流程**：

```C++
// ORB-SLAM3/src/Viewer.cc
void Viewer::Run() {
    while (1) {
        // 1. 获取当前帧/地图数据
        GetCurrentState();
        // 2. 绘制相机轨迹
        DrawCameraTrajectory();
        // 3. 渲染地图点云
        DrawMapPoints();
        // 4. 显示关键帧位姿
        DrawKeyFrames();
    }
}
```

- **显示内容**：
  - **相机轨迹**：当前帧位姿 + 历史轨迹
  - **地图点云**：激活点（绿色）| 非激活点（黑色）
  - **关键帧**：相机坐标系 + 共视关系线
  - **状态信息**：跟踪状态/关键帧数/地图点数
- **多地图支持**（ORB-SLAM3特有）：
  - 同时可视化多个子地图（Atlas系统）
  - 用不同颜色区分激活地图和非激活地图

------

**单目模式特有机制**

1. **尺度漂移处理**：
   - 回环检测中通过Sim3估计尺度因子
   - 全局BA优化尺度一致性
2. **多地图系统**（Atlas）：
   - 跟踪丢失时创建新地图（`Tracking::Reset()`）
   - 重定位到旧地图时执行地图融合（`LoopClosing::MergeMaps()`）
3. **IMU融合**（IMU-MONOCULAR）：
   - **前端**：IMU预积分约束位姿估计
   - **后端**：视觉-惯性联合优化（VI-BA）



#### 2.ORB_SLAM3如何实现相机与IMU数据融合

##### 	①什么是IMU预积分？预积分的作用是什么？

​	长久以来一个问题一直困扰着我，视觉传感器与IMU数据融合，到底是哪方面进行了融合？视觉SLAM进行图像帧位姿估计的核心就是通过特征点之间的重投影找到最优变换矩阵，从而得到最优位姿估计，那再加上IMU能为这一过程贡献一份力量吗？

​	实际上，**IMU在ORB_SLAM3中只为整个运动系统测量了三个值——加速度、角速度、时间**。而IMU预积分的过程，则是基于这三个值，计算出JPa、JPg、JVa、JVg，用于计算相对**位移与速度的预积分量更新结果** + A、B、Q，用于计算**预积分量更新结果的协方差矩阵**。

![image-20250819202920492](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250819202920492.png)

  **`JPa -- 位置预积分关于加速度零偏的雅可比矩阵`**

  **`JPg -- 位置预积分关于陀螺仪零偏的雅可比矩阵`**

  **`JVa -- 速度预积分量关于加速度计零偏（bias_a）的雅可比矩阵`**

  **`JVg -- 速度预积分量关于陀螺仪零偏（bias_g）的雅可比矩阵`**

  **`δba -- 加速度的偏差值`**

s  **`δbg -- 陀螺仪的偏差值`**

![image-20250819202947689](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250819202947689.png)

**`C_{k+1}`: 更新后的协方差矩阵，包含了预积分量在当前时刻的不确定性。**

**`C_k`: 上一个时间步长的不确定性。**

**`A`: 状态转移矩阵。它描述了预积分量从 k 时刻到 k+1 时刻的线性演化。代码中的 `A.block<...>` 就是在填充这个矩阵。**

**`B`: 噪声输入矩阵。它描述了IMU的噪声如何影响预积分量的变化。代码中的 `B.block<...>` 就是在填充这个矩阵。**

**`Q`: IMU的噪声协方差矩阵。它包含了陀螺仪和加速度计的测量噪声（`Nga`）。**

 **`B、C的计算过程比较繁琐，要用到旋转矩阵dR、加速度白噪声Wacc之类的数据。`**

​	综上所述，根据IMU测量的测量值，我们最终可以得到三个数据P + V + C，其中**C（协方差矩阵）可以用来度量P、V的可置信度**，而**P、V**则代表**机器人在相邻两帧之间的相对位置与相对速度信息**。纯视觉SLAM机器人只能通过对不同帧特征点之间的运动进行位姿估计来估计自身的运动轨迹，而当遇到稀疏特征或是追踪丢失等情况时，机器人便会迷失自己在全局地图中的位置。因此，当视觉传感器与IMU数据进行融合后，IMU数据为全局地图**提供了现实世界中的绝对尺度**，即使在跟踪丢失等情况下，机器人也能凭借IMU数据精准认知自身在**全局地图乃至世界中的精准位置（纠正漂移问题）**。除此之外，**IMU数据的预积分**为SLAM的**后端优化提供了新的约束条件**，使得**后端位姿估计更加准确**；且在后端优化的过程中，**基于预积分中计算的一系列雅可比矩阵**，能够**快速计算出陀螺仪与加速度数据有些许偏差后的P、V期望值**，便于误差函数的快速计算，提高求解效率。

​	总之，IMU的预积分在进行图像追踪时直接进行，避免了将数据堆在后端优化时一个一个处理（计算雅可比矩阵十分麻烦），极大提高了后端优化的效率；同时，IMU的预积分储存了相邻两图像帧之间的相对运动信息，使得机器人的运动变得十分明确，为机器人自身定位提供了极大帮助。

##### 	②IMU数据如何作为条件约束后端优化？即P、V如何影响SLAM的后端位姿估计？

​	IMU数据融合部分位于LocalMapping.cc中Run()函数中。该函数在LocalMapping（后端）线程启动时同步开始循环执行。

​	IMU测量值数据将被设置为g2o的顶点，并设置IMU、陀螺仪、加速度约束边（连接由IMU测量值数据设置而成的顶点），利用g2o进行图优化。详细位于Optimizer::LocalInertialBA()函数中！

```C++
// 检查连续关键帧是否都有IMU数据且已完成预积分
if (pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated) {
    // 初始化预积分噪声参数
    pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
    
    // 获取两个关键帧的优化顶点（位姿、速度、陀螺仪零偏、加速度计零偏）
    VertexPose* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
    VertexVelocity* VV1 = optimizer.vertex(maxKFid + 3*(pKFi->mPrevKF->mnId) + 1);
    VertexGyroBias* VG1 = optimizer.vertex(maxKFid + 3*(pKFi->mPrevKF->mnId) + 2);
    VertexAccBias* VA1 = optimizer.vertex(maxKFid + 3*(pKFi->mPrevKF->mnId) + 3);
    
    VertexPose* VP2 = optimizer.vertex(pKFi->mnId);
    VertexVelocity* VV2 = optimizer.vertex(maxKFid + 3*(pKFi->mnId) + 1);
    VertexGyroBias* VG2 = optimizer.vertex(maxKFid + 3*(pKFi->mnId) + 2);
    VertexAccBias* VA2 = optimizer.vertex(maxKFid + 3*(pKFi->mnId) + 3);
    
    // 创建并设置IMU预积分约束边
    EdgeInertial* ei = new EdgeInertial(pKFi->mpImuPreintegrated);
    ei->setVertex(0, VP1);  // 上一关键帧位姿
    ei->setVertex(1, VV1);  // 上一关键帧速度
    ei->setVertex(2, VG1);  // 上一关键帧陀螺仪零偏
    ei->setVertex(3, VA1);  // 上一关键帧加速度计零偏
    ei->setVertex(4, VP2);  // 当前关键帧位姿
    ei->setVertex(5, VV2);  // 当前关键帧速度
    optimizer.addEdge(ei);
    
    // 创建并设置陀螺仪零偏随机游走约束边
    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0, VG1);
    egr->setVertex(1, VG2);
    egr->setInformation(pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse());
    optimizer.addEdge(egr);
    
    // 创建并设置加速度计零偏随机游走约束边
    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0, VA1);
    ear->setVertex(1, VA2);
    ear->setInformation(pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse());
    optimizer.addEdge(ear);
}
```

​	**g2o中设置的误差函数（其中之一）**：

```C++
 _error = obs - VPose->estimate().Project(VPoint->estimate(),cam_idx);
```

​	可见，尽管有IMU数据约束，但是后端优化的**本质仍是计算重投影误差（BA）**来**进行位姿估计与优化**。

> [!tip]
>
> ​	**此处理解稍有片面！实际上IMU与视觉传感器进行数据融合，确实本质仍是使误差函数最优，但是误差函数的表达式并非该g2o设置的误差函数！（详细往后面看就明白了）**

​	实际上！IMU数据和视觉数据的融合不是在边的内部计算中完成的，而是在更高层的**优化框架**中实现的。在EdgeInertial类中，也存在computeError()函数，实际上一般的边类定义中都定义了computeError()函数，但是在代码中不会主动调用。那么IMU数据是在何处起作用的？

​	这就要牵涉到g2o的精妙之处了。g2o是一种图优化求解非线性优化的方式，其由多个**顶点**与**边**构成，顶点代表着一种类型的数据，而边则相当于约束，边会连接一定数量个顶点，根据边中定义的误差计算函数，会同时优化边所连接的所有顶点，从而使顶点的数据达到理想状态。IMU与视觉传感器数据融合就是利用这一原理，将**同步优化所有状态变量**，IMU与视觉传感器通过**不同的边约束同一组状态顶点**，**同步优化**所有**状态变量**。

​	**论文中提到的视觉-惯性联合优化误差函数：**

![image-20250826120550149](../Typora Picture/image-20250826120550149.png)

![image-20250826121424188](../Typora Picture/image-20250826121424188.png)

​	**实际上**，可以看作**论文中给出的误差函数为总式**，而实际上代码中将**总误差函数分解为多个g2o边**进行分别求解。

![image-20250826123349973](../Typora Picture/image-20250826123349973.png)

##### ③偏置b的计算过程与作用

###### 一、偏置 `b` 是什么？

IMU偏置 `b` 是一个6维向量，通常分为两部分：

- **加速度计偏置** `b_a = [b_ax, b_ay, b_az]^T`
- **陀螺仪偏置** `b_g = [b_gx, b_gy, b_gz]^T`

它不是固定不变的常数，而是会随着温度、电路状态等缓慢漂移的变量。它的核心作用是**修正IMU的原始测量值**，得到更接近真实的角速度和加速度：

```
真实值 ≈ 原始测量值 - 偏置
```

###### 二、偏置 `b` 的核心作用

1. **修正测量，减少积分漂移**：这是最直接的作用。准确的偏置估计是获得可信的IMU积分结果（旋转、速度、位置）的前提。一个微小的偏置误差会在积分过程中被指数级放大。
2. **作为优化状态量，连接IMU与视觉**：在VIO中，偏置 `b` 与相机位姿、地图点一样，是**被优化器估算的状态变量**。通过将 `b` 纳入优化问题，IMU信息和视觉信息得以在一个统一的概率框架内共同修正系统状态。
3. **实现IMU的在线标定**：系统不需要依赖出厂时粗糙的标定数据。它可以在运行过程中**自动地、实时地**估计出当前环境下IMU的精确偏置，极大地提升了对低成本IMU的利用效率。

###### 三、偏置 `b` 的计算过程（优化过程）

偏置 `b` 不是被“计算”出来的，而是被“**优化**”出来的。整个过程是一个**不断迭代的期望最大化**过程：

1. **初始猜测**：通常假设偏置为0，即 `b = [0,0,0,0,0,0]^T`。基于这个初始值，进行IMU预积分，得到初始的预积分量 `(ΔR, Δv, Δp)` 和**雅可比矩阵 `J`**（如 `JRg`, `JVg`, `JVa` 等）。
2. **构建优化问题**：将以下变量设为顶点，构建图优化模型：
   - **待优化变量**：关键帧的位姿 `T`、速度 `v`、**IMU偏置 `b`**、地图点 `X`。
   - **约束边**：
     - `EdgeInertial`：连接连续两帧的 `T`, `v` 和 `b`，其误差由预积分量计算。
     - `EdgeReprojection`：连接帧的 `T` 和地图点 `X`，其误差为重投影误差。
3. **优化求解**：g2o等优化器开始工作，尝试微调所有顶点的值（包括给 `b` 一个变化量 `δb`）来最小化总误差。
4. **更新预积分量**：**这是最关键的一步**。当优化器改变 `b` 时，我们**不会**用新的 `b` 去重新积分所有IMU数据。而是使用**一阶近似**，利用步骤1中计算好的雅可比矩阵 `J` 来快速更新预积分量。
   `Δ量_{new} ≈ Δ量_{old} ○ Exp(J * δb)` （`○` 代表相应的更新操作）
   这个过程非常高效，是VIO算法实时性的保证。
5. **迭代收敛**：用更新后的预积分量重新计算误差，优化器继续迭代，直到总误差收敛到最小。此时得到的偏置 `b` 就是在当前数据下的**最优估计值**。
6. **持续跟踪**：偏置是缓慢变化的，因此这个优化过程会持续在整个SLAM运行过程中进行，不断跟踪和修正偏置的最新值。

------

###### 代码实例剖析

以您提供的代码为例，这是上述第4步（更新预积分量）中针对**旋转预积分量**的具体实现：

```C++
Eigen::Matrix3f Preintegrated::GetDeltaRotation(const Bias &b_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    // 1. 计算偏置的变化量 δb_g
    Eigen::Vector3f dbg;
    dbg << b_.bwx - b.bwx, b_.bwy - b.bwy, b_.bwz - b.bwz;

    // 2. 使用一阶近似更新旋转预积分量
    return NormalizeRotation(dR * Sophus::SO3f::exp(JRg * dbg).matrix());
}
```

- **`b_`**：优化器提议的**新偏置值**。
- **`b`**：预积分时所用的**原始偏置值**（存储在类成员变量中）。
- **`dbg`**：这就是偏置的变化量 **`δb_g`**。它是优化过程的**输出**，是优化器为了最小化误差而尝试做出的调整。
- **`JRg`**：这是预积分过程中计算并保存下来的**雅可比矩阵**。它编码了“旋转预积分量对陀螺仪偏置的敏感度”。
- **`JRg \* dbg`**：根据偏置的变化量 `δb_g`，估算出旋转预积分量需要相应做出的调整量（一个李代数向量）。
- **`Sophus::SO3f::exp(JRg \* dbg)`**：通过指数映射，将调整量（李代数）转换为一个旋转矩阵（李群），代表所需的**校正旋转**。
- **`dR \* ...`**：将原始的旋转预积分量 `dR` 右乘这个校正旋转矩阵，得到**更新后**的、更接近用新偏置 `b_` 积分所能得到的旋转预积分量。
- **`NormalizeRotation`**：保证输出矩阵的正交性，消除计算过程中的数值误差。

**这段代码的核心作用就是：当优化器想要试探一个新的偏置值 `b_` 时，它能极其高效地返回对应的旋转预积分量，从而让优化器能够评估在这个新偏置下总误差是大还是小，进而决定下一步的优化方向。** 它是连接“偏置优化”和“IMU测量”的桥梁，是整个联合优化得以实现的技术基石。



#### 3.ORB_SLAM3工程深入挖掘

##### 	①ORB_SLAM3工程原生支持单目、双目、RBG-D + IMU，似乎也原生支持相机数据实时传入（不用借助ROS2，直接读取USB相机数据）

![image-20250818103839801](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20250818103839801.png)

​	首先，标定相机内参，将相机内参与ORB特征点参数按格式写进yaml文件。（参考/Monocular/RealSense_D435i.yaml）

​	其次，根据相机驱动文件（参考/Monocular/mono_realsense_D435i.cc），传入所需要参数（若使用不同相机，驱动文件需要微调？or重新写一份？）

​	最后，传入所需参数（词典路径、SLAM参数（即第一步的yaml文件）、SLAM轨迹文件保存位置（可选传入）），即可实现实时SLAM稀疏建图。

> [!note]
>
> ​	**仅原生支持D435i与T265相机实时驱动！！！**（否则需要自己写代码！！！过于复杂，不如直接ROS2通信！！！）



#### 4.ORB_SLAM3代码阅读感想

​	①**代码量巨大**！！！随便一个文件就是一两千行的代码，核心线程更是四五千不止，而这样的文件有二十几个！！！阅读下来非常非常非常费力，需要足够的耐心与专注度完成注释与阅读。

​	②**牵涉过多C++特性的语法（偏工程化）**，如互斥锁与大量的面向对象编程。尽管这样的编程方式对新手阅读起来非常不友好，但是使得实现不同功能的代码之间既能够互相调用，又能够保证代码管理十分方便（层次清晰）。这样的编程方式是我后续需要加强的！

​	③ORB_SLAM虽然说算是SLAM最经典，也是较为古老的一批视觉SLAM算法，但是其在四大线程 -- 前端（Track）、后端（LocalMapping）、回环检测（LoopClothing）、地图（Map）中的代码实现细节，都**相较于书上讲的要复杂的多**（为工程项目的稳健性考量？难以想象这是如何写出来的）。目前位置虽然说基本看完了ORB_SLAM3的整个工作流程，但还是有些一头雾水（让我动手优化or写代码仍然无从下手），待看完ORB_SALM3衍生出的相关论文或许能好一些？

​	④**头晕....**

### 未来规划：

​	① 集中准备数学建模大赛！ 

​	② 完成ORB_SALM3衍生的论文阅读，学习ORB_SLAM3各线程的优化策略！

​	③ 根据论文所述回头看代码加深理解！ （一定要弄清楚四大线程的作用与具体实现过程！）

​	④ 复习线性代数 + 学习概率论知识，回头重新复习《视觉十四讲》理论部分，自行推导并理解ORB_SLAM3算法论文与代码中的后端优化过程！**（Important）**

## Date 9.8 ~ 

### 学习内容：

#### 0.基础知识普及计划 -- Git

[简介 - Git教程 - 廖雪峰的官方网站](https://liaoxuefeng.com/books/git/introduction/index.html)

##### 0.1.创建版本库

###### 〇 初始化当前文件夹为Git库

![image-20250912103455413](../Typora Picture/image-20250912103455413.png)

###### ①将文件添加进git库中

![image-20250911211036626](../Typora Picture/image-20250911211036626.png)

###### 	②查看仓库中所有文件的状态

![image-20250911211716540](../Typora Picture/image-20250911211716540.png)

###### 	③查看被修改文件的具体修改内容

![image-20250911211742515](../Typora Picture/image-20250911211742515.png)

##### 0.2版本回退与还原

###### 	①版本回退

​	**查看当前文件的所有版本**

![image-20250911211911068](../Typora Picture/image-20250911211911068.png)

​	**选择一个状态进行回退**

![image-20250911212050041](../Typora Picture/image-20250911212050041.png)

​	**回到上一个版本后重新回退到当前版本**（回退到上一版本后当前版本会自动消失，需要使用识别码索引）

​	**法一：**

![image-20250911212345311](../Typora Picture/image-20250911212345311.png)

​	**法二：**
![image-20250911212529758](../Typora Picture/image-20250911212529758.png)

###### 	②撤销修改 

![image-20250911215810435](../Typora Picture/image-20250911215810435.png)

​	撤销修改分为两种情况，**一为修改后未使用add添加至缓存区**，**二为修改后已使用add添加至缓存区。**

![image-20250911215230138](../Typora Picture/image-20250911215230138.png)

​	如果想**将已添加至缓存区的修改撤回至添加至缓存区之前（将缓存区中的文件拿出）**，则：

![image-20250911215411754](../Typora Picture/image-20250911215411754.png)

###### ③ 删除文件

![image-20250911220249568](../Typora Picture/image-20250911220249568.png)
	git add添加文件后删除，使用git status会显示**缓存区与当前工作区的区别**（工作区已不存在文件)。

​	**Choice1：从版本库中删除该文件**

![image-20250911220427609](../Typora Picture/image-20250911220427609.png)

​	**Choice2：删错文件，恢复文件**

![image-20250911220537569](../Typora Picture/image-20250911220537569.png)

##### 0.3远程仓库

###### ① GitHub远程仓库创建SSH链接

​	**本地链接创建**

![image-20250912075002964](../Typora Picture/image-20250912075002964.png)

​	创建成功的**Key保存在主目录下（C:/User/Li/.ssh）**

​	**GitHub创建链接**

![image-20250912080019274](../Typora Picture/image-20250912080019274.png)

###### ② 将本地代码添加到远程库GitHub中

​	**在GitHub中创建新Git库**

![image-20250912081748958](../Typora Picture/image-20250912081748958.png)

​	**建立与远程Git库的链接**

![image-20250912082001229](../Typora Picture/image-20250912082001229.png)

```shell
git remote add [远程库名字] [git仓库链接]
```

![image-20250912082054265](../Typora Picture/image-20250912082054265.png)

​	**将本地文件上传至远程Git库中**

![image-20250912082211965](../Typora Picture/image-20250912082211965.png)

###### ③ 删除远程Git库

![image-20250912082309025](../Typora Picture/image-20250912082309025.png)

​	实际上只是解除了**本地和远程的绑定关系**！

###### ④ 从远程库克隆

![image-20250912094214340](../Typora Picture/image-20250912094214340.png)

​	与建立远程Git库连接类似，此处也是直接**使用SSH链接对远程Git库进行克隆**即可。

##### 0.4 分支管理

![image-20250912094757686](../Typora Picture/image-20250912094757686.png)

###### ①创建、合并与删除分支

![image-20250912095043482](../Typora Picture/image-20250912095043482.png)

​	**创建分支，名字可以自定义。**

![image-20250912095206735](../Typora Picture/image-20250912095206735.png)

​	查看当前分支，**当前所在分支前会标记*号**。

![image-20250912095600995](../Typora Picture/image-20250912095600995.png)

![image-20250912101203086](../Typora Picture/image-20250912101203086.png)

​	**切换回主分支。**（也可以修改分支名字切换到其他分支）

![image-20250912095739881](../Typora Picture/image-20250912095739881.png)

​	在**dev分支进行修改的文件，在master主分支中并没有同步**。

![image-20250912095704016](../Typora Picture/image-20250912095704016.png)

​	因此，我们需要**对master分支与dev分支进行合并**。

![image-20250912095905846](../Typora Picture/image-20250912095905846.png)

​	**对dev分支进行删除**

![image-20250912102915145](../Typora Picture/image-20250912102915145.png)

​	如果在**mater主分支与dev分支同时对文件进行了修改**，merge时会发生冲突，**打开编辑器会提示你选择一项进行修改。**

​	修改完成后即可再次手动提交。

![image-20250912103051490](../Typora Picture/image-20250912103051490.png)

​	**查看分支合并日志。**

###### 	② 分支管理策略

![image-20250912110750586](../Typora Picture/image-20250912110750586.png)

​	**合并分支时加上 --no-ff 可以使用普通模式进行合并**，合并后的历史有分支，能看出来曾经做过合并，而**fast forward合并则看不出来曾经有合并**。

###### 	③Bug管理

![image-20250913092330217](../Typora Picture/image-20250913092330217.png)

![image-20250912120952469](../Typora Picture/image-20250912120952469.png)

​	当**遇到突发事件，需要创建一个新分支**，并且**需要保存当前未完成工作的分支**，则**使用stash功能将工作现场暂存**。

![image-20250912121138958](../Typora Picture/image-20250912121138958.png)

​	当**处理完突发事件后，回到将工作现场暂存的分支**，使用**git stash list 查看当前有多少个工作现场被储存**。

![image-20250912121301088](../Typora Picture/image-20250912121301088.png)

![image-20250912121422438](../Typora Picture/image-20250912121422438.png)

​	选择一种方式来恢复工作现场—— **不删除工作现场的暂存并恢复 or 删除工作现场的暂存并恢复**。

![image-20250912174521820](../Typora Picture/image-20250912174521820.png)

​	然而**当修复bug回到暂存工作区后，工作中的bug仍未被修改，此时则可以使用cherr-pick快速同步修改**。

​	即：**继续使用git stash将当前工作区暂存**；再**回到master主分支查看文件提交记录，找到修改bug的commit并复制部分代号**；**使用cherry-pick命令将两部分修改合并，即完成bug修改同步**！

![image-20250913100210786](../Typora Picture/image-20250913100210786.png)

###### ④ Feature管理

![image-20250913092304252](../Typora Picture/image-20250913092304252.png)

![image-20250913092820531](../Typora Picture/image-20250913092820531.png)

​	实际上就是**开了个新分支feature**，在**新分支feature上修改然后在dev分支中与新分支feature进行合并**。

![image-20250913093104838](../Typora Picture/image-20250913093104838.png)

​	但是，**如果新内容不需要进行合并**了，**删除feature分支时只需要将-d写为-D**即可**强制删除有内容的分支**。

![image-20250913100146088](../Typora Picture/image-20250913100146088.png)

###### ⑤ 多人协作

![image-20250913094045963](../Typora Picture/image-20250913094045963.png)

![image-20250913094124373](../Typora Picture/image-20250913094124373.png)

​	**查看远程仓库详细信息。**

![image-20250913094241603](../Typora Picture/image-20250913094241603.png)

​	推送时，可以**选择分支进行推送**！一般而言都是在**dev分支上进行开发**工作，**bug分支用于修复bug**，**feature分支用于添加新功能**。是否推送至远程仓库，**取决于是否有人需要与你的工作同步**！

![image-20250913094623941](../Typora Picture/image-20250913094623941.png)

![image-20250913095111575](../Typora Picture/image-20250913095111575.png)

​	单纯**从远程仓库origin中克隆仓库，无法克隆dev分支**，需要**手动创建远程仓库origin的dev分支到本地**，从而在本地进行开发！

![image-20250913095634347](../Typora Picture/image-20250913095634347.png)

​	当**两个人向dev分支中提交了一份有冲突的文件**，git会报错。此时需要**使用pull将已上传的文件下拉**，在**本地解决冲突后再重新上传**。

![image-20250913095846625](../Typora Picture/image-20250913095846625.png)

​	然而，**在pull之前，远程仓库需要知道pull哪一部分**，因此需要**指定本地分支与远程分支的链接**！（如图）

**总结！！！**

![image-20250913100110905](../Typora Picture/image-20250913100110905.png)

##### 0.5 标签管理

![image-20250913101132196](../Typora Picture/image-20250913101132196.png)

###### 	① 创建标签

![image-20250913101846935](../Typora Picture/image-20250913101846935.png)

![image-20250913103459526](../Typora Picture/image-20250913103459526.png)

​	实际上，**打标签是为上一次commit打标签**（而**不是为即将commit打标签**）。此外，还能**使用-a与-m指定当前标签名与其说明文字**。

![image-20250913103135918](../Typora Picture/image-20250913103135918.png)

​	使用**git tag可以查看所打标签的情况**。

![image-20250913103212735](../Typora Picture/image-20250913103212735.png)

​	使用**log找到未打标签的版本commit**，再使用 **git tag [tag] [commit_id]** 可以对该版本**补打标签**。

![image-20250913104022755](../Typora Picture/image-20250913104022755.png)

​	使用 **git show [tag] 能够看到当前标签的说明文字**。

![image-20250913104159576](../Typora Picture/image-20250913104159576.png)

###### ② 操作标签

![image-20250913104850035](../Typora Picture/image-20250913104850035.png)

​	**将标签推送到远程**仓库（创建的标签默认**不会自动推送到远程仓库**！！）

​	**删除本地标签：**

![image-20250913104641639](../Typora Picture/image-20250913104641639.png)

​	**删除远程仓库标签：**

![image-20250913105139526](../Typora Picture/image-20250913105139526.png)

![image-20250913105345934](../Typora Picture/image-20250913105345934.png)

#### 1.当务之急速通概率论与数理统计。

#### 2.概率论与数理统计与SLAM基础数学知识混学。

##### 	①SO(3)空间中，对于雅克比矩阵J(Φ)的推导

###### 0. 我们的目标：重新明确问题

我们的目标是找到一个3x3矩阵 `J`，它能描述**旋转向量 `φ` 的变化率（`φ̇`）** 和**刚体自身的角速度 `ω`** 之间的关系。

这个关系就是：

![image-20250918113839586](../Typora Picture/image-20250918113839586.png)

`φ̇` 是参数空间 `so(3)` 中的速度，而 `ω` 是刚体在物理世界中实际的角速度（也用 `so(3)` 向量表示）。

**雅克比矩阵存在的意义：**

![image-20250918115539653](../Typora Picture/image-20250918115539653.png)

------

###### 第1步：角速度的严格数学定义

这是整个推导的基石。对于一个随时间变化的旋转矩阵 R(t)，它的时间导数 Ṙ(t) 和它在**物体坐标系（Body Frame）**下的角速度 ω(t) 之间的关系被定义为：

![image-20250918113853105](../Typora Picture/image-20250918113853105.png)

这个公式的含义是：旋转矩阵的瞬时变化，等于用角速度 ω(t) 代表的那个微小旋转（一个反对称矩阵 ω(t)^），左乘在当前旋转 R(t) 上。

从这个定义式，我们可以反解出 ω(t)^：

![image-20250918113901709](../Typora Picture/image-20250918113901709.png)

这就是我们将要计算的目标。我们只要能把右边的 Ṙ(t)R(t)⁻¹ 算出来，并整理成 (某个矩阵 * φ̇)^ 的形式，那么“某个矩阵”就是我们想要的雅可比 J。

------

###### 第2步：计算 `Ṙ(t)` (指数映射的导数)

这是整个推导最核心、最困难的一步。我们知道 `R(t) = exp(φ(t)^)`。现在需要计算它的时间导数 `Ṙ(t)`。

直接对泰勒级数 `exp(A) = I + A + A²/2! + ...` 求导会遇到矩阵不可交换的问题，非常复杂。这里我们需要使用一个矩阵指数求导的**标准公式**（这个公式本身的推导需要用到更深的数学，但公式本身是后续推导的基石）。

对于任意随时间变化的矩阵 A(t)，其指数的导数有一个积分形式的解：

![image-20250918113911580](../Typora Picture/image-20250918113911580.png)

这个公式可能看起来很吓人，但它正是我们需要的“不省略”的那个环节。

现在，我们把 A(t) = φ(t)^ 和 Ȧ(t) = φ̇(t)^ 代入这个公式：

![QianJianTec1758166638858](../Typora Picture/QianJianTec1758166638858.png)

(为了简洁，我们暂时省略 (t))

------

###### 第3步：化简积分项

现在我们来处理积分内部的 exp(sφ^) φ̇^ exp(-sφ^)。

这其实是李群中的伴随表示 (Adjoint representation)。有一个重要的恒等式：

![image-20250918113924795](../Typora Picture/image-20250918113924795.png)

其中 R 是旋转矩阵，A 是一个向量。

令 R(s) = exp(sφ^)，它的逆（转置）就是 exp(-sφ^)。令向量 A = φ̇。

那么：

![image-20250918113935876](../Typora Picture/image-20250918113935876.png)

这个恒等式极大地简化了我们的问题。现在 `Ṙ(t)` 的表达式变为：

![QianJianTec1758166795339](../Typora Picture/QianJianTec1758166795339.png)因为 `(·)^` hat算子是线性的，积分也是线性的，我们可以把积分和hat算子交换顺序：

![QianJianTec1758166843138](../Typora Picture/QianJianTec1758166843138.png)由于 `φ̇` 不依赖于积分变量 `s`，我们可以把它提到积分外面：

![QianJianTec1758166873746](../Typora Picture/QianJianTec1758166873746.png)

------

###### 第4步：求解雅可比 `J`

现在我们得到了 `Ṙ(t)` 的一个完美形式。让我们把它与第1步中的角速度定义进行比较：

- **角速度定义**: R˙(t)=ω(t)∧R(t)
- **我们刚推得**: R˙(t)=((∫01R(s)ds)ϕ˙)∧R(t)

通过逐项对比，我们可以清晰地看到：

![image-20250918114146001](../Typora Picture/image-20250918114146001.png)

对等式两边同时使用 `(·)∨` vee算子（hat的逆运算），得到：

![image-20250918114156005](../Typora Picture/image-20250918114156005.png)

我们再回顾一下最初的目标 ω(t) = J(φ)φ̇(t)。

显而易见，左雅可比 J_l(φ) 就是这个积分项:

![image-20250918114208537](../Typora Picture/image-20250918114208537.png)

###### 结论：为什么是积分？

到这里，我们逻辑上就完全清晰了：

1. 雅可比 `J` 来自于**导数**关系 `ω = Jφ̇`。
2. 为了求解 `J`，我们必须计算 `R` 的导数 `Ṙ`。
3. 计算 `Ṙ` 的标准数学工具给出的结果**恰好包含一个积分**。
4. 因此，`J` 的**最终计算公式**就是一个积分。

**积分不是定义，而是求解导数关系后得到的结果。**

------

###### 附：求解积分得到最终公式（此部分在之前的回答中已推导）

现在，我们来完成最后一步，求解这个积分，得到可以直接计算的代数表达式。

![image-20250918114220400](../Typora Picture/image-20250918114220400.png)

对 `s` 积分得到：

![image-20250918114234093](../Typora Picture/image-20250918114234093.png)

这就是最终的左雅可比公式。



#### 3.混合学习后重新读论文！！！SLAM代码重读计划延后！！！

### 未来规划：