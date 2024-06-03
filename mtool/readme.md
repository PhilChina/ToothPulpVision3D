[TOC]



## mio.py

### 得到一组dicom序列图像，get_dicom_image

```python
## 得到一组dicom序列图像
def get_dicom_image(dire):
    '''
    加载一组dicom序列图像
    :param dire: dicom序列所在的文件夹路径，E.g. "E:/Work/Database/Teeth/origin/1/"
    :return: (array,origin,spacing,direction)
    array:  图像数组
    origin: 三维图像坐标原点
    spacing: 三维图像坐标间距
    direction: 三维图像坐标方向
    注意：实际取出的数组不一定与MITK或其他可视化工具中的方向一致！
    可能会出现旋转\翻转等现象，这是由于dicom头文件中的origin,spacing,direction的信息导致的
    在使用时建议先用matplotlib.pyplot工具查看一下切片的方式是否异常，判断是否需要一定的预处理
    '''
```

### 得到出2D/3D的医学图像(除.dcm序列图像)，get_medical_image

```python
## 得到2D/3D的医学图像(除.dcm序列图像)
def get_medical_image(path):
    '''
    加载一幅2D/3D医学图像(除.dcm序列图像)，支持格式：.nii, .nrrd, ...
    :param path: 医学图像的路径
    :return:(array,origin,spacing,direction)
    array:  图像数组
    origin: 三维图像坐标原点
    spacing: 三维图像坐标间距
    direction: 三维图像坐标方向
    image_type: 图像像素的类型
    注意：实际取出的数组不一定与MITK或其他可视化工具中的方向一致！
    可能会出现旋转\翻转等现象，这是由于dicom头文件中的origin,spacing,direction的信息导致的
    在使用时建议先用matplotlib.pyplot工具查看一下切片的方式是否异常，判断是否需要一定的预处理
    '''
```

### 得到一张普通格式的图像数组，get_normal_image

```python
## 加载一张普通格式图片 2D
def get_normal_image(path):
    '''
    加载一幅普通格式的2D图像，支持格式：.jpg, .jpeg, .tif ...
    :param path: 医学图像的路径
    :return: array: numpy格式
    '''
```

### 将数组保存为3D医学图像格式，save_medical_image

```python
## 将numpy数组保存为3D医学图像格式，支持 .nii, .nrrd
def save_medical_image(array, origin, spacing, direction, target_path, type=None):
    '''
    将得到的数组保存为医学图像格式
    :param array: 想要保存的医学图像数组，为避免错误，这个函数只识别3D数组
    :param origin:读取原始数据中的原点
    :param space: 读取原始数据中的间隔
    :param direction: 读取原始数据中的方向
    :param target_path: 保存的文件路径，注意：一定要带后缀，E.g.,.nii,.nrrd SimpleITK会根据路径的后缀自动判断格式，填充相应信息
    :param type: 像素的储存格式
    :return: None 无返回值
    注意，因为MITK中会自动识别当前载入的医学图像文件是不是标签(label)【通过是否只有0,1两个值来判断】
    所以在导入的时候，MITK会要求label的文件格式为unsigned_short/unsigned_char型，否则会有warning
    '''
```

### 将数组保存为普通的2D图像，save_normal_image

```python
## 将numpy数组保存为普通的2D图像，支持.jpg, .jpeg, .tif
def save_normal_image(array,target_path):
    '''
    将得到的数组保存为普通的2D图像
    :param array: 想要保存的图像数组
    :param target_path: 保存的文件路径，注意：一定要带后缀，E.g.,.jpg,.png,.tif
    :return: None 无返回值
    '''
```

### 按顺序得到单签目录下，所有文件(包括文件夹)的名字，get_files_name

```python
## 按顺序得到当前目录下，所有文件（包括文件夹）的名字
def get_files_name(dire):
    '''
    按顺序得到当前目录下，所有文件（包括文件夹）的名字
    :param dire: 文件夹目录
    :return:files[list]，当前目录下所有的文件（包括文件夹）的名字，顺序排列
    '''
```

-----------------------------

### 读取DICOMDIR文件，get_dicomdir_info

```python
def get_dicomdir_info(path):
    '''
    读取DICOMDIR中的信息，返回以下主要信息：
    PatientID, PatientName, StudyDate, StudyDescription, StudyID, SerialNumber. SeriesCount,   SeriesDescription, SeriesFolder, SeriesModality
    病人ID，   病人名字，    病历日期，  病历描述，        病历ID，  序列号，      序列包含图片数， 序列描述，        序列所在文件夹，序列模态
    :param path: DICOMDIR文件的路径
    :return: list,每一个序列的以上基本信息
    DICOMDIR文件格式说明：https://www.medicalconnections.co.uk/kb/DICOMDIR/
    https://pydicom.github.io/pydicom/dev/auto_examples/input_output/plot_read_dicom_directory.html
    四级结构：PATIENT –> STUDY –> SERIES –> IMAGE
    '''
```

## mutils.py

### 归一化，(0,1)标准化，norm_zero_one

```python
## 归一化 (0,1)标准化
def norm_zero_one(array):
    '''
    根据所给数组的最大值、最小值，将数组归一化到0-1
    :param array: 数组
    :return: array: numpy格式数组
    '''
```

### 归一化，Z-score标准化，norm_z_score

```python
## 归一化，Z-score标准化
def norm_z_score(array):
    '''
    根据所给数组的均值和标准差进行归一化，归一化结果符合正态分布，即均值为0，标准差为1
    :param array: 数组
    :return: array: numpy格式数组
    '''
```

### 求一幅标签图像的重心（中心点），get_center_point

```python
## 求一幅标签图像的重心（中心点）
def get_center_point(image):
    '''
    得到图片的重心点
    :param image: numpy格式数据，得到图片的重心位置，可适应多维度
    :return: List，维度和label一样，得到每一个维度上的图片重心坐标[0为起始点]
    理论上可适应各种图片，但因为非标签图片像素值较大，运算时会出现溢出报错，所以这里只允许0,1标签图像
    '''
```

### 将一个二维数组进行左右翻转，get_flip_lr

```python
## 二维数组进行左右翻转
def get_flip_lr(array):
    '''
    将二维数组进行左右翻转
    :param array: 二维数组
    :return: numpy数组
    '''
```

### 将一个二维数组进行上下翻转，get_flip_ud

```python
## 二维数组进行上下翻转
def get_flip_ud(array):
    '''
    将二维数组进行上下翻转
    :param array: 二维数组
    :return: numpy数组
    '''
```

### 获得标签的最大连通区域，get_largest_connected_region，暂无验证与示例(补)

```python
## 获得标签的最大连通区域
def get_largest_connected_region(mask,background=0):
    '''
    将得到的标签求最大连通区域
    :param mask: 得到的标签数组
    :param background: 标签数组的背景，默认是0
    :return: largest_connected_region，numpy数组，只包含0,1标签
    因为用到skimage中的函数label，所以这里以mask指代标签
    '''
```



## mpretreat.py

### 将圆形视野中的前景提取出来，getForeground

```python
## 将圆形视野中的前景提出出来
def getForeground(indire, outdire):
    '''
    将CT/MRI图中的前景提取出来，将保存好的文件以原格式储存到另一文件夹中
    因为一些医学图像的窗位是圆形的，E.g.圆形窗位中的背景是-1024左右，而圆形窗位之外的方框背景在-2048
    这个函数将所有的背景归一到-1024
    :param indire: 医学图像的文件夹
    :param outdire: 保存到的医学图像文件夹
    :return: None
    注意！此函数不可以使用.dcm格式数据
    '''
```

![](E:\Work\Github\Medical_Image_Toolkit\mtool\img\origin.png)

![](E:\Work\Github\Medical_Image_Toolkit\mtool\img\getForeground.png)

### 得到标签所在的长方形矩形（不是最小外接矩），getBounding

```python
## 得到label的外接矩形，不是最小外接，是长方矩形
def getBounding(arrays):
    '''
    得到他的各种边界
    https://blog.csdn.net/dcrmg/article/details/89927816
    :param arrays: 3D 图像数组
    :return: list 得到切割的原点以及每条边的距离
    '''
```

### 根据器官标签的位置对相应的器官标签，器官图像，肿瘤标签进行剪裁，clipImage

```python
## 根据器官的位置进行裁剪
def clipImage(organLabel, organImage=None, tumorLabel=None):
    '''
    根据器官的位置对三个数据进行裁剪
    :param organLabel: 器官的标签，根据器官的标签来对三个数据进行剪裁
    :param organImage: MRI/CT图像数组
    :param tumorLabel: 可选，肿瘤标签数组
    :return: [organLabel,organImage]/[organLabel,organImage,tumorLabel]
    '''
```





































