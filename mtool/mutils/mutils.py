import numpy as np


## 归一化 (0,1)标准化
def norm_zero_one(array, span=None):
    """
    根据所给数组的最大值、最小值，将数组归一化到0-1
    :param span:
    :param array: 数组
    :return: array: numpy格式数组
    """
    array = np.asarray(array).astype(np.float)
    if span is None:
        mini = array.min()
        maxi = array.max()
    else:
        mini = span[0]
        maxi = span[1]
        array[array < mini] = mini
        array[array > maxi] = maxi

    range = maxi - mini

    def norm(x):
        return (x - mini) / range

    return np.asarray(list(map(norm, array))).astype(np.float)


## 归一化，Z-score标准化
def norm_z_score(array):
    '''
    根据所给数组的均值和标准差进行归一化，归一化结果符合正态分布，即均值为0，标准差为1
    :param array: 数组
    :return: array: numpy格式数组
    '''
    array = np.asarray(array).astype(np.float)
    mu = np.average(array)  ## 均值
    sigma = np.std(array)  ## 标准差

    def norm(x):
        return (x - mu) / sigma

    return np.asarray(list(map(norm, array))).astype(np.float), mu, sigma


## 去除标点符号
def removePunctuation(text, replaced, besides=None):
    '''
    去除标点符号
    :param text: 文本信息
    :param replaced: 替换的字符
    :param besides: 不需要去除的标点符号 []
    :return: newText
    '''
    import string
    temp = []
    for c in text:
        if c in string.punctuation:
            if besides == None:
                temp.append(c)
            else:
                if c in besides:
                    temp.append(c)
                else:
                    temp.append(replaced)
        else:
            temp.append(c)
    newText = ''.join(temp)
    return newText


## 对3D进行插值
def get_3d_interpolation(image, dst_shape):
    '''
    :param image: numpy image
    :param dst_shape: expected image shape
    :return:
    '''
    from scipy.interpolate import RegularGridInterpolator

    image = np.asarray(image)
    shape = image.shape

    px, py, pz = shape
    sx, sy, sz = dst_shape
    # print("shape:{} px:{} py:{} pz:{} sx:{} sy:{} sz:{}".format(shape, px, py, pz, sx, sy, sz))

    x = np.linspace(1, sx, px)
    y = np.linspace(1, sy, py)
    z = np.linspace(1, sz, pz)

    fn = RegularGridInterpolator((x, y, z), image)

    x = np.linspace(1, sx, sx)
    y = np.linspace(1, sy, sy)
    z = np.linspace(1, sz, sz)

    x_pts, y_pts, z_pts = np.meshgrid(x, y, z, indexing='ij')
    pts = np.concatenate([x_pts.reshape(-1, 1), y_pts.reshape(-1, 1), z_pts.reshape(-1, 1)], axis=1)

    resuts = np.asarray(fn(pts)).reshape((sx, sy, sz))
    return resuts


## 对3D进行旋转
def get_3d_rotate(image, angle, plane='z', mode="nearest"):
    '''
    :param image: numpy image
    :param angle: (y_rotate_angle,x_rotate_angle)
    :return:
    '''

    from scipy.ndimage import rotate

    if plane == 'z':
        axes = (0, 1)
    elif plane == 'y':
        axes = (0, 2)
    elif plane == 'x':
        axes = (1, 2)
    else:
        raise Exception("No rotation plane specified")

    ## reshape = False keeps the shape of original image
    image = rotate(image, angle, axes=axes, reshape=False, mode=mode)
    return image


def test_get_3d_interpolation():
    from mtool.mio import get_medical_image, save_medical_image
    img, o, s, d, _ = get_medical_image("../../data/Changhai/image/cao xue e.nrrd")
    print(img.shape)
    s = (s[0], s[1], s[2] / (84.5 / 80))

    result = get_3d_interpolation(img, map(int, [84.5, img.shape[1], img.shape[2]]))
    print(result.shape)
    save_medical_image(result, "../../cao.nrrd", o, s, d)


def test_get_3d_rotate():
    from mtool.mio import get_medical_image, save_medical_image
    img, o, s, d, _ = get_medical_image("../../data/Changhai/image/bai yu zhen.nrrd")
    org, _, _, _, _ = get_medical_image("../../data/Changhai/organ/bai yu zhen.nrrd")

    img = get_3d_rotate(image=img, angle=45, plane='x')
    org = get_3d_rotate(image=org, angle=45, plane='x')
    org = np.asarray(org > 0.1).astype(np.int)

    save_medical_image(img, "../../bai-img.nrrd", o, s, d)
    save_medical_image(org, "../../bai-org.nrrd", o, s, d)


if __name__ == "__main__":
    image = np.random.random([2, 2, 2]) * 2000

    print(image.max())
    print(image.min())

    image = norm_zero_one(image, span=[0, 2400])
    print(image.max())
    print(image.min())

    image = norm_zero_one(image)
    print(image.max())
    print(image.min())

    # test_get_3d_rotate()
    # test_get_3d_interpolation()
