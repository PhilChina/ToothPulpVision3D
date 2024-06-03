import matplotlib.pyplot as plt

import numpy as np
import os

def draw_surface(image, show_axis=True, rstride=1, cstride=1):
    '''
    画出三维形式的分布图
    :param image:      分布的2D矩阵
    :param show_axis:  是否显示坐标轴
    :param rstride:    行间距
    :param cstride:    列间距
    :return:
    '''
    image = np.asarray(image)

    assert len(image.shape) == 2, "image is not a 2D image"

    width, height = image.shape

    X = np.linspace(0, width, width)
    Y = np.linspace(0, height, height)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure()

    # 创建3d图形的两种方式
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')

    # rstride:行之间的跨度  cstride:列之间的跨度
    # rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现
    # vmax和vmin  颜色的最大值和最小值
    ax.plot_surface(X, Y, image, rstride=rstride, cstride=cstride, cmap=plt.get_cmap('rainbow'))

    # zdir : 'z' | 'x' | 'y' 表示把等高线图投射到哪个面
    # offset : 表示等高线图投射到指定页面的某个刻度
    ax.contourf(X, Y, image, zdir='z', offset=-2)

    # 设置图像z轴的显示范围，x、y轴设置方式相同
    ax.set_zlim(-2, 2)

    # 去掉坐标轴
    if not show_axis:
        plt.axis('off')
    plt.show()

def _relative_path(path):
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    return os.path.join(script_dir, path)


def get_knot_mesh():
    mesh = o3d.io.read_triangle_mesh(_relative_path("../img/knot.ply"))
    mesh.compute_vertex_normals()
    return mesh

if __name__ == '__main__':
    import open3d as o3d

    print("Testing mesh in open3d ...")
    mesh = get_knot_mesh()
    o3d.visualization.draw_geometries([mesh])
    print(mesh)
    print('Vertices:')
    print(np.asarray(mesh.vertices))  # 每个点的坐标xyz
    print('Triangles:')
    print(np.asarray(mesh.triangles))  # 每个面的三个点的索引
