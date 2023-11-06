# import matplotlib.pyplot as plt
# import numpy as np
# from config import args_parser
# from scipy.signal import savgol_filter
#
# filepath1 = './result_sample_725/seed1cora_LDA1.0_fedavg_L-hop0_maxacc0.67.dat'
# filepath2 = './result_sample_725/seed1cora_LDA1.0_fedprox_L-hop0_maxacc0.726.dat'
# filepath3 = './result_sample_725/seed1cora_LDA1.0_pos_fedprox_L-hop0_maxacc0.71.dat'
# filepath4 = './result_sample_725/seed1cora_LDA1.0_sample_pos_fedprox_L-hop0_maxacc0.704.dat'
# filepath5 = './result_sample_725/seed1cora_LDA1.0_decaysample_pos_fedprox_L-hop0_maxacc0.744.dat'
#
#
#
# def openfile(filepath):
#     temp = []
#     with open(filepath, 'r') as f:
#         while True:
#             line = f.readline()
#             # 将末尾的\n符号删除后，如果没有数据，则break
#             if line.rstrip('\n') == '':
#                 break
#             line = float(line[0:len(line)-1])
#             temp.append(line)
#     return temp
#
#
# if __name__ == '__main__':
#     args = args_parser()
#
#     file1 = openfile(filepath1)
#     file1 = savgol_filter(file1, 69, 3)
#
#     file2 = openfile(filepath2)
#     file2 = savgol_filter(file2, 69, 3)
#
#     file3 = openfile(filepath3)
#     file3 = savgol_filter(file3, 69, 3)
#
#     file4 = openfile(filepath4)
#     file4 = savgol_filter(file4, 69, 3)
#
#     file5 = openfile(filepath5)
#     file5 = savgol_filter(file5, 69, 3)
#
#
#     # plt.xlim(0, 175)
#     # plt.ylim(0.98, 0.995)
#
#     plt.plot(np.array(file1), c='c', label='fedavg', linestyle='-')
#     plt.plot(np.array(file2), c='b', label='fedprox', linestyle='-')
#     plt.plot(np.array(file3), c='r', label='pos-prox', linestyle='--')
#     plt.plot(np.array(file4), c='g', label='sample pos-prox', linestyle='-.')
#     plt.plot(np.array(file5), c='#FFA500', label='05decay sample pos-prox', linestyle='-')
#
#
#     plt.legend(loc='best')
#     plt.ylabel('acc')
#     plt.xlabel('epoch')
#     plt.grid()
#
#     #
#     plt.savefig('./result_sample_725/seed1_LDA1_sample')
#     plt.show()



# # confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt
#
# classes = ['0', '1', '2', '3', '4', '5', '6']
# # dirichlet LDA5
# # confusion_matrix = np.array([(79, 10, 97, 133, 59, 25, 29),(30, 56, 48, 109, 45, 50, 9),(32, 19, 58, 78, 48, 57, 29),(23, 32, 85, 123, 69, 73, 36),(83, 22, 48, 89, 60, 14, 24),(28, 20, 44, 127, 77, 34, 23), (76, 58, 38, 159, 68, 45, 30)],dtype=int)
#
# # rate 0.5
# confusion_matrix = np.array([(193, 8, 23, 57, 28, 21, 20),(14, 115, 14, 33, 20, 13, 7),(32, 20, 238, 67, 29, 20, 12),(51, 37, 66, 537, 61, 37, 29),(29, 17, 37, 58, 247, 27, 11),(18, 15, 28, 35, 26, 171, 5), (14, 5, 12, 31, 15, 9, 96)],dtype=int)
#
# plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
# plt.title('rate Non-IID')
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes)
# plt.yticks(tick_marks, classes)
#
# thresh = confusion_matrix.max() / 2.
# # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
# # ij配对，遍历矩阵迭代器
# iters = np.reshape([[[i, j] for j in range(7)] for i in range(7)], (confusion_matrix.size, 2))
# for i, j in iters:
#     plt.text(j, i, format(confusion_matrix[i, j]))  # 显示对应的数字
#
# plt.ylabel('label')
# plt.xlabel('Client')
# plt.tight_layout()
# plt.show()



# import matplotlib.pyplot as plt
# import numpy as np
#
# # epoch,acc,loss,val_acc,val_loss
# x_axis_data = [0.3, 0.5, 0.8, 1.0]
# y_axis_data1 = [0.584, 0.672, 0.654, 0.658]
# y_axis_data2 = [0.46, 0.618, 0.544, 0.544]
#
#
# # 画图
# plt.plot(x_axis_data, y_axis_data1, 'b*--', alpha=0.5, linewidth=1, label='fedprox')  # '
# plt.plot(x_axis_data, y_axis_data2, 'rs--', alpha=0.5, linewidth=1, label='fedavg')
#
# ## 设置数据标签位置及大小
# for a, b in zip(x_axis_data, y_axis_data1):
#     plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)  #  ha='center', va='top'
# for a, b1 in zip(x_axis_data, y_axis_data2):
#     plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=8)
#
#
#
# plt.legend()  # 显示上面的label
# plt.xlabel('sample rate')
# plt.ylabel('accuracy')  # accuracy
#
# plt.savefig('./result/sample_rate05')
# plt.show()




import matplotlib.pyplot as plt
import numpy as np
from config import args_parser
from scipy.signal import savgol_filter
from matplotlib.patches import ConnectionPatch

def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom', x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)

def openfile(filepath):
    temp = []
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            # 将末尾的\n符号删除后，如果没有数据，则break
            if line.rstrip('\n') == '':
                break
            line = float(line[0:len(line)-1])
            temp.append(line)
    return temp
#

filepath1 = './result_center/center_citeseer_localep3_maxacc0.71.dat'
filepath2 = './result_noniid/k12_citeseer_avg_rate0.5_fedavg_L-hop0_maxacc0.6980000000000001.dat'
filepath5 = './result_noniid/k12_citeseer_degree_avg_rate0.5_fedavg_L-hop0_maxacc0.71.dat'

#center_citeseer_localep3_maxacc0.71.dat
#center_pubmed_localep3_maxacc0.802.dat

if __name__ == '__main__':
    args = args_parser()

    file1 = openfile(filepath1)
    file1_a = savgol_filter(file1, 33, 3)

    file2 = openfile(filepath2)
    file2_a = savgol_filter(file2, 33, 3)

    file5 = openfile(filepath5)
    file5_a = savgol_filter(file5, 33, 3)

    x = [i for i in range(500)]
    # plt.xlim(0, 346)
    # plt.ylim(0.60, 0.65)
    fig, ax = plt.subplots(1, 1)


    # plt.xlim(0, 100)
    # plt.ylim(0.4, 0.7)
    # plt.plot(np.array(file1), c='c', alpha=0.3)
    plt.plot(np.array(file1_a), c='#B8860B', label='baseline', linestyle='-')
    plt.plot(np.array(file2_a), c='#40E0D0', label='FedAvg', linestyle='-.')
    plt.plot(np.array(file5_a), c='#FF4500', label='FedGda', linestyle='-')


    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.legend(loc='best', fontsize=15)
    plt.title('citeseer', fontsize=15)
    plt.ylabel('acc', fontsize=15)
    plt.xlabel('epoch', fontsize=15)
    plt.grid()


    plt.savefig('./result_noniid/citeseer_noniid')
    plt.show()


















