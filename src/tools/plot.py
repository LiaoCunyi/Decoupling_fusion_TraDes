import numpy as np
importances = np.array([3.73,32.03,50.69,29.75,54.95,
                        35.74,51.13,17.36,41.48,
                        94.22,100,12.46,95.04,28.31,84.62])
importances = importances/100*0.352
print(np.shape(importances))
import matplotlib.pyplot as plt
# # 正常显示中文标签
# plt.rcParams["font.family"] = 'Arial Unicode MS'
# # 用来正常显示负号
# plt.rcParams['axes.unicode_minus']=False

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# plt.rcParams['axes.labelsize']=16
plt.rcParams['xtick.labelsize']=14
plt.rcParams['ytick.labelsize']=14
# plt.rcParams['legend.fontsize']=12
plt.rcParams['figure.figsize']=[10,6]
# 使用样式
plt.style.use("ggplot")
label = ["$f$","$C_t$","$P_t$",r"$\beta_v$","$h_t$",
         "$d$","$h_{ra}$","$h_{rb}$",r"$\Delta h$",r"$\Delta h_v$","$l$","$C_r$",
         r"$\cos \alpha$",r"$\cos\beta$",r"$\cos \gamma$",]

# price = [39.5,39.9,45.4,38.9,33.34]
# for x,y in enumerate(importances):
#     print(x,y)
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
y = importances
fig = plt.figure()

# 生成第一个子图在1行2列第一列位置
ax1 = fig.add_subplot(111)

# 添加轴标签
ax1.set_xlabel('物理特征')
ax1.set_ylabel('重要性得分')
# 绘图并设置柱子宽度0.5
ax1.bar(x, y, width=0.8,tick_label=label,color='blue',alpha=0.8)

# 为每个条形图添加数值标签
# for x,y in enumerate(importances):
#     ax1.text(x,y+1,y,ha='center',fontsize=14)

from matplotlib.pyplot import savefig
plt.savefig('C:\\Users\ACTL\Desktop\\1.jpg')
plt.show()