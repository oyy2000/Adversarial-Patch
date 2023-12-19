import matplotlib.pyplot as plt

# {"patch_3": ["0.097", "0.003"], "patch_5": ["0.145", "0.004"], "patch_7": ["0.246", "0.011"], "patch_9": ["0.777", "0.024"], "patch_16": ["0.893", "0.152"]}
#{"patch_3": ["0.122", "0.005"], "patch_5": ["0.249", "0.016"], "patch_7": ["0.347", "0.047"], "patch_9": ["0.787", "0.062"], "patch_16": ["0.895", "0.322"]}
resnet32_fixed_results = [i * 100 for i in [0.097, 0.145, 0.246, 0.777, 0.893]]
resnet32_random_results = [i * 100 for i in [0.122, 0.249, 0.347, 0.787, 0.895]]

#{"patch_3": ["0.114", "0.003"], "patch_5": ["0.177", "0.006"], "patch_7": ["0.391", "0.011"], "patch_9": ["0.610", "0.025"], "patch_16": ["0.900", "0.316"]}
#{"patch_3": ["0.169", "0.007"], "patch_5": ["0.338", "0.018"], "patch_7": ["0.576", "0.039"], "patch_9": ["0.454", "0.062"], "patch_16": ["0.899", "0.317"]}
resnet20_fixed_results = [i * 100 for i in [0.114, 0.177, 0.391, 0.610, 0.900]]
resnet20_random_results = [i * 100 for i in [0.169, 0.338, 0.576, 0.454, 0.899]]

fixed_results = [i * 100 for i in resnet32_fixed_results]
random_results = [i * 100 for i in resnet32_random_results]
fontsize = 20
line_width=4
scatter_size=8
plt.rcParams.update({'font.size': fontsize})
plt.figure(figsize=(6, 5))

# subfigure one with resnet20 results
# plt.subplot(2, 1, 1)
# plt.title("ResNet20", fontsize=fontsize)
# plt.plot([3, 5, 7, 9, 16], resnet20_fixed_results, label="Fixed", linewidth=line_width, color="red", marker="o", markersize=scatter_size)
# plt.plot([3, 5, 7, 9, 16], resnet20_random_results, label="Random", linewidth=line_width, color="blue", marker="o", markersize=scatter_size)
# # plt.xlabel("Patch Size", fontsize=fontsize)
# plt.ylabel("ASR / %", fontsize=fontsize)
# plt.ylim(0, 101)
# plt.xticks([3, 5, 7, 9, 16])

plt.subplot(1, 1, 1)
# plt.title("ResNet32", fontsize=fontsize)
plt.plot([3, 5, 7, 9, 16], resnet32_fixed_results, label="Fixed", linewidth=line_width, color="red", marker="o", markersize=scatter_size)
plt.plot([3, 5, 7, 9, 16], resnet32_random_results, label="Random", linewidth=line_width, color="blue", marker="o", markersize=scatter_size)
plt.xlabel("Patch Size", fontsize=fontsize)
plt.ylabel("ASR / %", fontsize=fontsize)
plt.ylim(0, 101)
plt.legend(loc="lower right", fontsize=fontsize)
plt.xticks([3, 5, 7, 9, 16])

plt.tight_layout()
plt.savefig("fixed_vs_random.png")
plt.show()