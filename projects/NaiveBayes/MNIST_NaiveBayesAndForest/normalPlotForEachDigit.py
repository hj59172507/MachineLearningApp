import matplotlib.pyplot as plt

#plot normal distribution graph for each digit using mean pixel values for each digit using un-threshold training value
def plotNormalForDigits(imgs, labels, classCount):
	classMat, normalPara = [],[]
	for i in range(0, classCount):
		classMat.append([])
		normalPara.append([])
	for i in range(0, len(imgs)):
		classMat[labels[i]].append(imgs[i])
	for i in range(0,classCount):
		normalPara[i] = (np.mean(classMat[i]), np.std(classMat[i]))
		x = np.linspace(normalPara[i][0]-3*normalPara[i][0],normalPara[i][0]+3*normalPara[i][0],100)
		plt.figure(i)
		plt.title(f'Normal density plot for digit {i}')
		plt.plot(x, norm.pdf(x, normalPara[i][0], normalPara[i][0]))
