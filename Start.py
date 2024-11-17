import math
from collections.abc import Iterable

import hysteresis as hys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from scipy.integrate import quad
from scipy.optimize import curve_fit
from sklearn import linear_model

import DataModel as DM


def circleArea(diameter: float) -> float():
    return diameter *diameter * math.pi

n1, n2 = 400, 500
dSteel = pd.Series([3.15E-3, 3.15E-3, 3.15E-3])
tIron = pd.Series([0.71E-3,0.70E-3,0.69E-3])
wIron = pd.Series([4.51E-3, 4.50E-3, 4.46E-3, 4.53E-3, 4.56E-3])
dCuNi = pd.Series([4.99E-3,4.99E-3,4.99E-3])
l1 = pd.Series([4.492E-2,4.500E-2,4.490E-2,4.482E-2,4.472E-2])
#a2 = 2.216E-5
a2 = 3.38E-5
r1 = 2.047
r2 = 9.943E3
c = 494.17E-9
u0 = 4 * np.pi * 1E-7

aSteel = circleArea(dSteel.mean()/2)
aIron = tIron.mean() * wIron.mean()
aCuNi = circleArea(dCuNi.mean()/2)

print(aSteel)
print(aIron)
print(aCuNi)

jm3 = r'$Jm^{-3}$'
muapp = r'$\mu_{dif} / Hm^{-1}$'
mur = r'$\mu_{r}$'

data = pd.read_csv('Hysteresis_Collated.csv', names=['vxAir','vyAir','vxSteel','vySteel','vxIron','vyIron','vxCuNi40','vyCuNi40','vxCuNi7','vyCuNi7'])
#guessParams = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#good params for iron and steel
#guessParams = [0.5,0.0001,400,0.5,0.0001,400,0.5,0.0001,400,0.5,0.0001,400]
guessParams = [1,0.000001,0,1,0.000001,0,1,0.000001,0,1,0.000001,0,1,0.000001,0,1,0.000001,0]
maxfevs = 10000000

def first_3_dp(num):
    if num > 0.99E3 or num < 1E-2:
        return '{:.3e}'.format(num)
    else:
        return '{:.3}'.format(num)

def smooth(x , y , dx):
    _x = np.array(x)
    _y = np.array(y)
    _dx = float(dx)

    smoothX = []
    smoothY = []
    while len(_x) > 0:
        hitXIndices = np.where((_x < _x[0] + _dx) & (_x > _x[0] - _dx))
        smoothX.append(np.average(_x[hitXIndices]))
        smoothY.append(np.average(_y[hitXIndices]))
        _x = np.delete(_x, hitXIndices)
        _y = np.delete(_y, hitXIndices)
    
    return smoothX, smoothY

def objective(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r):
    #return z+a*x +b*x**2+c*x**3+d*x**4+e*x**5+f*x**6+g*x**7+h*x**8+i*x**9+j*x**10+k*x**11+l*x**12+m*x**13+n*x**14+o*x**15+p*x**16+q*x**17+r*x**18+s*x**19+t*x**20+u*x**21+v*x**22+w*x**23+y*x**24
    return a * np.tanh( b * (x+c)) + d* np.tanh( e*(x+f))+ g*np.tanh( h *(x+i)) + j *np.tanh(k*(x+l))+m *np.tanh(n*(x+o))+p *np.tanh(q*(x+r))

def getOptimisedObjective(popt, xValues):
    return objective(xValues, *popt)

def fitHyst(dataFrame, dx):
    middleH, middleB =smooth(dataFrame["H"].values, dataFrame["B"].values, dx)
    try:
        poptMiddle, _ = curve_fit(objective, middleH, middleB, p0=guessParams, maxfev = maxfevs)
    except ValueError as err:
        return

    linspaceH = np.linspace(min(middleH), max(middleH), 1000)
    middleBCurve = getOptimisedObjective(poptMiddle, linspaceH)

    #plots vertical averages and their fit
    #plt.scatter(middleH, middleB)
    #plt.plot(linspaceH, middleBCurve)

    bValues = dataFrame['B'].values
    hValues = dataFrame['H'].values
    aboveMiddleBValues = []
    aboveMiddleHValues = []
    belowMiddleBValues = []
    belowMiddleHValues = []

    for i in range(0, len(hValues) -1):
        if bValues[i] > getOptimisedObjective(poptMiddle, hValues[i]):
            aboveMiddleBValues.append(bValues[i])
            aboveMiddleHValues.append(hValues[i])
        else:
            belowMiddleBValues.append(bValues[i])
            belowMiddleHValues.append(hValues[i])

    #plots separation of top and bottom datasets
    #plt.scatter(aboveMiddleHValues, aboveMiddleBValues, color='green')
    #plt.scatter(belowMiddleHValues, belowMiddleBValues, color= 'red')

    aboveMiddleHValuesSmooth, aboveMiddleBValuesSmooth = smooth(aboveMiddleHValues, aboveMiddleBValues, dx)
    belowMiddleHValuesSmooth, belowMiddleBValuesSmooth = smooth(belowMiddleHValues, belowMiddleBValues, dx)

    #plots top and bottom averages
    #plt.scatter(aboveMiddleHValuesSmooth, aboveMiddleBValuesSmooth, color='purple')
    #plt.scatter(belowMiddleHValuesSmooth, belowMiddleBValuesSmooth, color='pink')

    poptTop, _ = curve_fit(objective, aboveMiddleHValuesSmooth, aboveMiddleBValuesSmooth, p0=guessParams, maxfev = maxfevs)
    poptBottom, _ = curve_fit(objective, belowMiddleHValuesSmooth, belowMiddleBValuesSmooth, p0=guessParams, maxfev = maxfevs)

    #plots fits to top and bottom averages 
    plt.plot(linspaceH, getOptimisedObjective(poptTop, linspaceH), color='orange', label='Upper fit')
    plt.plot(linspaceH, getOptimisedObjective(poptBottom, linspaceH), color ='green', label='Lower fit')
    plt.xlabel('$H / Am^{-1}$')
    plt.ylabel('B / T')
    poptCombined = poptTop - poptBottom
    #areaHysteresis = quad(objective, min(belowMiddleHValues), max(belowMiddleHValues), args=(poptCombined[0],poptCombined[1],poptCombined[2],poptCombined[3],poptCombined[4],poptCombined[5],poptCombined[6],poptCombined[7],poptCombined[8],poptCombined[9],poptCombined[10],poptCombined[11],poptCombined[12],poptCombined[13],poptCombined[14],poptCombined[15],poptCombined[16],poptCombined[17],poptCombined[18],poptCombined[19],poptCombined[20],poptCombined[21],poptCombined[22],poptCombined[23],poptCombined[24]))
    #areaHysteresis = quad(objective, min(belowMiddleHValues), max(belowMiddleHValues), args=(poptCombined[0],poptCombined[1],poptCombined[2],poptCombined[3],poptCombined[4],poptCombined[5],poptCombined[6],poptCombined[7],poptCombined[8],poptCombined[9],poptCombined[10],poptCombined[11]))
    return [poptTop, poptBottom, min(belowMiddleHValues), max(belowMiddleHValues)]

def integralObj(min, max, args):
    return quad(objective, min, max, args=(args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8],args[9],args[10],args[11],args[12],args[13],args[14],args[15],args[16],args[17]))


def findLoopArea(curveFits):
    return integralObj(curveFits[2], curveFits[3], curveFits[0])[0]-integralObj(curveFits[2], curveFits[3], curveFits[1])[0]

def findGradients(curveFits):
    samples = 100000
    xrange = np.linspace(curveFits[2],curveFits[3], samples)
    dy = [[],[]]
    for j in range(0,2):
        for i in range(0,samples-1):
            dy[j].append(getOptimisedObjective(curveFits[j], xrange[i+1])-getOptimisedObjective(curveFits[j], xrange[i]))
    return [dy/(xrange[1]-xrange[0]), xrange[0:samples-1]]
    # upperFitValues = getOptimisedObjective(curveFits[0], xrange)
    # lowerFitValues = getOptimisedObjective(curveFits[1], xrange)
    # return [[upperFitValues / xrange,lowerFitValues / xrange], xrange]
    

HBAir = pd.DataFrame()
HBAir["H"], HBAir["B"] = data['vxAir']*n1/(l1.mean() * r1), data['vyAir']*r2*c/(n2*a2)
HBAir.plot.scatter(x="H", y="B", s=0.5, color='black', marker="+", label='Data')
airFits = fitHyst(HBAir, 25)
airArea = findLoopArea(airFits)
print(findLoopArea(airFits))

#Performing linear regression on the data for Air
airRegResult = sm.ols(formula="B ~ H", data = HBAir).fit()
print("Linear regression results for Air B vs H plot")
print(airRegResult.summary())
print(airRegResult.params[1])
airRegC = airRegResult.params[0]
airRegK = airRegResult.params[1]
modelH = np.linspace(HBAir["H"].min(),HBAir["H"].max(),1000)
modelB = airRegResult.params[0]+modelH*airRegResult.params[1]
plt.plot(modelH,modelB, color='red', label='Linear regression fit')
plt.legend()
plt.text(-5000, -0.02, f'B = {first_3_dp(airRegResult.params[1])}H+{first_3_dp(airRegResult.params[0])}\n Area = {first_3_dp(airArea)} {jm3} \n {mur} = {first_3_dp(airRegResult.params[1]/u0)} (by Linear Regression)')
plt.title('Hysteresis of Air (B vs H)')
plt.grid(True)
plt.savefig("HBAir.pdf")


plt.figure("Air μ")
airGradients = findGradients(airFits)
plt.plot(airGradients[1], airGradients[0][0]/u0, color='orange', label='Upper fit')
plt.plot(airGradients[1], airGradients[0][1]/u0, color='green', label='Lower fit')
plt.xlabel('$H / Am^{-1}$')
plt.ylabel('$\mu_{dif}$')
plt.title('Differential Permeability of Air During Hysteresis')
plt.legend()
plt.grid(True)
plt.savefig("MuAir.pdf")

HBSteel = pd.DataFrame()
HBSteel["H"], HBSteel["B"] = data['vxSteel']*n1/(l1.mean() * r1), (data['vySteel']*r2*c/(n2*aSteel)+(aSteel-a2)*(airRegC+airRegK*(n2*data['vxSteel'])/(l1.mean()*r1))/aSteel)
HBSteel.plot.scatter(x="H", y="B", s=0.5, color='black', marker="+", label='Data')
steelFits = fitHyst(HBSteel, 25)
steelArea = findLoopArea(steelFits)
print(steelArea)
plt.legend()
plt.title('Hysteresis of Mild Steel (B vs H)')
plt.grid(True)
plt.text(7500, -1.0, f'Area = {first_3_dp(steelArea)} {jm3}')
plt.savefig("HBSteel.pdf")

plt.figure("Steel mu")
steelGradients = findGradients(steelFits)
plt.plot(steelGradients[1], steelGradients[0][0]/u0, color='orange', label='Upper fit')
plt.plot(steelGradients[1], steelGradients[0][1]/u0, color='green', label="Lower fit")
plt.xlabel('$H / Am^{-1}$')
plt.ylabel('$\mu_{dif}$')
plt.title('Differential Permeability of Mild Steel During Hysteresis')
plt.legend()
plt.grid(True)
plt.savefig("MuSteel.pdf")

HBIron = pd.DataFrame()
HBIron["H"], HBIron["B"] = data['vxIron']*n1/(l1.mean() * r1), (data['vyIron']*r2*c/(n2*aIron)+(aIron-a2)*(airRegC+airRegK*(n2*data['vxIron'])/(l1.mean()*r1))/aIron)
HBIron.plot.scatter(x="H", y="B", s=0.5, color='black', marker="+", label='Data')
ironFits = fitHyst(HBIron, 25)
ironArea = findLoopArea(ironFits)
print(ironArea)
plt.legend()
plt.title('Hysteresis of Transformer Iron (B vs H)')
plt.grid(True)
plt.text(5000, -0.5, f'Area = {first_3_dp(ironArea)} {jm3}')
plt.savefig("HBIron.pdf")

plt.figure("Iron mu")
ironGradients = findGradients(ironFits)
plt.plot(ironGradients[1], ironGradients[0][0]/u0, color='orange', label='Upper fit')
plt.plot(ironGradients[1], ironGradients[0][1]/u0, color='green', label = 'Lower fit')
plt.xlabel('$H / Am^{-1}$')
plt.ylabel('$\mu_{dif}$')
plt.title('Differential Permeability of Transformer Iron During Hysteresis')
plt.legend()
plt.grid(True)
plt.savefig("MuIron.pdf")

HBCuNi40 = pd.DataFrame()
HBCuNi40["H"], HBCuNi40["B"] = data['vxCuNi40']*n1/(l1.mean() * r1), (data['vyCuNi40']*r2*c/(n2*aCuNi)+(aCuNi-a2)*(airRegC+airRegK*(n2*data['vxCuNi40'])/(l1.mean()*r1))/aCuNi)
HBCuNi40.plot.scatter(x="H", y="B", s=0.5, color='black', marker="+", label='Data')
cuNi40Fits = fitHyst(HBCuNi40, 25)
cuNi40Area = findLoopArea(cuNi40Fits)
print(cuNi40Area)

cuNi40RegResult = sm.ols(formula="B ~ H", data = HBCuNi40).fit()
print("Linear regression results for CuNi40 B vs H plot")
print(cuNi40RegResult.summary())
cuNi40RegC = cuNi40RegResult.params[0]
cuNi40RegK = cuNi40RegResult.params[1]
modelH = np.linspace(HBCuNi40["H"].min(),HBCuNi40["H"].max(),1000)
modelB = cuNi40RegResult.params[0]+modelH*cuNi40RegResult.params[1]
plt.plot(modelH,modelB, color='red', label='Linear regression fit')

plt.legend()
plt.title(r'Hysteresis of CuNi at T$\approx$ 316K')
plt.grid(True)
plt.text(-5000,-0.025,f'B={first_3_dp(cuNi40RegResult.params[1])}H+{first_3_dp(cuNi40RegResult.params[0])}\n Area = {first_3_dp(cuNi40Area)} {jm3} \n {mur} = {first_3_dp(cuNi40RegResult.params[1]/u0)} (by Linear Regression)')
plt.savefig("HBCuNi40.pdf")

plt.figure("CuNi40 mu")
cuNi40Gradients = findGradients(cuNi40Fits)
plt.plot(cuNi40Gradients[1], cuNi40Gradients[0][0]/u0, color ='orange', label='Upper fit')
plt.plot(cuNi40Gradients[1], cuNi40Gradients[0][1]/u0, color='green', label='Lower fit')
plt.xlabel('$H / Am^{-1}$')
plt.ylabel('$\mu_{dif} $')
plt.title(r'Differential Permeability of CuNi at T$\approx$ 316K')
plt.legend()
plt.grid(True)
plt.savefig("MuCuNi40.pdf")

HBCuNi7 = pd.DataFrame()
HBCuNi7["H"], HBCuNi7["B"] = data['vxCuNi7']*n1/(l1.mean() * r1), (data['vyCuNi7']*r2*c/(n2*aCuNi)+(aCuNi-a2)*(airRegC+airRegK*(n2*data['vxCuNi7'])/(l1.mean()*r1))/aCuNi)
HBCuNi7.plot.scatter(x="H", y="B", s=0.5, color='black', marker="+", label='Data')
cuNi7Fits = fitHyst(HBCuNi7, 25)
cuNi7Area = findLoopArea(cuNi7Fits)
print(cuNi7Area)
plt.legend()
plt.title(r'Hysteresis of CuNi at T$\approx$ 280K')
plt.grid(True)
plt.text(5000, -0.04, f'Area = {first_3_dp(cuNi7Area)} {jm3}')
plt.savefig("HBCuNi7.pdf")

plt.figure("CuNi7 mu")
cuNi7Gradients = findGradients(cuNi7Fits)
plt.plot(cuNi7Gradients[1], cuNi7Gradients[0][0]/u0, color='orange', label='Upper fit')
plt.plot(cuNi7Gradients[1], cuNi7Gradients[0][1]/u0, color='green', label='Lower fit')
plt.xlabel('$H / Am^{-1}$')
plt.ylabel('$\mu_{dif}$')
plt.title(r'Differential Permeability of CuNi at T$\approx$ 280K')
plt.legend()
plt.grid(True)
plt.savefig("MuCuNi7.pdf")

plt.figure("a")
plt.scatter(data['vxSteel'].values, data['vySteel'].values)
# plt.figure("a")
# plt.scatter(data['vxIron'].values, data['vyIron'].values)
#plt.figure("a")
#plt.scatter(data['vxCuNi40'].values, data['vyCuNi40'].values)
# plt.figure("a")
# plt.scatter(data['vxCuNi7'].values, data['vyCuNi7'].values)

plt.show()

