import numpy as np
# 厄米共轭
def getmat_H(mat):
    return mat.conj().T

def genRandComplexMatrix(m,n):
    return np.random.randn(m, n) + 1j*np.random.randn(m, n)

def constructOrthBasis(mat):
    rows=mat.shape[0]
    cols=mat.shape[1]
    if(cols>rows):
        raise Exception('Invalid Matrix')
    else:
        while(cols<rows):
            vec=np.random.randn(rows)+1j*np.random.randn(rows)
            for i in range(cols):
                vec-=np.vdot(mat[:,i],vec)*mat[:,i]/np.linalg.norm(mat[:,i])
            vec*=1/np.linalg.norm(vec)
            mat=np.column_stack((mat,vec))
            cols=cols+1
        return mat

def svd(matA):
    # 找出较小的矩阵
    vfirstFlag=False
    if (matA.shape[0] > matA.shape[1]):
        vfirstFlag=True
        firstMatrix = getmat_H(matA)
        secondMatrix = matA
    else:
        firstMatrix = matA
        secondMatrix = getmat_H(matA)

    # 计算较小的矩阵的特征值和特征向量，筛选奇异值
    tmpMatrix = np.matmul(firstMatrix, secondMatrix)
    eigenvalue,firstEigenvector=np.linalg.eigh(-tmpMatrix)
    eigenvalue=-eigenvalue #将特征值从大到小排序

    # 找出非零的特征值以及对应
    maxEigenvaluePos=np.where(eigenvalue>1e-15)[-1][-1]+1 #特征值都是实数，不用加abs
    eigenvalue=eigenvalue[:maxEigenvaluePos]
    firstEigenvector=firstEigenvector[:,:maxEigenvaluePos]
    singularValue=eigenvalue**0.5

    # 根据已有特征值计算令一边,numpy eigh计算得到的特征向量是列向量，无需转置
    secondEigenvector=np.matmul(secondMatrix,firstEigenvector)
    secondEigenvector/=singularValue

    firstEigenvector = constructOrthBasis(firstEigenvector)
    secondEigenvector = constructOrthBasis(secondEigenvector)

    #return U,sigma,VH
    if(vfirstFlag):
        return secondEigenvector,singularValue,getmat_H(firstEigenvector)
    else:
        return firstEigenvector,singularValue,getmat_H(secondEigenvector)