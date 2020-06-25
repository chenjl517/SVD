from svd import *

def unitTest(m,n,detail=False,specificMatrix=None):
      print("the complex matrix A:")
      if(type(specificMatrix) is np.ndarray):
            matA=specificMatrix
      else:
            matA = genRandComplexMatrix(m,n)
      print(matA,'\n\n')

      U,S,VH=svd(matA)

      # 将sigma转化为矩阵形式
      matlikeS=np.zeros([matA.shape[0],matA.shape[1]])
      for i in range(S.shape[0]):
            matlikeS[i][i]=S[i]
      S=matlikeS

      if(detail):
            print('U:\n',U,'\n')
            print('sigma:\n',S,'\n')
            print('V^\dagger:\n',VH,'\n')

      print('\n',
            '%-20s%20s%24s'%('',"----verify----",''),
            '\n')

      outputPattern='%-60s%-6s'

      print(outputPattern%
            ("V*S*VH-A=0:",np.linalg.norm(U@S@VH-matA)))

      print(outputPattern%(
            "verify U is unitary(1) -- ||U^\daggerU-I||=:",
            np.linalg.norm(getmat_H(U)@U-np.eye(U.shape[1],U.shape[1]))))
      print(outputPattern%(
            "verify U is unitary(2) -- ||UU^\dagger-I||=:",
            np.linalg.norm(U@getmat_H(U)-np.eye(U.shape[1],U.shape[1]))))

      print(outputPattern%
            ("verify V is unitary(1) -- ||V^\daggerV-I||=:",
            np.linalg.norm(VH@getmat_H(VH)-np.eye(VH.shape[0],VH.shape[1]))))
      print(outputPattern%
            ("verify V is unitary(2) -- ||VV^\dagger-I||=:",
            np.linalg.norm(getmat_H(VH)@VH-np.eye(VH.shape[0],VH.shape[1]))))


      print(outputPattern%
            ("verify singular is positive -- no negative numbers:",not np.any(S<0)))

      print('\n',
            '%-18s%20s%24s'%('',"----end----",''),
            '\n')
      return U,S,VH

# matA=np.array([[4,11,14],[8,7,-2]])
np.set_printoptions(precision=3)
U,S,VH=unitTest(4,3,True)