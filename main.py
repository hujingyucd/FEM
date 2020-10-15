import sympy
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2 as cv
from qt_plot import Plotter



def compute_area(x1,y1,x2,y2,x3,y3):
    len1 = float(math.sqrt((x1-x2)**2+(y1-y2)**2))
    len2 = float(math.sqrt((x2-x3)**2+(y2-y3)**2))
    len3 = float(math.sqrt((x1-x3)**2+(y1-y3)**2))
    s = (len1+len2+len3)/2
    area = math.sqrt((s*(s-len1)*(s-len2)*(s-len3)))
    return area

#guarantee the input node_id1 < node_id2 < node_id3 and corresponding to coord of (xi,yi)
def stiffness_mat(node_id1,node_id2,node_id3,x1,y1,x2,y2,x3,y3,E = 1.,mu = 0., thick = 1):
    b1 = y2 - y3
    c1 = x3 - x2
    b2 = y3 - y1
    c2 = x1 - x3
    b3 = y1 - y2
    c3 = x2 - x1
    Area = compute_area(x1,y1,x2,y2,x3,y3)
    mat_B = np.array([[b1,0,b2,0,b3,0],[0,c1,0,c2,0,c3],[c1,b1,c2,b2,c3,b3]])
    mat_B = float(1/(Area*2))*mat_B
    mat_D = float(E/(1.-mu**2))*np.array([[1,mu,0],[mu,1,0],[0,0,(1-mu)/2]])
    mat_K = np.dot(np.transpose(mat_B),mat_D)
    mat_K = float(thick*Area)*np.dot(mat_K,mat_B)
    return mat_K,node_id1-1,node_id2-1,node_id3-1

#print(stiffness_mat(,2.,0.,0.,1.,0.,0.,E = 1e7,mu = float(1/3),thick=0.1)/1.0e6)
#print(stiffness_mat(,0,1,2,0,2,1,E=1e7,mu=1/3,thick=0.1)/1.0e6)

def assembly_mat(num_nodes,*args):
    dim = 2
    res = np.zeros((num_nodes*dim,num_nodes*dim))

    for element in args:
        mat_K,node0,node1,node2 = element

        res[node0 * 2, node0 * 2] += mat_K[0, 0]
        res[node0 * 2 + 1, node0 * 2] += mat_K[1, 0]
        res[node0 * 2, node0 * 2 + 1] += mat_K[0, 1]
        res[node0 * 2 + 1, node0 * 2 + 1] += mat_K[1, 1]

        res[node0 * 2, node1 * 2] += mat_K[0, 2]
        res[node0 * 2 + 1, node1 * 2] += mat_K[1, 2]
        res[node0 * 2, node1 * 2 + 1] += mat_K[0, 3]
        res[node0 * 2 + 1, node1 * 2 + 1] += mat_K[1, 3]

        res[node0 * 2, node2 * 2] += mat_K[0, 4]
        res[node0 * 2 + 1, node2 * 2] += mat_K[1, 4]
        res[node0 * 2, node2 * 2 + 1] += mat_K[0, 5]
        res[node0 * 2 + 1, node2 * 2 + 1] += mat_K[1, 5]


        res[node1 * 2, node0 * 2] += mat_K[2, 0]
        res[node1 * 2 + 1, node0 * 2] += mat_K[3, 0]
        res[node1 * 2, node0 * 2 + 1] += mat_K[2, 1]
        res[node1 * 2 + 1, node0 * 2 + 1] += mat_K[3, 1]

        res[node1 * 2, node1 * 2] += mat_K[2, 2]
        res[node1 * 2 + 1, node1 * 2] += mat_K[3, 2]
        res[node1 * 2, node1 * 2 + 1] += mat_K[2, 3]
        res[node1 * 2 + 1, node1 * 2 + 1] += mat_K[3, 3]

        res[node1 * 2, node2 * 2] += mat_K[2, 4]
        res[node1 * 2 + 1, node2 * 2] += mat_K[3, 4]
        res[node1 * 2, node2 * 2 + 1] += mat_K[2, 5]
        res[node1 * 2 + 1, node2 * 2 + 1] += mat_K[3, 5]


        res[node2 * 2, node0 * 2] += mat_K[4, 0]
        res[node2 * 2 + 1, node0 * 2] += mat_K[5, 0]
        res[node2 * 2, node0 * 2 + 1] += mat_K[4, 1]
        res[node2 * 2 + 1, node0 * 2 + 1] += mat_K[5, 1]

        res[node2 * 2, node1 * 2] += mat_K[4, 2]
        res[node2 * 2 + 1, node1 * 2] += mat_K[5, 2]
        res[node2 * 2, node1 * 2 + 1] += mat_K[4, 3]
        res[node2 * 2 + 1, node1 * 2 + 1] += mat_K[5, 3]

        res[node2 * 2, node2 * 2] += mat_K[4, 4]
        res[node2 * 2 + 1, node2 * 2] += mat_K[5, 4]
        res[node2 * 2, node2 * 2 + 1] += mat_K[4, 5]
        res[node2 * 2 + 1, node2 * 2 + 1] += mat_K[5, 5]
    return res


'''
A = stiffness_mat(1,2,3,2,1,2,0,0,1,E = 1e7,mu=1/3,thick=0.1)
B = stiffness_mat(2,3,4,2,0,0,1,0,0,E = 1e7,mu=1/3,thick=0.1)




sub_K = assembly_mat(4,A,B)[:4,:4]
vec = np.dot(np.linalg.inv(sub_K),np.array([0,50000,0,50000]))




points = [[2+vec[0],1+vec[1]],[2+vec[2],0+vec[3]],[0,1],[0,0]]

xr = []
yr = []
coord = [points[0],points[1],points[2],points[0],points[1],points[2],points[3],points[1]]
xr,yr = zip(*coord)
plt.figure()
plt.plot(xr,yr)
plt.show()
'''

'''


points = [[2,1],[2,0],[0,1],[0,0],[0,-0.8]]

xr = []
yr = []
coord = [points[0],points[1],points[2],points[0],points[1],points[2],points[3],points[1]]
xr,yr = zip(*coord)
plt.figure()
plt.plot(xr,yr)
plt.show()

'''


'''
def List2Tuple(L):
    for i in range(len(L)):
        L[i] = tuple(L[i])
    return L

def Array2List(L):
    for i in range(len(L)):
        L[i] = L[i].tolist()
    return L

P = [[200,100],[200,50],[200,0],[150,100],[150,50],[150,0],[100,100],[100,50],[100,0],[50,100],[50,50],[50,0],[0,100],[0,50],[0,0]]



#print(P)

triangles = [[1,2,4],[4,5,7],[7,8,10],[10,11,13],[2,4,5],[5,7,8],[8,10,11],[11,13,14],[2,3,5],[5,6,8],[8,9,11],[11,12,14],[3,5,6],[6,8,9],[9,11,12],[12,14,15]]
Stiff = []
for triangle in triangles:
    i = triangle[0]-1
    j = triangle[1]-1
    k = triangle[2]-1

    A = stiffness_mat(i,j,k,P[i][0],P[i][1],P[j][0],P[j][1],P[k][0],P[k][1],E = 1e7,mu=1/3,thick=0.1)
    Stiff.append(A)

print(len(Stiff))

Mat_K = assembly_mat(15,Stiff[0],Stiff[1],Stiff[2],Stiff[3],Stiff[4],Stiff[5],Stiff[6],Stiff[7],Stiff[8],Stiff[9],Stiff[10],Stiff[11],Stiff[12],Stiff[13],Stiff[14],Stiff[15])
print(Mat_K.shape)
sub_K = Mat_K[:24,:24]
print(sub_K.shape)
Power = np.array([0,50000,0,50000,0,50000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
print(Power.shape)
move = np.dot(np.linalg.inv(sub_K),Power)
print(move)


R = [[2 + move[0], 1+move[1]],[2+move[2],0.5+move[3]],[2+move[4],0+move[5]],[1.5+move[6],1+move[7]],[1.5+move[8],0.5+move[9]],[1.5+move[10],0+move[11]],[1+move[12],1+move[13]],[1+move[14],0.5+move[15]],[1+move[16],0+move[17]],[0.5+move[18],1+move[19]],[0.5+move[20],0.5+move[21]],[0.5+move[22],0+move[23]],[0,1],[0,0.5],[0,0]]
R = np.array(R)
print(R.shape)

plotter = Plotter()
Triangles = []
for i in range(0,11,3):
    Triangles.append([R[i],R[i+1],R[i+3]])
    Triangles.append([R[i+1], R[i + 1+1], R[i + 3+1]])

for i in range(1,11,3):
    print(i)
    Triangles.append([R[i],R[i+2],R[i+3]])
    Triangles.append([R[i+1], R[i + 2 + 1], R[i + 3 + 1]])


Triangles = np.array(Triangles)
plotter.draw_contours('./test.png',[('blue',Triangles[0]),('blue',Triangles[1]),('blue',Triangles[2]),('blue',Triangles[3]),('blue',Triangles[4]),('blue',Triangles[5]),('blue',Triangles[6]),('blue',Triangles[7]),('blue',Triangles[8]),('blue',Triangles[9]),('blue',Triangles[10]),('blue',Triangles[11]),('blue',Triangles[12]),('blue',Triangles[13]),('blue',Triangles[14]),('blue',Triangles[15])])

'''

'''

xr = []
yr = []
coords = [R[0],R[1],R[2],R[5],R[8],R[11],R[14],R[13],R[12],R[9],R[6],R[3],R[0]]

xr,yr = zip(*coords)
plt.figure()
plt.plot(xr,yr)
plt.show()
'''














