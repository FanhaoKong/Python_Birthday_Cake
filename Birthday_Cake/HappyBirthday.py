from scipy.integrate import odeint
import numpy as np
from mayavi import mlab

def cakebase(radius,height,z):
    dphi=np.pi/50
    dr,dz=1.0/100,1.0/100
    R,phi=np.mgrid[0:radius:dr,-np.pi-dphi:np.pi+dphi:dphi]
    phi2,Z2=np.mgrid[-np.pi-dphi:np.pi+dphi:dphi,z:z+height:dz]
    X=R*np.cos(phi)
    Y=R*np.sin(phi)
    Z=np.sin(np.sqrt(X**2+Y**2))+z
    X2=radius*np.cos(phi2)
    Y2=radius*np.sin(phi2)
    #bottom
    mlab.mesh(X,Y,Z,colormap='Pastel1')
    #cover
    mlab.mesh(X,Y,Z+height,colormap='Pastel1')
    mlab.mesh(X2,Y2,Z2,colormap='pink')

def boundary(radius,z,wide,height,period):
    dphi=np.pi/2000
    phai=np.arange(-np.pi-dphi,np.pi+dphi,dphi)
    R=np.zeros(len(phai),dtype='float64')
    R2=np.zeros(len(phai),dtype='float64')
    zentry=np.zeros(len(phai),dtype='float64')
    zentry2=np.zeros(len(phai),dtype='float64')
    for i in range(len(zentry)):
        R[i]=radius+height*np.sin(period*phai[i])
        R2[i]=radius+height*np.sin(-period*phai[i])
        zentry[i]=z+wide*np.sin(period/20*phai[i])+height*np.sin(2*period*phai[i]+np.pi/2)
        zentry2[i]=z+wide*np.sin(period/20*phai[i])+height*np.sin(-2*period*phai[i]-np.pi/2)
    distribution=np.array([R,phai,zentry]).T
    distribution2=np.array([R2,phai,zentry2]).T
    position=np.zeros((len(phai),3),dtype='float64')
    position2=np.zeros((len(phai),3),dtype='float64')
    for i in range(len(position)):
        position[i][0]=distribution[i][0]*np.cos(distribution[i][1])
        position[i][1]=distribution[i][0]*np.sin(distribution[i][1])
        position[i][2]=distribution[i][2]
        position2[i][0]=distribution2[i][0]*np.cos(distribution2[i][1])
        position2[i][1]=distribution2[i][0]*np.sin(distribution2[i][1])
        position2[i][2]=distribution2[i][2]
    position=position.T
    position2=position2.T
    mlab.plot3d(position[0],position[1],position[2],position[2], \
                tube_sides=20,tube_radius=1,colormap='pink')
    mlab.plot3d(position2[0],position2[1],position2[2],position2[2], \
                tube_sides=20,tube_radius=1,colormap='pink')
        
def boundary2(radius,z,wide,height,period):
    dphi=np.pi/2000
    phai=np.arange(-np.pi-dphi,np.pi+dphi,dphi)
    R=np.zeros(len(phai),dtype='float64')
    R2=np.zeros(len(phai),dtype='float64')
    zentry=np.zeros(len(phai),dtype='float64')
    zentry2=np.zeros(len(phai),dtype='float64')
    for i in range(len(zentry)):
        R[i]=radius+height*np.sin(period*phai[i])
        R2[i]=radius+height*np.sin(-period*phai[i])
        zentry[i]=z+wide*np.sin(period/20*phai[i]+np.pi)+height*np.sin(2*period*phai[i]+np.pi/2)
        zentry2[i]=z+wide*np.sin(period/20*phai[i]+np.pi)+height*np.sin(-2*period*phai[i]-np.pi/2)
    distribution=np.array([R,phai,zentry]).T
    distribution2=np.array([R2,phai,zentry2]).T
    position=np.zeros((len(phai),3),dtype='float64')
    position2=np.zeros((len(phai),3),dtype='float64')
    for i in range(len(position)):
        position[i][0]=distribution[i][0]*np.cos(distribution[i][1])
        position[i][1]=distribution[i][0]*np.sin(distribution[i][1])
        position[i][2]=distribution[i][2]
        position2[i][0]=distribution2[i][0]*np.cos(distribution2[i][1])
        position2[i][1]=distribution2[i][0]*np.sin(distribution2[i][1])
        position2[i][2]=distribution2[i][2]
    position=position.T
    position2=position2.T
    mlab.plot3d(position[0],position[1],position[2],position[2], \
                tube_sides=20,tube_radius=1,colormap='pink')
    mlab.plot3d(position2[0],position2[1],position2[2],position2[2], \
                tube_sides=20,tube_radius=1,colormap='pink')
        
def lorenz(w,t,sigma,rou,beta):
    #w:position 
    #sigma, rou, beta are the three parameters in the equation
    x,y,z=w.tolist()
    return sigma*(y-x),x*(rou-z)-y,x*y-beta*z

def drawlorenz(xc,yc,zc,thetax,thetay,thetaz,order):
    matx=np.matrix([[1,0,0],[0,np.cos(thetax),-np.sin(thetax)],[0,np.sin(thetax),np.cos(thetax)]], \
                   dtype='float64')
    maty=np.matrix([[np.cos(thetay),0,-np.sin(thetay)],[0,1,0],[np.sin(thetay),0,np.cos(thetay)]], \
                   dtype='float64')
    matz=np.matrix([[np.cos(thetaz),-np.sin(thetaz),0],[np.sin(thetaz),np.cos(thetaz),0],[0,0,1]], \
                   dtype='float64')
    
    time=np.linspace(0,50,10000)
    track1=odeint(lorenz,(0.0,1.0,0.0),time,args=(10.0,28.0,3.0))
    track1=np.mat(track1)*matx*maty*matz
    track2=odeint(lorenz,(0.0,1.01,0.0),time,args=(10.0,28.0,3.0))
    track2=np.mat(track2)*matx*maty*matz
    track3=odeint(lorenz,(0.0,1.02,0.0),time,args=(10.0,28.0,3.0))
    track2=np.mat(track2)*matx*maty*matz
    track1,track2,track3=np.array(track1),np.array(track2),np.array(track3)
    for i in range(len(track1)):
        track1[i]=track1[i]+np.array([xc,yc,zc])
        track2[i]=track2[i]+np.array([xc,yc,zc])
        track2[i]=track3[i]+np.array([xc,yc,zc])
    X,Y,Z=track1.T
    X2,Y2,Z2=track2.T
    X3,Y3,Z3=track3.T
    if order==0:
        mlab.plot3d(X,Y,Z,time,tube_radius=0.3,colormap='spring')
        mlab.plot3d(X2,Y2,Z2,time,tube_radius=0.3,colormap='spring')
        #mlab.plot3d(X3,Y3,Z3,time,tube_radius=0.15,colormap='Greens')
    if order==1:
        mlab.plot3d(X,Y,Z,time,tube_radius=0.3,colormap='Wistia')
        mlab.plot3d(X2,Y2,Z2,time,tube_radius=0.3,colormap='Wistia')
        #mlab.plot3d(X3,Y3,Z3,time,tube_radius=0.15,colormap='Wistia')

def LORENZ(radius,z,part):
    angle=2*np.pi/part
    position=np.zeros((part,3),dtype='float64')
    for i in range(len(position)):
        position[i][0]=radius*np.cos(angle*i)
        position[i][1]=radius*np.sin(angle*i)
        position[i][2]=z
        drawlorenz(position[i][0],position[i][1],position[i][2],0,0,i*angle+(np.pi/4),i%2)
    return position

def fruit(radius,center,mode):
    dtheta,dphi=np.pi/180,np.pi/180
    theta,phi=np.mgrid[-np.pi-dtheta:np.pi+dtheta:dtheta, \
                     -np.pi-dphi:np.pi+dphi:dphi]
    x=radius*(0.8+0.2*np.sin(12*phi))*np.sin(theta)*np.cos(phi)+center[0]
    y=radius*(0.8+0.2*np.sin(12*phi))*np.sin(theta)*np.sin(phi)+center[1]
    z=radius*np.cos(theta)+center[2]
    if mode==1:
        mlab.mesh(x,y,z)
    if mode==2:
        mlab.mesh(x,y,z,colormap='winter')

def controlfruit(radius,part,R,z):
    for i in range(part):
        center=np.array([radius*np.cos((np.pi*2/part)*i),radius*np.sin((np.pi*2/part)*i),z])
        if i%2==0:
            fruit(R,center,1)
        else:
            fruit(R,center,2)

def decoration(radius,center):
    #big flower in the middle
    dtheta,dphi=np.pi/180,np.pi/180
    theta,phi=np.mgrid[-np.pi-dtheta:np.pi+dtheta:dtheta, \
                     -np.pi-dphi:np.pi+dphi:dphi]
    x=radius*(0.9+0.1*np.sin(20*phi))*np.sin(theta)*np.cos(phi)+center[0]
    y=radius*(0.9+0.1*np.sin(20*phi))*np.sin(theta)*np.sin(phi)+center[1]
    z=radius*0.4*(np.sin(16*theta))*np.cos(theta)+center[2]
    mlab.mesh(x,y,z,colormap='pink')
    
    
def sineboundary(radius,z,height,period):
    dphi=np.pi/360
    phai=np.arange(-np.pi-dphi,np.pi+dphi,dphi)
    R=np.zeros(len(phai),dtype='float64')
    zentry=np.zeros(len(phai),dtype='float64')
    for i in range(len(zentry)):
        R[i]=radius+height*np.sin(period*phai[i])
        zentry[i]=z+height*np.sin(period*phai[i])
    distribution=np.array([R,phai,zentry]).T
    position=np.zeros((len(phai),3),dtype='float64')
    for i in range(len(position)):
        position[i][0]=distribution[i][0]*np.cos(distribution[i][1])
        position[i][1]=distribution[i][0]*np.sin(distribution[i][1])
        position[i][2]=distribution[i][2]
    position=position.T
    mlab.plot3d(position[0],position[1],position[2],position[2], \
                tube_sides=20,tube_radius=2,colormap='autumn')
        
def multiflower(size,center,thetax,thetay,thetaz):
    dtheta=np.pi/360
    theta=np.arange(-np.pi-dtheta,np.pi+dtheta,dtheta)
    R=size*np.cos(6*theta)
    R2=0.7*size*np.cos(6*theta)
    X=R*np.cos(theta)
    Y=R*np.sin(theta)
    Z=np.zeros(len(X))
    X2=R2*np.cos(theta)
    Y2=R2*np.sin(theta)
    Z2=np.full(len(X),2.0)
    matx=np.matrix([[1,0,0],[0,np.cos(thetax),-np.sin(thetax)],[0,np.sin(thetax),np.cos(thetax)]], \
                   dtype='float64')
    maty=np.matrix([[np.cos(thetay),0,-np.sin(thetay)],[0,1,0],[np.sin(thetay),0,np.cos(thetay)]], \
                   dtype='float64')
    matz=np.matrix([[np.cos(thetaz),-np.sin(thetaz),0],[np.sin(thetaz),np.cos(thetaz),0],[0,0,1]], \
                   dtype='float64')
    position=np.mat([X,Y,Z]).T
    position=position*matx*maty*matz
    position=np.array(position)
    position2=np.mat([X2,Y2,Z2]).T
    position2=position2*matx*maty*matz
    position2=np.array(position2)
    for i in range(len(position)):
        position[i]=position[i]+center
        position2[i]=position2[i]+center
    position=position.transpose()
    position2=position2.transpose()
    X,Y,Z=position[0],position[1],position[2]
    X2,Y2,Z2=position2[0],position2[1],position2[2]
    mlab.plot3d(X,Y,Z,tube_radius=1,tube_sides=20)
    mlab.plot3d(X2,Y2,Z2,tube_radius=1,tube_sides=20)
    
def controlflower(radius,size,number,z):
    for i in range(number):
        centerx=radius*np.cos(i*np.pi*2/number+np.pi/number)
        centery=radius*np.sin(i*np.pi*2/number+np.pi/number)
        centerz=z
        thetax=np.pi/2;thetay=0;thetaz=np.pi/2-i*np.pi*2/number-np.pi/number
        multiflower(size,np.array([centerx,centery,centerz]),thetax,thetay,thetaz)
    
print("\nLoading...\n")
try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
if len(engine.scenes) == 0:
    engine.new_scene()

scene = engine.scenes[0]
scene.scene.background = (0.0, 0.0, 0.0)

cakebase(200,80,80)
cakebase(240,80,0)
#def boundary(radius,z,wide,height,period):
boundary(244,40,30,4,100)
boundary(204,120,30,4,100)
boundary2(244,40,30,4,100)
boundary2(204,120,30,4,100)
sineboundary(230,100,60,20)
sineboundary(220,90,60,20)
sineboundary(210,80,60,20)
sineboundary(200,70,60,20)
LORENZ(200,160,17)
controlfruit(130,16,30,160)
controlflower(240,20,10,40)
controlflower(200,20,10,120)
decoration(100,np.array([0,0,160]))
mlab.show()
a=input('\n    Happy Birthday to you!\n')


