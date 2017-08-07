import numpy as np
import random
import itertools
import scipy.ndimage
import scipy.misc
import matplotlib.pyplot as plt


class gameOb():
    def __init__(self,coordinates,size,color,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.color = color
        self.reward = reward
        self.name = name

class gameEnv():
    def __init__(self,partial,size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        self.bg = np.zeros([size,size])
        a,a_big,measurements,goal,hero = self.reset()


    def getFeatures(self):
        return np.array([self.objects[0].x,self.objects[0].y]) / float(self.sizeX)

    def reset(self):
        self.objects = []
        self.orientation = 0
        self.hero = gameOb(self.newPosition(0),1,[1,1,1],None,'hero')

        self.measurements = [0.0,1.0]

        self.objects.append(self.hero)
        battery = gameOb([0,0], 1, [0,0,1], 1, 'battery')
        self.battery = battery
        self.objects.append(battery)
        for i in range(1):
            bug = gameOb(self.newPosition(0),1,[0,1,0],1,'goal')
            self.objects.append(bug)
        self.goal = bug
        state,s_big = self.renderEnv()
        self.state = state
        return state,s_big,self.measurements,[self.goal.x,self.goal.y],[self.hero.x,self.hero.y]

    def moveChar(self,action):
        # 0 - up, 1 - down, 2 - left, 3 - right, 4 - 90 counter-clockwise, 5 - 90 clockwise
        hero = self.objects[0]
        blockPositions = [[-1,-1]]
        for ob in self.objects:
            if ob.name == 'block': blockPositions.append([ob.x,ob.y])
        blockPositions = np.array(blockPositions)
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if action < 4 :
            if self.orientation == 0:
               direction = action
            if self.orientation == 1:
               if action == 0: direction = 1
               elif action == 1: direction = 0
               elif action == 2: direction = 3
               elif action == 3: direction = 2
            if self.orientation == 2:
               if action == 0: direction = 3
               elif action == 1: direction = 2
               elif action == 2: direction = 0
               elif action == 3: direction = 1
            if self.orientation == 3:
               if action == 0: direction = 2
               elif action == 1: direction = 3
               elif action == 2: direction = 1
               elif action == 3: direction = 0

            if direction == 0 and hero.y >= 1 and [hero.x,hero.y - 1] not in blockPositions.tolist():
                hero.y -= 1
            if direction == 1 and hero.y <= self.sizeY-2 and [hero.x,hero.y + 1] not in blockPositions.tolist():
                hero.y += 1
            if direction == 2 and hero.x >= 1 and [hero.x - 1,hero.y] not in blockPositions.tolist():
                hero.x -= 1
            if direction == 3 and hero.x <= self.sizeX-2 and [hero.x + 1,hero.y] not in blockPositions.tolist():
                hero.x += 1
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        self.hero = hero
        return penalize

    def newPosition(self,sparcity):
        iterables = [ range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        for objectA in self.objects:
            if (objectA.x,objectA.y) in points: points.remove((objectA.x,objectA.y))
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]

    def checkGoal(self):
        if len(self.objects) > 1:
            hero = self.hero
            others = self.objects[1:]
            ended = False
            for other in others:
                if hero.x == other.x and hero.y == other.y and hero != other:
                    self.objects.remove(other)
                    ended == True
                    if other.name == 'goal':
                        goal = gameOb(self.newPosition(0),1,[0,1,0],1,'goal')
                        self.objects.append(goal)
                        self.goal = goal
                        self.measurements[0] += 1
                        return other.reward,False
                    if other.name == 'battery':
                        battery = gameOb([0,0],1,[0,0,1],1,'battery')
                        self.objects.append(battery)
                        self.battery = battery
                        self.measurements[1] = 1.0
                        return other.reward,False
            if ended == False:
                return 0.0,False
        else:
            return 0.0,False

    def renderEnv(self):
        if self.partial == True:
            padding = 2
            a = np.ones([self.sizeY+(padding*2),self.sizeX+(padding*2),3])
            a[padding:-padding,padding:-padding,:] = 0
            a[padding:-padding,padding:-padding,:] += np.dstack([self.bg,self.bg,self.bg])
        else:
            a = np.zeros([self.sizeY,self.sizeX,3])
            padding = 0
            a += np.dstack([self.bg,self.bg,self.bg])
        hero = self.objects[0]
        for item in self.objects:
            a[item.y+padding:item.y+item.size+padding,item.x+padding:item.x+item.size+padding,:] = item.color
        if self.partial == True:
            a = a[(hero.y):(hero.y+(padding*2)+hero.size),(hero.x):(hero.x+(padding*2)+hero.size),:]
        a_big = a
        a_big[self.goal.y+padding:self.goal.y+self.goal.size+padding,self.goal.x+padding:self.goal.x+self.goal.size+padding,:] = [0,1,0]
        a_big = scipy.misc.imresize(a_big,[32,32,3],interp='nearest')
        return a,a_big

    def step(self,action):
        if self.objects != []:
            penalty = self.moveChar(action)
        reward,done = self.checkGoal()
        self.measurements[1] -= 0.025
        if self.measurements[1] <= 0:
            done = True
            self.measurements[1] = 0.0
        state,s_big = self.renderEnv()
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state,self.measurements,done
        else:
            goal = None
            for ob in self.objects:
                if ob.name == 'goal':
                    goal = ob
            return state,s_big,self.measurements,[self.goal.x,self.goal.y],[self.hero.x,self.hero.y],done