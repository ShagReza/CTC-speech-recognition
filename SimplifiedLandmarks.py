V=[1,2,3,4,5,6]
VS=[1,2,3,4,5,6,30]
C=[i for i in range(7,30)]
# we may define more events than  real landmarks. This is just for coding simplification, and will not alter the results.
def SimplifiedLandmarks():
    k=0
    Events=[]
    for i in range(6):
        for j in range(24):
            Events.append([V[i],C[j]])
            Events.append([C[j]],V[i])
    States=V
    return (Events, States)