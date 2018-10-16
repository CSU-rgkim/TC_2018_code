#################################### Import ###############################
from collections import defaultdict
from heapq import *
import xlrd
from copy import deepcopy
from random import randint,random,seed
import math
import numpy
import random
from sklearn.ensemble import RandomForestRegressor
import timeit

################################ defn. Variables #############################

'''
defn. of last move and best move:
last_move[0]: core = 0, link = 1
last_move[1]: swapped core index 1
last_move[2]: swapped core index 2
last_move[3]: removed link source
last_move[4]: removed link destination
last_move[5]: added link source
last_move[6]: added link destination
'''
#################### Creating 64 nodes ##########################
def create_nodes():
    nodes = []
    # nodes_orig=[]
    mc = 8
    cpu = 0
    gpu = 24
    for i in range(0, 64):
        nodes.append(9999)
    for i in range(0, 4):
        nodes[i * 16 + 5] = cpu;
        nodes[i * 16 + 6] = cpu + 1;
        cpu = cpu + 2
        nodes[i * 16 + 0] = mc;
        nodes[i * 16 + 3] = mc + 1;
        nodes[i * 16 + 12] = mc + 2;
        nodes[i * 16 + 15] = mc + 3;
        mc = mc + 4
    for i in range(0, 64):
        if nodes[i] == 9999:
            nodes[i] = gpu;
            gpu = gpu + 1
            # nodes_orig.append(i)
    return deepcopy(nodes)

########################## Creating 4*4*4 mesh link connectivity #############################
def create_mesh():
    links = numpy.zeros(shape=(64, 64))
    for i in range(0, 64):
        for j in range(0, 64):
            zs = int(i / 16)
            ts = i % 16
            ys = int(ts / 4)
            xs = ts % 4
            zd = int(j / 16)
            td = j % 16
            yd = int(td / 4)
            xd = td % 4
            if (abs(zs - zd) == 1) and (xs == xd) and (ys == yd):  # z-links
                links[i][j] = 1
            elif (zs == zd) and (ys == yd) and (abs(xd - xs) == 1):  # x-links
                links[i][j] = 1
            elif (zs == zd) and (xs == xd) and (abs(yd - ys) == 1):  # y-links
                links[i][j] = 1
    return deepcopy(links)

################################### Load traffic ####################################
def load_traffic():
    with open("input_traffic.txt") as f:
        content = f.readlines()
    # also remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    traffic = []
    injection=[]
    for i in range(0, len(content)):
        temp = content[i].split()
        t = []
        injection1=0.0
        for j in range(0, len(temp)):
            t.append(float(temp[j]))
            injection1=injection1+float(temp[j])
        traffic.append(t)
        injection.append(injection1)
        
    return traffic, injection

################################# Dijkstra shortest path algorithm ###############################
def dijkstra(edges, f, t):
    g = defaultdict(list)
    for l,r,c in edges:
        g[l].append((c,r))

    q, seen = [(0,f,())], set()
    while q:
        (cost,v1,path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t: return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 not in seen:
                    heappush(q, (cost+c, v2, path))

    return float("inf")

def make_edges(nodes,links):
    edges=[]
    temp=[]
    for i in range(0, 64):
        for j in range(0, 64):
            if links[i][j] == 1:
                temp.append(nodes[i])
                temp.append(nodes[j])
                zs=int(i/16)
                ts=i%16
                ys=int(ts/4)
                xs=ts%4

                zd = int(j / 16)
                td = j % 16
                yd = int(td / 4)
                xd = td % 4
                if xs==xd and ys==yd and abs(zs-zd)==1:
                    temp.append(4)
                elif zs==zd and (xs!=xd or ys!=yd):
                    temp.append(3+ math.ceil((((xd-xs)**2)+((ys-yd)**2))**0.5))
                else:
                    print("link perturbation went wrong: ", i, " , ", j)
                edges.append(temp)
                #edges1.append(temp)
                temp = []
                # c=c+1
    return edges

####################################### params calculation ##########################

def calc_params(nodes,links, traffic):
    m = 0
    d = 0
    link_util = []
    edges = make_edges(nodes, links)
    for i in range(0, 64):
        t = []
        for j in range(0, 64):
            t.append(0)
        link_util.append(t)
    # dev
    for i in range(0, 64):
        for j in range(0, 64):
            if nodes[i] == nodes[j]:
                continue
            # print(i, "  ", j)
            p = str(dijkstra(edges, nodes[i], nodes[j]))
            p_break = p.split(',')
            for k in range(1, len(p_break) - 2):
                node1 = int(p_break[k][2:])
                node2 = int(p_break[k + 1][2:])
                ind1 = nodes.index(node1)
                ind2 = nodes.index(node2)
                if links[ind1][ind2] != 1:
                    print('something is wrong..!!')
                link_util[ind1][ind2] = link_util[ind1][ind2] + traffic[nodes[i]][nodes[j]]

    for i in range(0, 64):
        for j in range(0, 64):
            m = m + link_util[i][j]
    m = m / 144
    for i in range(0, 64):
        for j in range(i, 64):
            if (links[i][j] != 1):
                continue
            d = d + (link_util[i][j] + link_util[j][i] - m)**2
    d = d**0.5

    return m, d

################################ PHV calculator #####################################

def dominates(p, q, k=None):
    if k is None:
        k = len(p)
    d = True
    while d and k < len(p):
        d = not (q[k] > p[k])
        k += 1
    return d

def insert(p, k, pl):
    ql = []
    while pl and pl[0][k] > p[k]:
        ql.append(pl[0])
        pl = pl[1:]
    ql.append(p)
    while pl:
        if not dominates(p, pl[0], k):
            ql.append(pl[0])
        pl = pl[1:]
    return ql

def slice(pl, k, ref):
    p = pl[0]
    pl = pl[1:]
    ql = []
    s = []
    while pl:
        ql = insert(p, k + 1, ql)
        p_prime = pl[0]
        s.append(((p[k] - p_prime[k]), ql))
        p = p_prime
        pl = pl[1:]
    ql = insert(p, k + 1, ql)
    s.append(((p[k] - ref[k]), ql))
    return s

def phv_calculator(archive, new_pt):
    ps = deepcopy(archive)
    ps.append(deepcopy(new_pt))
    ref = [0,0]
    n = min([len(p) for p in ps])
    pl = ps[:]
    pl.sort(key=lambda x: x[0], reverse=True)
    s = [(1, pl)]
    for k in range(n - 1):
        s_prime = []
        for x, ql in s:
            for x_prime, ql_prime in slice(ql, k, ref):
                s_prime.append((x * x_prime, ql_prime))
        s = s_prime
    vol = 0
    for x, ql in s:
        vol = vol + x * (ql[0][n - 1] - ref[n - 1])
    return vol
    #print(vol)

############################### Make Perturbation #################################

def perturb(nodes,links, case, last_move):
    r1 = random.random()
    if case==0:
        threshold=0.6
    else:
        threshold=1.1
    if (r1 < threshold):  # exchange cores 0.6
        while 1:
            rs = random.randint(0, 63)
            rd = random.randint(0, 63)
            i = nodes[rs]
            j = nodes[rd]
            t1 = nodes[rs]
            nodes[rs] = nodes[rd]
            nodes[rd] = t1  # exchanged nodes
            last_move[0] = 0 #core swap
            last_move[1] = rs
            last_move[2] = rd
            last_move[3] = -1
            last_move[4] = -1
            last_move[5] = -1
            last_move[6] = -1
            break
    else:  # change links
        while 1:
            rl1 = random.randint(0, 3)
            rs1 = random.randint(0, 15)
            rd1 = random.randint(0, 15)
            rl2 = random.randint(0, 3)
            rs2 = random.randint(0, 15)
            rd2 = random.randint(0, 15)
            if (rs1 == rd1) or (rs2 == rd2):
                continue
            l1 = links[rl1 * 16 + rs1][rl1 * 16 + rd1]  # remove
            l2 = links[rl2 * 16 + rs2][rl2 * 16 + rd2]  # add
            if (l1 == 0) or (l2 == 1):  # link absent/present
                continue

            # move links
            links[rl1 * 16 + rs1][rl1 * 16 + rd1] = 0
            links[rl1 * 16 + rd1][rl1 * 16 + rs1] = 0
            links[rl2 * 16 + rs2][rl2 * 16 + rd2] = 1
            links[rl2 * 16 + rd2][rl2 * 16 + rs2] = 1
            # check for islands
            edges_trial=[]
            temp=[]
            for i in range(0, 64):
                for j in range(0, 64):
                    temp.append(nodes[i])
                    temp.append(nodes[j])
                    if links[i][j] == 1:
                        temp.append(1)
                        # edges1.append(temp)
                    else:
                        temp.append(999)
                    edges_trial.append(temp)
                    # edges1.append(temp)
                    temp = []
            island=0
            for i in range(0, 64):
                p = str(dijkstra(edges_trial, nodes[rl1 * 16 + rs1], nodes[i]))
                q = str(dijkstra(edges_trial, nodes[rl1 * 16 + rd1], nodes[i]))
                p=p.split(',')
                q=q.split(',')
                cost_trial1=int(p[0][1:])
                cost_trial2=int(q[0][1:])
                if (cost_trial1>100 or cost_trial2>100):
                    island=1
                    break
            if (island==1): #reverse everything and restart
                links[rl1 * 16 + rs1][rl1 * 16 + rd1] = 1
                links[rl1 * 16 + rd1][rl1 * 16 + rs1] = 1
                links[rl2 * 16 + rs2][rl2 * 16 + rd2] = 0
                links[rl2 * 16 + rd2][rl2 * 16 + rs2] = 0
                continue
            # if reached till this point, then everything successful
            last_move[0]=1
            last_move[3]=rl1 * 16 + rs1
            last_move[4]=rl1 * 16 + rd1
            last_move[5]=rl2 * 16 + rs2
            last_move[6]=rl2 * 16 + rd2
            last_move[1]=-1
            last_move[2]=-1
            break
    return nodes, links

################################# reverse perturb ######################

def reverse_perturb(nodes,links,last_move):
    case=last_move[0]
    if case==0: #core swap
        rs=last_move[1]
        rd=last_move[2]
        t1 = nodes[rs]
        nodes[rs] = nodes[rd]
        nodes[rd] = t1  # re-exchanged nodes
    else:
        a=last_move[3]
        b=last_move[4]
        c=last_move[5]
        d=last_move[6]
        links[a][b] = 1
        links[b][a] = 1
        links[c][d] = 0
        links[d][c] = 0 #reversed links
    return nodes,links

################################# best perturb ##########################

def best_perturb(nodes,links, best_move):
    case = best_move[0]
    if case == 0:  # core swap
        rs = best_move[1]
        rd = best_move[2]
        t1 = nodes[rs]
        nodes[rs] = nodes[rd]
        nodes[rd] = t1
    else: #link place
        a = best_move[3]
        b = best_move[4]
        c = best_move[5]
        d = best_move[6]
        links[a][b] = 0
        links[b][a] = 0
        links[c][d] = 1
        links[d][c] = 1
    return nodes, links


################################ MAIN #####################################
def main(start_time):
    glo=0
    last_move = []
    best_move = []
    for i in range(0, 7):
        last_move.append(-1)
        best_move.append(-1)
    num_links = 144
    current_sw_mean = 0
    current_sw_dev = 0
    random.seed(1000)
    count1=0
    phv=-1
    best_phv=0
    global_archive=[]
    local_archive=[] #complete information of NoC
    local_pareto=[] #co-ordinates only
    train_set=[]
    labels=[]
    nodes=create_nodes() #create mesh
    links=create_mesh() #create mesh
    traffic,injection=load_traffic() #load benchmark
    mesh_mean, mesh_dev = calc_params(nodes,links,traffic)
    elapsed=timeit.default_timer()-start_time
    new_point = [0, 0, deepcopy(nodes), deepcopy(links),elapsed] #mesh
    new_pareto = [0,0] #shadows new_point
    local_pareto.append([0,0]) #shadows local_archive
    local_archive.append([0,0,deepcopy(nodes),deepcopy(links),elapsed]) #normalizing to remove absoluteness
    mesh_nodes=deepcopy(nodes)
    mesh_links=deepcopy(links)
    inp=[0,0]
    for i in range(0,len(nodes)):
		inp.append(injection[nodes[i]])
    train_set.append(inp)
    labels.append(9999)
    #print(train_set)

    f=0
    stop=0
    bias=0.01
    bx=0
    num_iter=10
    w1=0.7
    w2=0.3

    while 1:
        bx=bx+1
        count1=0
        for i in range(0,7):
            best_move[i]=-1
            last_move[i]=-1
        for a in range(0, num_iter):  # num_iter seperate perturbations from same starting point
            # make a perturbation
            nodes,links=perturb(nodes,links,0, last_move)
            current_sw_mean, current_sw_dev = calc_params(nodes,links, traffic)
            current_sw_mean=(1-current_sw_mean/mesh_mean)*w1
            current_sw_dev=(1-current_sw_dev/mesh_dev)*w2
            new_pareto=[current_sw_mean,current_sw_dev]
            phv = phv_calculator(local_pareto, new_pareto)
            if phv > best_phv:  # if current phv greater than current best phv i.e. successful perturbation
                best_move=deepcopy(last_move)
                best_phv=phv
            else:
                ### optional: can add SA-like features here
                count1=count1+1
            nodes,links=reverse_perturb(nodes,links, last_move)

        if(count1==num_iter): #no improvement in num_iter perturbations, assume reached minima
            print('********************************')
            bias = bias - 0.001
            bx=0
            ########### save the config #########
            elapsed=timeit.default_timer()-start_time
            asdq=0
            if(len(local_archive)>10):
				asdq=10
            else:
				asdq=len(local_archive)
            for b in range(0,asdq):
				text_file = open("Output"+str(glo)+".txt", "w")
				glo=glo+1
				text_file.write("mean= %f \n" % local_archive[b][0])
				text_file.write("dev= %f \n" % local_archive[b][1])
				text_file.write("timestamp= %f \n" % local_archive[b][4])
				for i in range(0,64):
					text_file.write("%s, " % local_archive[b][2][i])
					if i%16==15:
						text_file.write("\n")
				text_file.write("\n")
				for i in range(0, 64):
					for j in range(0, 64):
						text_file.write("%d, " % local_archive[b][3][i][j])
					text_file.write("\n")
				text_file.close()
            f=1
            global_archive.append(local_archive)
            local_archive=[]
            local_pareto=[]
            for i in range(0,len(labels)):
                if labels[i]==9999:
                    labels[i]=best_phv
            ######### send out for training ############

            regr = RandomForestRegressor(100)
            regr.fit(train_set, labels)

            stop=stop+1
            if(stop>=100):
                quit()

            ######### predict good start point ##########
            nodes = deepcopy(mesh_nodes)
            links = deepcopy(mesh_links)
            mphv=best_phv
            best_phv = 0
            c2=0
            while 1:
                n=random.randint(0,5)
                for i in range(0,n):
                    nodes,links=perturb(nodes,links,1,last_move)
                tm,td=calc_params(nodes,links,traffic)
                tm=(1-tm/mesh_mean)*w1
                td=(1-td/mesh_dev)*w2
                inpt=[tm,td]
                for x in range(0,len(nodes)):
                    if nodes[x]<8: #CPU
                        inpt.append(1)
                    elif nodes[x]>7 and nodes[x]<24: #MC
                        inpt.append(2)
                    elif nodes[x]>23: #GPU
                        inpt.append(3)

                predicted_end_phv=regr.predict([inpt])+bias #adding slight bias
                print('predicted= ',predicted_end_phv, ' & mphv= ',mphv)
                if(predicted_end_phv>=mphv): # choose this as start point
                    local_pareto.append([tm,td])
                    elapsed=timeit.default_timer()-start_time
                    local_archive.append([tm,td,deepcopy(nodes),deepcopy(links),elapsed])
                    break
                else:
                    c2=c2+1 #good starting point not found
                if c2==20:
                    break
            if c2==20: #if good starting points not found, pick random
                num_soln=0
                for i in range(0,len(global_archive)): # count total number of solutions obtained so far
					num_soln = num_soln + len(global_archive[i])
                threshold=float(50)/num_soln
                r1 = random.random()
                if (r1<threshold): #pick random
                    nodes = deepcopy(mesh_nodes)
                    links = deepcopy(mesh_links)
                    for i in range(0,10): # make 10 random perturbation
                        nodes,links=perturb(nodes,links,0,last_move)
                    tm,td=calc_params(nodes,links,traffic)
                    tm=(1-tm/mesh_mean)*w1
                    td=(1-td/mesh_dev)*w2
                    local_pareto.append([tm,td])
                    elapsed=timeit.default_timer()-start_time
                    local_archive.append([tm,td,deepcopy(nodes),deepcopy(links),elapsed])
                else: #pick from existing solutions
					r2=random.randint(0,len(global_archive)-1)
					r3=random.randint(0,len(global_archive[r2])-1)
					local_archive.append(global_archive[r2][r3])
					local_pareto.append([global_archive[r2][r3][0],global_archive[r2][r3][1]])
				
        else: ## good solution found
            nodes, links = best_perturb(nodes, links,best_move)
            current_sw_mean, current_sw_dev = calc_params(nodes, links, traffic)
            current_sw_mean = (1 - current_sw_mean / mesh_mean)*w1
            current_sw_dev = (1 - current_sw_dev / mesh_dev)*w2
            ### new potential best candidate ready
            ## add candidate to local archive
            elapsed=timeit.default_timer()-start_time
            new_point = [current_sw_mean, current_sw_dev, deepcopy(nodes), deepcopy(links), elapsed]
            new_pareto = [current_sw_mean, current_sw_dev]
            i=0
            while len(local_pareto)>0:
                if (dominates(new_pareto, local_pareto[i], 0)):  # if new point completes dominates any existing point
                    del local_pareto[i]
                    del local_archive[i]
                    i=i-1
                i=i+1
                if (i>=len(local_archive)):
                    break
            local_pareto.append(new_pareto)
            local_archive.append(new_point)
            if (len(local_archive)>15): #optional: trimming to reduce size
				b_temp1 = []
				for i in range(0, len(local_archive)):
					score=0.7*(1-local_archive[i][0])+0.3*(1-local_archive[i][1]) #custom scoring function
					b_temp1.append(score)
				b_num = numpy.array(b_temp1)
				b_ind = b_num.argsort()[:10]  # index of lowest n scores
				reduced_archive = []
				reduced_pareto=[]
				for i in range(0, len(b_ind)):
					reduced_archive.append(local_archive[b_ind[i]])
					reduced_pareto.append(local_pareto[b_ind[i]])
				local_archive = deepcopy(reduced_archive)
				local_pareto = deepcopy(reduced_pareto)
				
            t2 = [current_sw_mean, current_sw_dev]
            for k in range(0,len(nodes)):
				if nodes[k]<8: #CPU
				    t2.append(1)
				elif nodes[k]>7 and nodes[k]<24: #MC
				    t2.append(2)
				elif nodes[k]>23: #GPU
				    t2.append(3)
            train_set.append(t2)
            labels.append(9999)


start_time=timeit.default_timer()
main(start_time)

