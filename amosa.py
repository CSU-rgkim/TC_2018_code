#################################### Import ###############################
from collections import defaultdict
from heapq import *
import xlrd
from copy import deepcopy
from random import randint, random, seed
import math
import numpy
import random
import timeit
import sys

################################ Global Variables #############################

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
    for i in range(0, len(content)):
        temp = content[i].split()
        t = []
        for j in range(0, len(temp)):
            t.append(float(temp[j]))
        traffic.append(t)
    return traffic

################################# Dijkstra shortest path algorithm ###############################
def dijkstra(edges, f, t):
    g = defaultdict(list)
    for l, r, c in edges:
        g[l].append((c, r))

    q, seen = [(0, f, ())], set()
    while q:
        (cost, v1, path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t: return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 not in seen:
                    heappush(q, (cost + c, v2, path))

    return float("inf")

def make_edges(nodes, links):
    edges = []
    temp = []
    for i in range(0, 64):
        for j in range(0, 64):
            if links[i][j] == 1:
                temp.append(nodes[i])
                temp.append(nodes[j])
                zs = int(i / 16)
                ts = i % 16
                ys = int(ts / 4)
                xs = ts % 4

                zd = int(j / 16)
                td = j % 16
                yd = int(td / 4)
                xd = td % 4
                if xs == xd and ys == yd and abs(zs - zd) == 1:
                    temp.append(4)
                elif zs == zd and (xs != xd or ys != yd):
                    temp.append(3 + math.ceil((((xd - xs) ** 2) + ((ys - yd) ** 2)) ** 0.5))
                edges.append(temp)
                # edges1.append(temp)
                temp = []
                # c=c+1
    return deepcopy(edges)

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
            d = d + (link_util[i][j] + link_util[j][i] - m) ** 2
    d = d ** 0.5

    return m, d

############################### Make Perturbation #################################
def perturb(nodes,links):
    threshold=0.6
    r1=random.random()
    if (r1 < threshold):  # exchange cores if < 0.6
        while 1:
            rs = random.randint(0, 63)
            rd = random.randint(0, 63)
            i = nodes[rs]
            j = nodes[rd]
            t1 = nodes[rs]
            nodes[rs] = nodes[rd]
            nodes[rd] = t1  # exchanged nodes
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
            break
    return nodes, links

######################################## Dominance ###############################

def delta_dom(point1, point2):
    count1 = 1

    p1=[point1[0],point1[1]]
    p2=[point2[0],point2[1]]
    if not dominates (p1,p2):
        return 0

    for i in range(0,2): ### here number obj = 2
        if p1[i]==p2[i]: ### if same, ignore
            continue
        count1=count1*abs(point1[i]-point2[i])
    return count1

def dominates(p, q, k=None):
    if k is None:
        k = len(p)
    d = True
    while d and k < len(p):
        d = not (q[k] < p[k])
        k += 1
    return d

################################ MAIN $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def main(start_time):
    random.seed(1000)
    count1 = -1
    best_sw_archive = []

    nodes = create_nodes()  # create mesh
    links = create_mesh()  # create mesh
    traffic = load_traffic()  # load benchmark
    mesh_mean, mesh_dev = calc_params(nodes, links, traffic)
    elapsed = timeit.default_timer() - start_time
    best_sw = [1, 1, deepcopy(nodes), deepcopy(links), elapsed]  # normalizing 
    amosa_flag = 1

    for i in range(0, 5):
        v = random.randint(0, 10)
        for j in range(0, v):
            nodes, links = perturb(nodes, links)
        
        tm, td, = calc_params(nodes, links, traffic)
        tm = tm / mesh_mean
        td = td / mesh_dev
        elapsed = timeit.default_timer() - start_time
        sw = [tm, td, deepcopy(nodes), deepcopy(links), elapsed]
        best_sw_archive.append(sw)  # initializing the archive

    ################################################ Enter AMOSA ############################################

    temp=500
    z = 0
    el = timeit.default_timer() - start_time
    x = int((el / 1200) + 1) * 1200

    while 1:
        temp=temp*0.95
        ### cluster/prune if size becomes too big
        score=[]
        for i in range(0,len(best_sw_archive)):
			s=best_sw_archive[i][0]*0.7+best_sw_archive[i][1]*0.3 #custom scoring
			score.append(s)
        if len(best_sw_archive) > 30:
            b_num = numpy.array(score)
            b_ind = b_num.argsort()[:20]  # index of lowest 20 phv
            reduced_archive = []
            for i in range(0, len(b_ind)):
                reduced_archive.append(deepcopy(best_sw_archive[b_ind[i]]))
            best_sw_archive = deepcopy(reduced_archive)

        ### terminate
        z = z + 1
        if z == 2500: ## put stop condition here, currently just letting it run for ~infinity
            quit(0)

        r = random.randint(0, len(best_sw_archive) - 1)
        best_sw = deepcopy(best_sw_archive[r])  # best_sw is the randomly picked current pt

        for i in range(0, 500):  # make num_iter perturbations
            ### print out after some time e.g. every ~20 mins
            el = timeit.default_timer() - start_time
            if el > x:
                x=x+1200
                max_print=-1
                if len(best_sw_archive)>10:
                    max_print=10
                else:
                    max_print=len(best_sw_archive)
                for k in range(0, max_print): #printing to file
                    count1=count1+1
                    text_file = open("Output" + str(count1) + ".txt", "w")
                    text_file.write("mean= %f \n" % best_sw_archive[k][0])
                    text_file.write("dev= %f \n" % best_sw_archive[k][1])
                    text_file.write("timestamp= %f \n" % best_sw_archive[k][4])
                    for asd in range(0, 64):
                        text_file.write("%s, " % best_sw_archive[k][2][asd])
                        if asd % 16 == 15:
                            text_file.write("\n")
                    text_file.write("\n")
                    for asd in range(0, 64):
                        for j in range(0, 64):
                            text_file.write("%d, " % best_sw_archive[k][3][asd][j])
                        text_file.write("\n")
                    text_file.close()

            nodes = deepcopy(best_sw[2])
            links = deepcopy(best_sw[3])
            ### do perturbation
            nodes, links = perturb(nodes, links)
            tm, td = calc_params(nodes, links, traffic)
            tm = tm / mesh_mean
            td = td / mesh_dev
            elapsed = timeit.default_timer() - start_time
            new_sw = [tm, td, deepcopy(nodes), deepcopy(links), elapsed]

            # check dominance
            if best_sw[0] < new_sw[0] and best_sw[1] < new_sw[1]:  # current point dominates new point
                d=0
                c1=0
                for i in range(0,len(best_sw_archive)):
                    d_temp=delta_dom(best_sw_archive[i], new_sw)
                    if (d_temp!=0):
                        c1=c1+1
                    d = d+d_temp
                d=d+delta_dom(best_sw,new_sw)
                d_avg=d/(c1+1)
                
                m_factor=0
                if d_avg*temp>5:
                    m_factor=5
                else:
                    m_factor=d_avg*temp
                prob = float(1 / (1+2.718**m_factor))
                rp = random.random()
                if rp < prob:  # set new point as current point
                    best_sw = deepcopy(new_sw)

            elif new_sw[0] < best_sw[0] and new_sw[1] < best_sw[1]:  # new point dominates current point
                d=0
                for j in range(0,len(best_sw_archive)):
                    archive_point=[best_sw_archive[j][0],best_sw_archive[j][1]]
                    new_point=[new_sw[0],new_sw[1]]
                    if(dominates(archive_point,new_point,0)):
                        d=d+1

                if d > 1:  # new point dominated by k points in archive
                    min1=999
                    ind_min1=-1
                    for j in range(0, len(best_sw_archive)):
                        archive_point = [best_sw_archive[j][0], best_sw_archive[j][1]]
                        new_point = [new_sw[0], new_sw[1]]
                        if (dominates(archive_point, new_point, 0)):
                            dom=delta_dom(archive_point,new_point)
                            if dom<min1:
                                min1=dom
                                ind_min1=j

                    prob = float(1 / (2.718**(-min1)))
                    rp=random.random()
                    if rp < prob:  # set new point as current point
                        best_sw = deepcopy(best_sw_archive[ind_min1])
                    else:
                        best_sw = deepcopy(new_sw)

                else:  # new point either dominates or non-dominates other points
                    # anyone that is dominated by new point is bad
                    temp_archive = []
                    for j in range(0, len(best_sw_archive)):
                        if not (new_sw[0] < best_sw_archive[j][0] and new_sw[1] < best_sw_archive[j][1]):  # new point does not dominate these archive points
                            temp_archive.append(deepcopy(best_sw_archive[j]))
                    best_sw_archive = deepcopy(temp_archive)
                    best_sw_archive.append(deepcopy(new_sw))
                    best_sw = deepcopy(new_sw)

            else:  # non-dominance stand-off
                d = 0
                for j in range(0, len(best_sw_archive)):
                    archive_point = [best_sw_archive[j][0], best_sw_archive[j][1]]
                    new_point = [new_sw[0], new_sw[1]]
                    if (dominates(archive_point, new_point, 0)):
                        d = d + 1

                if d > 1:  # new point dominated by k points in archive
                    sum1=0
                    for i in range(0,len(best_sw_archive)):
                        archive_point = [best_sw_archive[j][0], best_sw_archive[j][1]]
                        new_point = [new_sw[0], new_sw[1]]
                        if (dominates(archive_point, new_point, 0)):
                            sum1=sum1+delta_dom(archive_point,new_point)
                    sum1=sum1/d #delta_dom_avg
                    
                    m_factor=0
                    if sum1*temp>5:
                        m_factor=5
                    else:
                        m_factor=sum1*temp
                    prob = float(1 / (1+2.718**(m_factor)))
                    rp=random.random()
                    if rp < prob:  # set new point as current point
                        best_sw = deepcopy(new_sw)
                else:  # new point either dominates or non-dominates other points
                    # anyone that is dominated by new point is bad
                    temp_archive = []
                    for j in range(0, len(best_sw_archive)):
                        if not (new_sw[0] < best_sw_archive[j][0] and new_sw[1] < best_sw_archive[j][1]):  # new point does not dominate these archive points
                            temp_archive.append(deepcopy(best_sw_archive[j]))
                    best_sw_archive = deepcopy(temp_archive)
                    best_sw_archive.append(deepcopy(new_sw))
                    best_sw = deepcopy(new_sw)



start_time = timeit.default_timer()
main(start_time)

