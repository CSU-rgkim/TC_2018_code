#################################### Import ###############################
from collections import defaultdict
from heapq import *
from copy import deepcopy
from random import randint, random, seed
import math
import numpy
import random
import timeit

########################### Creating node placements #############################

def create_nodes_greedy_local(level, count1, sorted_node, traffic, nodes1):
    nodes = deepcopy(nodes1)
    c = 0
    core_placed = sorted_node[level]
    core_comm = numpy.array(traffic[core_placed])
    rank = (-core_comm).argsort()  # stores the cores in order of communication with placed core
    layer_order = [-1, -1, -1, -1]
    layer_placed = int(count1 / 16)
    if (layer_placed == 0):  # core placed was in this layer
        layer_order = [0, 1, 2, 3]
    elif (layer_placed == 1):  # core placed was in this layer
        layer_order = [1, 2, 0, 3]
    elif (layer_placed == 2):  # core placed was in this layer
        layer_order = [2, 1, 3, 0]
    else:  # core placed was in this layer
        layer_order = [3, 2, 1, 0]

    layer = -1
    for i in range(0, 64):
        layer = layer_order[int(i / 16)]
        if nodes[layer * 16 + i % 16] != -1:  # already filled
            continue
        while rank[c] in nodes:
            c = c + 1
        nodes[layer * 16 + i % 16] = rank[c]
        c = c + 1

    return nodes


def create_nodes_greedy_global(level, count1, sorted_node, traffic, nodes1, type1):
    nodes = deepcopy(nodes1)
    c = 0
    rank = sorted_node  # stores the cores in order of communication
    layer_order = [-1, -1, -1, -1]
    layer_placed = int(count1 / 16)
    if (layer_placed == 0):  # core placed was in this layer
        layer_order = [0, 1, 2, 3]
    elif (layer_placed == 1):  # core placed was in this layer
        layer_order = [1, 2, 0, 3]
    elif (layer_placed == 2):  # core placed was in this layer
        layer_order = [2, 1, 3, 0]
    else:  # core placed was in this layer
        layer_order = [3, 2, 1, 0]

    if type1 == 1:
        layer_order.reverse()

    layer = -1
    for i in range(0, 64):
        layer = layer_order[int(i / 16)]
        if nodes[layer * 16 + i % 16] != -1:  # already filled
            continue
        while rank[c] in nodes:
            c = c + 1
        nodes[layer * 16 + i % 16] = rank[c]
        c = c + 1

    return nodes


def create_nodes_greedy_rand(level, count1, sorted_node, traffic, nodes1):
    nodes = deepcopy(nodes1)
    c = 0
    rank = numpy.random.permutation(64)
    for i in range(0, 64):
        if nodes[i] != -1:  # already filled
            continue
        while rank[c] in nodes:
            c = c + 1
        nodes[i] = rank[c]
        c = c + 1

    return nodes


########################## Creating link connectivity #############################
def create_links_only_tsv():  # add 48 tsvs only
    links = numpy.zeros(shape=(64, 64))
    for i in range(0, 3):
        for j in range(0, 16):
            links[16 * i + j][16 * i + j + 16] = 1
            links[16 * i + j + 16][16 * i + j] = 1
    return links


##### link adding policies part 1 ####

def create_links_mesh(links_only_tsv):
    links = deepcopy(links_only_tsv)
    i = 0
    for i in range(0, 64):
        for j in range(i, 64):
            zi = int(i / 16)
            zj = int(j / 16)
            if (zi != zj):
                continue
            source = i - 16 * zi
            dest = j - 16 * zj
            xs = source % 4
            ys = int(source / 4)
            xd = dest % 4
            yd = int(dest / 4)
            if (xd == xs and abs(ys - yd) == 1) or (yd == ys and abs(xs - xd) == 1):
                links[i][j] = 1
                links[j][i] = 1
    # print(links[0])
    return links

################################### Load traffic ####################################
def load_traffic():
    with open("input_traffic.txt") as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    traffic = []
    # print(content)
    for i in range(0, len(content)):
        temp = content[i].split()
        t = []
        for j in range(0, len(temp)):
            # print(temp[j])
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
                else:
                    print("link perturbation went wrong: ", i, " , ", j)
                edges.append(temp)
                # edges1.append(temp)
                temp = []
                # c=c+1
    return deepcopy(edges)


####################################### params calculation ##########################

def calc_params_virtual_mesh(nodes, links, traffic):
    # links is a mesh, following x-y-z routing
    m = 0
    d = 0
    num_cases = 0  # different than num_links
    link_util = []
    for i in range(0, 64):
        t = []
        for j in range(0, 64):
            t.append(0)
        link_util.append(t)

    for i in range(0, 64):
        for j in range(0, 64):
            if nodes[i] == nodes[j]:
                continue
            ind1 = i
            ind3 = j
            # print(ind1, ' <--> ', ind3)
            while int(ind1 / 16) != int(ind3 / 16):  # not in same layer go z first
                if int(ind1 / 16) < int(ind3 / 16):
                    link_util[ind1][ind1 + 16] = link_util[ind1][ind1 + 16] + traffic[nodes[i]][nodes[j]]
                    ind1 = ind1 + 16
                else:
                    link_util[ind1][ind1 - 16] = link_util[ind1][ind1 - 16] + traffic[nodes[i]][nodes[j]]
                    ind1 = ind1 - 16
            while ind1 % 4 != ind3 % 4:  # go x next
                if ind1 % 4 < ind3 % 4:
                    link_util[ind1][ind1 + 1] = link_util[ind1][ind1 + 1] + traffic[nodes[i]][nodes[j]]
                    ind1 = ind1 + 1
                else:
                    link_util[ind1][ind1 - 1] = link_util[ind1][ind1 - 1] + traffic[nodes[i]][nodes[j]]
                    ind1 = ind1 - 1
            while int(ind1 / 4) != int(ind3 / 4):  # go y last
                if int(ind1 / 4) < int(ind3 / 4):
                    link_util[ind1][ind1 + 4] = link_util[ind1][ind1 + 4] + traffic[nodes[i]][nodes[j]]
                    ind1 = ind1 + 4
                else:
                    link_util[ind1][ind1 - 4] = link_util[ind1][ind1 - 4] + traffic[nodes[i]][nodes[j]]
                    ind1 = ind1 - 4

    for i in range(0, 64):
        for j in range(0, 64):
            m = m + link_util[i][j]
    m = m / 144
    for i in range(0, 64):
        for j in range(i, 64):
            if (links[i][j] != 1):
                continue
            d = d + (link_util[i][j] + link_util[j][i] - m) ** 2
    d = (d ** 0.5) / 144
    # print(num_cases)
    return m, d


def calc_params_virtual(nodes, links, traffic):
    m = 0
    d = 0
    num_cases = 0  # different than num_links
    link_util = []
    edges = make_edges(nodes, links)
    for i in range(0, 64):
        t = []
        for j in range(0, 64):
            t.append(0)
        link_util.append(t)
    for i in range(0, 64):
        for j in range(0, 64):
            if nodes[i] == nodes[j]:
                continue
            p = str(dijkstra(edges, nodes[i], nodes[j]))
            p_break = p.split(',')
            if p_break[0][1:].isdigit():  # considering possibility of islands
                num_cases = num_cases + 1

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
    m = m / num_cases
    for i in range(0, 64):
        for j in range(i, 64):
            if (links[i][j] != 1):
                continue
            d = d + (link_util[i][j] + link_util[j][i] - m) ** 2
    d = (d ** 0.5) / num_cases
    # print(num_cases)
    return m, d


##################################### link placement part2 ###########################

def create_links_sw2(virtual_node, traffic, links, num_links):  # greedy2: fij/dij

    traffic1 = numpy.array(traffic)
    for i in range(0, 64):
        for j in range(0, 64):
            if (i == j):
                continue
            loc_i = virtual_node.index(i)  # source
            loc_j = virtual_node.index(j)  # dest
            z_i = int(loc_i / 16)
            z_j = int(loc_j / 16)
            y_i = int((loc_i - 16 * z_i) / 4)
            y_j = int((loc_j - 16 * z_j) / 4)
            x_i = (loc_i - 16 * z_i) % 4
            x_j = (loc_j - 16 * z_j) % 4
            dij = abs(z_i - z_j) + math.ceil(((x_i - x_j) ** 2 + (y_i - y_j) ** 2) ** 0.5)
            traffic1[i][j] = traffic1[i][j] / dij
    virt_link = deepcopy(links)
    i = 0
    while (i < num_links):
        x = (numpy.unravel_index(numpy.argmax(traffic1, axis=None), traffic1.shape))
        # print(numpy.argmax(traffic1),x,i)
        s = x[0]
        d = x[1]
        traffic1[s][d] = -1
        traffic1[d][s] = -1
        loc_s = virtual_node.index(s)
        loc_d = virtual_node.index(d)
        if (int(loc_s / 16) == int(loc_d / 16)) and sum(virt_link[loc_s]) < 8 and sum(
                virt_link[loc_d]) < 8:  # should be in same layer and there shouldn't exist a link beforehand
            if virt_link[loc_s][loc_d] != 1:
                virt_link[loc_s][loc_d] = 1
                virt_link[loc_d][loc_s] = 1
                i = i + 1

    return virt_link


def create_links_flexible_greedy2(virtual_node, traffic, links, threshold,
                                  num_links):  # pseudo-greedy1: threshold% fij and (1-threshold)% not fij
    virt_link = deepcopy(links)
    traffic1 = numpy.array(traffic)
    i = 0
    while (i < num_links):
        r = random.random()
        if (r < threshold):  # fij
            while 1:
                x = (numpy.unravel_index(numpy.argmax(traffic1, axis=None), traffic1.shape))
                s = x[0]
                d = x[1]
                traffic1[s][d] = 0
                traffic1[d][s] = 0
                loc_s = virtual_node.index(s)
                loc_d = virtual_node.index(d)
                if (int(loc_s / 16) == int(loc_d / 16)) and sum(virt_link[loc_s]) < 8 and sum(
                        virt_link[loc_d]) < 8:  # should be in same layer and there shouldn't exist a link beforehand
                    if virt_link[loc_s][loc_d] != 1:
                        virt_link[loc_s][loc_d] = 1
                        virt_link[loc_d][loc_s] = 1
                        i = i + 1
                        break
        else:  # add random
            while 1:
                layer = random.randint(0, 3)
                x1 = random.randint(0, 3)
                x2 = random.randint(0, 3)
                y1 = random.randint(0, 3)
                y2 = random.randint(0, 3)
                if (x1 == x2 and y1 == y2):  # same core-same core
                    continue
                if virt_link[16 * layer + 4 * y1 + x1][16 * layer + 4 * y2 + x2] != 1 and sum(
                        virt_link[16 * layer + 4 * y1 + x1]) < 8 and sum(virt_link[16 * layer + 4 * y2 + x2]) < 8:
                    virt_link[16 * layer + 4 * y1 + x1][16 * layer + 4 * y2 + x2] = 1
                    virt_link[16 * layer + 4 * y2 + x2][16 * layer + 4 * y1 + x1] = 1
                    i = i + 1
                    break

    return virt_link


############################## main ################################

def main(start_time):
    num_links = 144
    random.seed(1000)
    traffic = load_traffic()  # load benchmark
    total_injection = []
    links_only_tsv = create_links_only_tsv()

    for i in range(0, 64):
        total_injection.append(sum(traffic[i]))

    #################### sort ####################

    total_injection = numpy.array(total_injection)
    sorted_node = (-total_injection).argsort()

    start_time = timeit.default_timer()
    hours = 0
    level = 0
    nodes1 = []
    for i in range(0, 64):
        nodes1.append(-1)

    ############## place nodes first ##############
    virtual_links = create_links_mesh(links_only_tsv)
    ref_node = []
    for i in range(0, 64):
        ref_node.append(i)
    ref_mean, ref_dev = calc_params_virtual_mesh(ref_node, virtual_links, traffic)

    current_level = [nodes1]
    next_level = []
    while (level < 64):  # each level one node is placed
        global_min_score = 999
        elapsed = timeit.default_timer() - start_time
        hours = hours + float(elapsed / 3600)
        start_time = timeit.default_timer()
        for i in range(0, len(current_level)):
            # make max of 64 sub-cases of node placements at each level, marking by count1 variable here
            for count1 in range(0, 64):
                nodes = deepcopy(current_level[i])
                if (nodes[count1] != -1):  # spot already taken
                    continue
                nodes[count1] = sorted_node[level]
                ## virtual links is always a mesh at this point
                local_min_score = 999
                for j in range(0, 6):  # make 5 different cases
                    if j == 0:
                        virtual_nodes = create_nodes_greedy_local(level, count1, sorted_node, traffic,
                                                                  nodes)  # creating possible node configs
                    elif j > 0 and j < 3:
                        virtual_nodes = create_nodes_greedy_global(level, count1, sorted_node, traffic, nodes,
                                                                   j - 1)  # creating possible node configs
                    else:
                        virtual_nodes = create_nodes_greedy_rand(level, count1, sorted_node, traffic,
                                                                 nodes)  # creating possible node configs

                    # evaluate quality

                    mean, dev = calc_params_virtual_mesh(virtual_nodes, virtual_links, traffic)
                    mean = float(mean) / ref_mean
                    dev = float(dev) / ref_dev
                    score = 0.5 * mean + 0.5 * dev

                    if score < local_min_score:
                        local_min_score = score

                temp = []
                # print(nodes, local_min_score)
                temp.append(nodes)
                temp.append(local_min_score)
                if local_min_score < global_min_score:
                    global_min_score = local_min_score
                next_level.append(temp)

        compensation_factor = 1.08 - level * 0.03 / 64
        index = 0
        print(len(next_level))
        while 1:  # compulsory pruning
            if next_level[index][1] > compensation_factor * global_min_score:  # unworthy solution
                del next_level[index]  # trimming
                index = index - 1
            index = index + 1
            if index >= len(next_level):
                break
        ### next_level is trimmed beyond this point
        ## if next level is still too big at this point (> N), trimming down even more
        current_level = []  # reset current level
        # print (len(next_level))

        if len(next_level) > 250:  # optional pruning
            ind1 = random.sample(range(0, len(next_level) - 1), 250)
            for x in range(0, len(ind1)):
                current_level.append(next_level[ind1[x]][0])
        else:
            for i in range(0, len(next_level)):
                current_level.append(next_level[i][0])
        print(len(current_level))

        next_level = []
        level = level + 1

    #### at this point all node placements are done #############

    text_file = open("Output_core.txt", "w")
    text_file.write("%f\n\n" % hours)
    for i in range(0, len(current_level)):
        for j in range(0, 64):
            text_file.write("%s\t" % current_level[i][j])
        text_file.write("\n")
    text_file.close()

    for i in range(0, len(current_level)):
        t = []
        t.append(deepcopy(current_level[i]))
        t.append(links_only_tsv)
        current_level[i] = t
    # print(current_level[0][1])

    start_time = timeit.default_timer()
    ### start link perturbation from here ###
    links_placed = 0
    # out of 96 links, place 80 links PCBB style, rest to ensure full connectivity
    while links_placed < 80:
        global_min_score = 999
        elapsed = timeit.default_timer() - start_time
        hours = hours + float(elapsed / 3600)
        start_time = timeit.default_timer()
        for i in range(0, len(current_level)):
            nodes1 = current_level[i][0]
            successful_attempt = 0
            local_min_score = 999
            for j in range(0, 48):  # make 48 attempts but ensure atleast 2 should be successful attempts

                links = deepcopy(current_level[i][1])
                if (j >= 46 and successful_attempt < 2):  # desperate times call for desperate measures
                    while 1:
                        layer = random.randint(0, 3)
                        source = random.randint(0, 15)
                        dest = random.randint(0, 15)
                        if source == dest:  # cant place link between myself
                            continue
                        if links[layer * 16 + source][layer * 16 + dest] == 1:  # bad luck, link already exists
                            continue
                        successful_attempt = successful_attempt + 1
                        break
                else:
                    layer = int(j / 16)
                    source = random.randint(0, 15)
                    dest = random.randint(0, 15)
                    if source == dest:  # cant place link between myself
                        continue
                    if links[layer * 16 + source][layer * 16 + dest] == 1:  # bad luck, link already exists
                        continue
                    successful_attempt = successful_attempt + 1
                ## if you've reached here, you can place the link
                links[layer * 16 + source][layer * 16 + dest] = 1
                links[layer * 16 + dest][layer * 16 + source] = 1

                threshold_options = [1, 0.75, 0.5, 0.25, 0]
                for link_variety in range(0, 6):
                    if (link_variety == 0):  # create sw connectivity
                        virtual_links = create_links_sw2(nodes1, traffic, links, 80 - links_placed)
                    else:  # create partial/complete greedy connectivity
                        virtual_links = create_links_flexible_greedy2(nodes1, traffic, links,
                                                                      threshold_options[link_variety - 1],
                                                                      80 - links_placed)
                    # evaluate quality

                    mean, dev = calc_params_virtual(nodes1, virtual_links, traffic)
                    score = 0.7 * mean + 0.3 * dev

                    if score < local_min_score:
                        local_min_score = score

                temp = []
                temp.append(nodes1)
                temp.append(links)
                temp.append(local_min_score)
                if local_min_score < global_min_score:
                    global_min_score = local_min_score
                next_level.append(temp)

        compensation_factor = 1.08 - links_placed * 0.03 / 80
        index = 0
        #print(len(next_level))

        while 1:  # compulsory pruning
            if next_level[index][2] > compensation_factor * global_min_score:  # unworthy solution
                del next_level[index]  # trimming
                index = index - 1
            index = index + 1
            if index >= len(next_level):
                break
        ### next_level is trimmed beyond this point
        ## if next level is still too big at this point (> N), trimming down even more
        current_level = []  # reset current level
        # print (len(next_level))

        if len(next_level) > 250:  # optional pruning
            ind = random.sample(range(0, len(next_level) - 1), 250)
            for x in range(0, len(ind)):
                t = []
                t.append(next_level[ind[x]][0])
                t.append(next_level[ind[x]][1])
                current_level.append(t)
        else:
            for x in range(0, len(next_level)):
                t = []
                t.append(next_level[x][0])
                t.append(next_level[x][1])
                current_level.append(t)
        #print(len(current_level))

        next_level = []
        links_placed = links_placed + 1

    ###### candidates available in current_level array at this point, now ensure full-connectivity ##############

    file_num = 0
    for i in range(0, len(current_level)):
        links = current_level[i][1]
        nodes = current_level[i][0]

        hit = 0  # look for 5 hits
        attempts = 0  # will look 20 times max only
        while hit < 5 and attempts < 20:
            attempts = attempts + 1
            links_fully_connect = create_links_flexible_greedy2(nodes, traffic, links, 0.2, 16)
            edges_trial = []
            temp = []
            for i in range(0, 64):
                for j in range(0, 64):
                    temp.append(nodes[i])
                    temp.append(nodes[j])
                    if links_fully_connect[i][j] == 1:
                        temp.append(1)
                        # edges1.append(temp)
                    else:
                        temp.append(999)
                    edges_trial.append(temp)
                    # edges1.append(temp)
                    temp = []
            island = 0
            for i in range(0, 64):
                for j in range(i + 1, 64):
                    p = str(dijkstra(edges_trial, nodes[j], nodes[i]))
                    # print(p)
                    p = p.split(',')
                    cost_trial1 = int(p[0][1:])
                    if (cost_trial1 > 100):
                        island = 1
                        break
                if island == 1:
                    break
            if island == 0:  # success full-connect
                hit = hit + 1
                text_file = open("Output" + str(file_num) + ".txt", "w")
                file_num = file_num + 1
                text_file.write("%f\n\n" % hours)

                for j in range(0, 64):
                    if j % 16 == 0:
                        text_file.write("\n")
                    text_file.write("%s\t" % nodes[j])
                text_file.write("\n")
                for i in range(0, 64):
                    for j in range(0, 64):
                        text_file.write("%s\t" % links_fully_connect[i][j])
                    text_file.write("\n")
                text_file.close()



############################### extras ##########################
start_time = timeit.default_timer()
main(start_time)



