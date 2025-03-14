package graph

import (
	"context"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GiorgosMarga/vanet_d_clustering/node"
)

var colors = []string{"red", "blue", "lightblue", "purple", "yellow", "green", "brown", "pink", "orange", "burlywood", "darkblue"}

type Graph struct {
	Nodes            map[int]*node.Node
	NumOfNodes       int
	wg               sync.WaitGroup
	clusters         map[int][]int
	minClusterNumber int
	d                int
}

func NewGraph(minClusterNumber, d int) *Graph {
	return &Graph{
		Nodes:            make(map[int]*node.Node),
		wg:               sync.WaitGroup{},
		clusters:         make(map[int][]int),
		minClusterNumber: minClusterNumber,
		d:                d,
	}
}

func (g *Graph) ReadFile(path string, splitter string) error {
	f, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	sf := string(f)
	sf = strings.Replace(sf, "\r\n", "\n", -1)
	splitted := strings.Split(sf, splitter)
	if len(splitted) != 2 {
		return fmt.Errorf("failed to split file correctly (%s)", path)
	}

	nodes, connections := string(splitted[0]), string(splitted[1])
	for _, n := range strings.Split(nodes, "\n") {
		splitted := strings.Split(strings.TrimSpace(n), " ")
		if len(splitted) != 4 {
			return fmt.Errorf("failed to split nodes correctly (%s)", path)
		}

		t := make([]float64, 4)
		var err error
		for idx := range splitted {
			t[idx], err = strconv.ParseFloat(splitted[idx], 64)
			if err != nil {
				return err
			}
		}

		node := node.NewNode(int(t[0]), g.d, t[1], t[2], t[3], fmt.Sprintf("./cars_info/%s_%d", path, int(t[0])))
		g.AddNode(node)
	}

	for _, connection := range strings.Split(connections, "\n") {
		fmt.Println(connection)
		splitted := strings.Split(strings.TrimSpace(connection), "-")
		if len(splitted) != 2 {
			return fmt.Errorf("failed to split connections correctly %s", connection)
		}

		node1, node2 := splitted[0], splitted[1]

		node1Id, err := strconv.Atoi(node1)
		if err != nil {
			return err
		}
		node2Id, err := strconv.Atoi(node2)
		if err != nil {
			return err
		}
		n1, n2 := g.Nodes[node1Id], g.Nodes[node2Id]
		n1.AddNeighbor(n2)

	}
	return nil
}

func (g *Graph) AddNode(n *node.Node) {
	g.NumOfNodes++
	g.Nodes[n.Id] = n
}
func (g *Graph) Print() {
	for _, cn := range g.Nodes {
		fmt.Printf("[%d]: ", cn.Id)
		fmt.Println("Degree: ", cn.Degree())
	}
}

func (g *Graph) PrintCH() {
	for _, cn := range g.Nodes {
		fmt.Printf("[%d]: CH: %d\n", cn.Id, cn.PCH[g.d].Id)
	}
}

func (g *Graph) PrintCHS() {
	for _, cn := range g.Nodes {
		fmt.Printf("[%d]: ", cn.Id)
		fmt.Println(cn.PCH)
	}
}

func (g *Graph) GenerateSUMOFile(filename string) error {
	f, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	for chId, cluster := range g.clusters {
		ln := fmt.Sprintf("%d %s ", chId, colors[chId%len(colors)])
		for idx := range cluster {
			if idx == len(cluster)-1 {
				ln += fmt.Sprintf("%d", cluster[idx])
			} else {
				ln += fmt.Sprintf("%d,", cluster[idx])
			}
		}
		f.WriteString(ln)
		f.WriteString("\n")
	}
	return nil
}

func (g *Graph) DHCV() {
	wg := sync.WaitGroup{}
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	for _, n := range g.Nodes {
		go n.Beacon(ctx)
	}

	for _, n := range g.Nodes {
		wg.Add(1)
		go func() {
			n.Start(ctx, g.d)
			wg.Done()
		}()
	}

	for _, n := range g.Nodes {
		go n.RelativeMax(g.d)
	}
	wg.Wait()

	g.formClusters()
	fmt.Println(g.clusters)

	for _, n := range g.Nodes {
		wg.Add(1)
		go func() {
			n.Exceptions(g.d)
			wg.Done()
		}()
	}
	wg.Wait()
	// exception 1
mainLoop:
	for {
		for _, n := range g.Nodes {
			ch := n.PCH[g.d]
			pathToCH := n.FindPath(ch)
			// for each node in the path, if the node is a CH then join
			// the CH might not have select itself as CH
			for _, node := range pathToCH {
				if node.PCH[g.d] != n.PCH[g.d] && node != ch {
					n.PCH[g.d] = n.PCH[g.d-1]
					fmt.Printf("[%d]: Passing through %d(%d) -> new CH: %d\n", n.Id, node.Id, node.PCH[g.d].Id, g.Nodes[node.Id].Id)
					continue mainLoop
					// if _, ok := g.clusters[node.Id]; ok && node != ch {
					// n passes through another cluster
					// if len(n.FindPath(node.PCH[g.d])) <= g.d+1 {
					// 	fmt.Printf("[%d]: Passing through %d(%d) -> new CH: %d\n", n.Id, node.Id, node.PCH[g.d].Id, g.Nodes[node.Id].Id)
					// 	// n.PCH[g.d] = node.PCH[g.d]
					// 	// n.PCH[g.d] = g.Nodes[node.Id]
					// 	// n.PCH[g.d] = n.PCH[g.d-1]

					// 	i = 0
					// 	continue mainLoop
					// }
				}
			}
		}
		break
	}
	g.formClusters()
	fmt.Println(g.clusters)

	fmt.Println("Exception 2")
	// exception 2
exceptionLoop:
	for chId, cluster := range g.clusters {
		// ch did not select itself as ch
		ch := g.Nodes[chId]
		if !slices.Contains(cluster, chId) {
			newCh := ch.PCH[g.d]
			for _, n := range cluster {
				currNode := g.Nodes[n]
				if len(currNode.FindPath(newCh)) >= g.d {
					// there is a node in the cluster that cant satisfy distance
					// in this case nodes form a cluster
					fmt.Printf("Exception2: [%d] can't satisfy distance with %d\n", currNode.Id, newCh.Id)
					ch.PCH[g.d] = ch
					g.formClusters()
					continue exceptionLoop
				}
			}
			// CMs join the new cluster
			fmt.Printf("[%d]: CH did not select itself as PCH (%d)\n", chId, ch.PCH[g.d].Id)
			for _, cm := range cluster {
				n := g.Nodes[cm]
				n.PCH[g.d] = ch.PCH[g.d]
			}
			delete(g.clusters, chId)
		}
	}
	g.formClusters()
	fmt.Println(g.clusters)
	fmt.Println("Exception 3")
	// exception 3
exception3Loop:
	for _, cluster := range g.clusters {
		if len(cluster) == 1 {
			// CH with no CMs
			n := g.Nodes[cluster[0]]
			for i := g.d - 1; i >= 0; i-- {
				potentialCH := n.CNN[i].PCH[g.d]
				if len(n.FindPath(potentialCH)) <= g.d {
					n.PCH[g.d] = potentialCH
					fmt.Printf("[%d]: CH node without CMs -> New CH %d\n", n.Id, n.PCH[g.d].PCH[g.d].Id)
					// find new pch's cluster and add new member
					pch := n.PCH[g.d].PCH[g.d].Id
					g.clusters[pch] = append(g.clusters[pch], n.Id)
					continue exception3Loop
				}
			}
			fmt.Printf("[%d]: CH node without CMs -> Can't satisfy distance d\n", n.Id)
		}
	}
	g.formClusters()
	fmt.Println(g.clusters)
	fmt.Println("Merge clusters")
	// merge clusters
mergeLoop:
	for ch, cluster := range g.clusters {
		if len(cluster) <= g.minClusterNumber {
			currCh := g.Nodes[ch]
			fmt.Printf("Found cluster with min members: %d\n", ch)
			bestCh := math.MaxFloat64
			var bestChNode *node.Node
			// TODO: search neighbor to d to find CHs
			// checks all CHs that have distance <= d and calculates the relative mobility
			for newCh := range g.clusters {
				if ch != newCh {
					chNode := g.Nodes[newCh]
					pathTo := g.Nodes[ch].FindPath(chNode)
					if len(pathTo) <= g.d && len(pathTo) > 0 {
						mob := g.Nodes[ch].GetRelativeMobility(chNode.Velocity, chNode.PosX, chNode.PosY, chNode.Degree())
						fmt.Printf("Comparing %d->%d %f\n", ch, newCh, mob)
						if mob < bestCh {
							bestCh = mob
							bestChNode = chNode
						}
					}
				}
			}

			// if there is a better ch node, then if all CMs are in distance <= d, merge else keep cluster
			if bestChNode != nil {

				for _, nodeId := range cluster {
					currNode := g.Nodes[nodeId]
					if currNode != currCh {
						p := currNode.FindPath(bestChNode)
						node.PrintPath(p)
						if len(p) > g.d {
							fmt.Printf("[%d]: can't satisfy d\n", currNode.Id)
							// one cluster member cant join new cluster because of distance, cluster remains
							continue mergeLoop
						}
					}
				}
				for idx := range cluster {
					fmt.Printf("[%d]: New cm %d\n", bestChNode.Id, cluster[idx])
					g.Nodes[cluster[idx]].PCH[g.d] = bestChNode
					if _, ok := g.clusters[bestChNode.Id]; ok {
						g.clusters[bestChNode.Id] = append(g.clusters[bestChNode.Id], cluster[idx])
					}
				}
				delete(g.clusters, currCh.Id)
			}
		}
	}
	g.formClusters()
	fmt.Println(g.clusters)

}

func (g *Graph) formClusters() {
	g.clusters = make(map[int][]int)

	for _, n := range g.Nodes {
		ch := n.PCH[g.d]
		if _, ok := g.clusters[ch.Id]; !ok {
			g.clusters[ch.Id] = make([]int, 0)
		}
		g.clusters[ch.Id] = append(g.clusters[ch.Id], n.Id)
	}
}

func (g *Graph) PlotGraph(filename string, d int) error {
	f, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}

	defer f.Close()

	f.WriteString("graph RandomGraph {\n\tgraph [layout=neato, splines=true, overlap=false];\n\n")

	for _, n := range g.Nodes {
		ch := n.PCH[d]
		color := colors[ch.Id%len(colors)]
		f.WriteString(fmt.Sprintf("\t%d [pos=\"%f,%f!\" fillcolor=\"%s\" style=\"filled\" label=\"%d,%.2f\"];\n", n.Id, n.PosX, n.PosY, color, n.Id, n.Velocity))
	}

	for _, n := range g.Nodes {
		for _, neighbor := range n.DHopNeighbors[1] {
			if neighbor.Id > n.Id {
				f.WriteString(fmt.Sprintf("\t%d -- %d;\n", n.Id, neighbor.Id))
			}
		}
	}

	f.WriteString("}")
	return nil
}
