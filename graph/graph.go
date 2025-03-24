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
	"github.com/GiorgosMarga/vanet_d_clustering/utils"
)

var colors = []string{
	"red", "blue", "lightblue", "purple", "green", "brown", "pink", "orange", "burlywood",
	"cyan", "magenta", "teal", "indigo", "violet", "gold", "silver", "gray", "olive", "navy", "maroon", "coral", "turquoise", "salmon", "plum", "orchid", "sienna",
	"khaki", "lavender", "beige", "crimson", "chartreuse", "aqua", "fuchsia", "tan", "tomato",
	"peru", "slateblue", "darkgreen", "goldenrod", "darkred", "lightgreen", "deeppink", "skyblue", "chocolate",
}

type Graph struct {
	Nodes            map[int]*node.Node
	NumOfNodes       int
	wg               sync.WaitGroup
	clusters         map[int][]int
	minClusterNumber int
	d                int
	f                *os.File
	links            int
}

func NewGraph(minClusterNumber, d int) (*Graph, error) {
	return &Graph{
		Nodes:            make(map[int]*node.Node),
		wg:               sync.WaitGroup{},
		clusters:         make(map[int][]int),
		minClusterNumber: minClusterNumber,
		d:                d,
		links:            0,
	}, nil
}
func (g *Graph) ResetGraph() {
	g.clusters = make(map[int][]int)
	for _, n := range g.Nodes {
		n.ResetNode()
	}
	g.links = 0
}
func (g *Graph) ParseGraphFile(path string, splitter string) error {
	g.ResetGraph()
	f, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	filename := utils.GetFileName(path)

	sf := string(f)
	sf = strings.Replace(sf, "\r\n", "\n", -1)
	splitted := strings.Split(sf, splitter)
	if len(splitted) != 2 {
		return fmt.Errorf("failed to split file correctly (%s)", path)
	}

	nodes, connections := string(splitted[0]), string(splitted[1])[:len(string(splitted[1]))-1]
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

		nodeId := t[0]
		if n, ok := g.Nodes[int(nodeId)]; ok {

			n.UpdateNode(t[1], t[2], t[3])
			continue
		}

		node := node.NewNode(int(t[0]), g.d, t[1], t[2], t[3], fmt.Sprintf("./cars_info/%s_%d.info", filename, int(t[0])))
		g.AddNode(node)
	}

	for _, connection := range strings.Split(connections, "\n") {
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
		g.links++
	}

	for _, n := range g.Nodes {
		if len(n.DHopNeighbors) == 0 {
			// this node doesnt exist in the snapshot and should be removed
			delete(g.Nodes, n.Id)
		}
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

func (g *Graph) Log(s string) error {
	_, err := g.f.WriteString(s)
	return err
}

func (g *Graph) NumOfClusters() int {
	return len(g.clusters)
}

func (g *Graph) AverageClusterSize() float64 {
	var clusterSizes float64 = 0
	for _, c := range g.clusters {
		clusterSizes += float64(len(c))
	}
	return clusterSizes / float64(g.NumOfClusters())
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
		go n.Start(ctx, g.d)
		wg.Add(1)
		// go func() {
			
		// 	wg.Done()
		// }()
		go func() {
			n.RelativeMax(g.d)
			wg.Done()
		}()
	}
	wg.Wait()

	g.formClusters()
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
					n.PCH[g.d] = g.Nodes[node.Id]
					g.Log(fmt.Sprintf("[%d]: Passing through %d(%d) -> new CH: %d\n", n.Id, node.Id, node.PCH[g.d].Id, g.Nodes[node.Id].Id))
					continue mainLoop
				}
			}
		}
		break
	}
	g.formClusters()
	g.Log(fmt.Sprintln(g.clusters))

	g.Log(fmt.Sprintln("Exception 2"))
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
					g.Log(fmt.Sprintf("Exception2: [%d] can't satisfy distance with %d\n", currNode.Id, newCh.Id))
					ch.PCH[g.d] = ch
					g.formClusters()
					continue exceptionLoop
				}
			}
			// CMs join the new cluster
			g.Log(fmt.Sprintf("[%d]: CH did not select itself as PCH (%d)\n", chId, ch.PCH[g.d].Id))
			for _, cm := range cluster {
				n := g.Nodes[cm]
				n.PCH[g.d] = ch.PCH[g.d]
			}
			delete(g.clusters, chId)
		}
	}
	g.formClusters()
	g.Log(fmt.Sprintln(g.clusters))
	g.Log(fmt.Sprintln("Exception 3"))
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
					g.Log(fmt.Sprintf("[%d]: CH node without CMs -> New CH %d\n", n.Id, n.PCH[g.d].PCH[g.d].Id))
					// find new pch's cluster and add new member
					pch := n.PCH[g.d].PCH[g.d].Id
					g.clusters[pch] = append(g.clusters[pch], n.Id)
					continue exception3Loop
				}
			}
			g.Log(fmt.Sprintf("[%d]: CH node without CMs -> Can't satisfy distance d\n", n.Id))
		}
	}
	g.formClusters()
	g.Log(fmt.Sprintln(g.clusters))
	g.Log(fmt.Sprintln("Merge clusters"))
	// merge clusters
mergeLoop:
	for ch, cluster := range g.clusters {
		if len(cluster) <= g.minClusterNumber {
			currCh := g.Nodes[ch]
			g.Log(fmt.Sprintf("Found cluster with min members: %d\n", ch))
			bestCh := math.MaxFloat64
			var bestChNode *node.Node
			// TODO: search neighbor to d to find CHs
			// checks all CHs that have distance <= d and calculates the relative mobility
			for newPotentialCh := range g.clusters {
				if ch != newPotentialCh {
					chNode := g.Nodes[newPotentialCh]
					pathTo := g.Nodes[ch].FindPath(chNode)
					if len(pathTo) <= g.d && len(pathTo) > 0 {
						mob := g.Nodes[ch].GetRelativeMobility(chNode.Velocity, chNode.PosX, chNode.PosY, chNode.Degree(), chNode.PCI())
						g.Log(fmt.Sprintf("Comparing %d->%d %f\n", ch, newPotentialCh, mob))
						if mob < bestCh {
							bestCh = mob
							bestChNode = chNode
						}
					}
				}
			}

			// if there is a better ch node and if all CMs are in distance <= d merge
			// else keep cluster as it is
			if bestChNode != nil {
				pathToPotentialCh := currCh.FindPath(bestChNode)
				for _, cm := range pathToPotentialCh {
					if cm.PCH[g.d] != bestChNode {
						g.Log(fmt.Sprintf("[%d]: passing through another cluster: %d\n", currCh.Id, cm.PCH[g.d].Id))
						// passing through another cluster, dont merge
						continue mergeLoop
					}
				}
				for _, nodeId := range cluster {
					currNode := g.Nodes[nodeId]
					if currNode != currCh {
						p := currNode.FindPath(bestChNode)
						if len(p) > g.d {
							g.Log(fmt.Sprintf("[%d]: can't satisfy d\n", currNode.Id))
							// one cluster member cant join new cluster because of distance, cluster remains
							continue mergeLoop
						}
					}
				}
				for idx := range cluster {
					g.Log(fmt.Sprintf("[%d]: New cm %d\n", bestChNode.Id, cluster[idx]))
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

	for _, n := range g.Nodes {
		n.SendWeights()
	}

}

func (g *Graph) CalculateDensity() float32 {
	return float32((2 * g.links)) / float32((len(g.Nodes) * (len(g.Nodes) - 1)))
}
func (g *Graph) formClusters() {
	g.clusters = make(map[int][]int)

	for _, n := range g.Nodes {
		ch := n.PCH[g.d]
		if ch == nil {
			fmt.Printf("%+v\n", n)
			panic(fmt.Sprintf("[%d]: nil ch: %+v\n", n.Id, n.PCH))
		}
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
		for _, neighbor := range n.DHopNeighbors {
			if neighbor.Id > n.Id {
				f.WriteString(fmt.Sprintf("\t%d -- %d;\n", n.Id, neighbor.Id))
			}
		}
	}

	f.WriteString("}")
	return nil
}
