package graph

import (
	"fmt"
	"os"
	"sync"

	"github.com/GiorgosMarga/vanet_d_clustering/node"
)

var colors = []string{"red", "blue", "lightblue", "purple", "yellow", "green", "brown", "pink", "black", "orange", "burlywood", "darkblue"}

type Graph struct {
	Size       int
	Nodes      map[int]*node.Node
	NumOfNodes int
	wg         sync.WaitGroup
	clusters   map[int][]int
}

func NewGraph(size int) *Graph {
	return &Graph{
		Size:     size,
		Nodes:    make(map[int]*node.Node),
		wg:       sync.WaitGroup{},
		clusters: make(map[int][]int),
	}
}

func (g *Graph) AddNode(n *node.Node) {
	g.NumOfNodes++
	g.Nodes[n.Id] = n
}
func (g *Graph) Print() {
	for _, cn := range g.Nodes {
		fmt.Printf("[%d]: ", cn.Id)
		for _, n := range cn.Neighbors {
			fmt.Printf("%d ", n.Id)
		}
		fmt.Println("Degree: ", cn.Degree())
	}
}

func (g *Graph) PrintCH(d int) {
	for _, cn := range g.Nodes {
		fmt.Printf("[%d]: CH: %d\n", cn.Id, cn.PCH[d].Id)
	}
}

func (g *Graph) PrintCHS() {
	for _, cn := range g.Nodes {
		fmt.Printf("[%d]: ", cn.Id)
		fmt.Println(cn.PCH)
	}
}

func (g *Graph) DHCV(d int) {
	// 	wg := sync.WaitGroup{}
	// 	ctx, cancel := context.WithCancel(context.Background())
	// 	defer func() {
	// 		cancel()
	// 		wg.Wait()
	// 	}()
	// 	for _, n := range g.Nodes {
	// 		wg.Add(1)
	// 		go func() {
	// 			n.Start(ctx)
	// 			wg.Done()
	// 		}()
	// 	}
	// 	g.initialization()
	// 	for i := range d {
	// 		for _, n := range g.Nodes {
	// 			g.wg.Add(1)
	// 			go func() {
	// 				n.relativeMax(i)
	// 				g.wg.Done()
	// 				// fmt.Printf("[%d]: Finished Round %d\n", n.Id, i+1)
	// 			}()
	// 		}
	// 		g.wg.Wait()
	// 	}

	// 	time.Sleep(500 * time.Millisecond)
	// 	fmt.Println(g.Nodes[4].isCH(d))
	// 	g.Nodes[4].printPCH(d)
	// 	g.Nodes[5].printPCH(d)

	// 	for _, n := range g.Nodes {
	// 		ch := n.PCH[d]
	// 		if _, ok := g.clusters[ch.Id]; !ok {
	// 			g.clusters[ch.Id] = make([]int, 0)
	// 		}
	// 		g.clusters[ch.Id] = append(g.clusters[ch.Id], n.Id)
	// 	}

	// 	// exception 3
	// 	for ch, cms := range g.clusters {
	// 		if len(cms) == 1 {
	// 			node.Node := g.Nodes[ch]
	// 			// cluster has onle 1 member (the cluster head)
	// 			cnn1 := g.Nodes[ch].CNN[1]
	// 			if cnn1.isCH(d) {
	// 				g.Nodes[ch].PCH[d] = cnn1
	// 			} else {
	// 				g.Nodes[ch].PCH[d] = cnn1.PCH[d]
	// 			}
	// 			node.Node.f.WriteString(fmt.Sprintf("No CMS. Chose new clusterhead %d\n", node.Node.PCH[d].Id))
	// 		}
	// 	}
	// 	// exception 1
	// mainLoop:
	// 	for _, n := range g.Nodes {
	// 		n.f.WriteString(n.writeCNN(d))
	// 		n.f.WriteString(n.writePCH(d))
	// 		ch := n.PCH[d]
	// 		pathToCH := n.findPath(ch)
	// 		if n.Id == 3 {
	// 			printPath(pathToCH)
	// 		}
	// 		for _, node.Node := range pathToCH {
	// 			if node.Node.PCH[d] != ch && ch != node.Node {
	// 				n.f.WriteString(fmt.Sprintf("Passing through %d(%d)\n", node.Node.Id, node.Node.PCH[d].Id))
	// 				// node.Node passes through another cluster
	// 				for idx := d - 1; idx >= 0; idx-- {
	// 					pch := n.PCH[idx]
	// 					if _, ok := n.Neighbors[pch.Id]; ok {
	// 						n.PCH[d] = pch
	// 						n.f.WriteString(fmt.Sprintf("Setting new CH %d %d\n", node.Node.Id, pch.Id))
	// 						continue mainLoop
	// 					}
	// 				}
	// 				n.f.WriteString(fmt.Sprintf("Didn't find a better CH. Old CH: %d\tNew CH: %d\n", n.PCH[d].Id, node.Node.PCH[d].Id))
	// 				n.printPCH(d)
	// 				n.PCH[d] = n.CNN[1]
	// 				break
	// 			}
	// 		}
	// 	}
	// 	g.clusters = make(map[int][]int)
	// 	for _, n := range g.Nodes {
	// 		ch := n.PCH[d]
	// 		if _, ok := g.clusters[ch.Id]; !ok {
	// 			g.clusters[ch.Id] = make([]int, 0)
	// 		}
	// 		g.clusters[ch.Id] = append(g.clusters[ch.Id], n.Id)
	// 	}

	// fmt.Println(g.clusters)
}

// func (g *Graph) initialization() {
// 	for _, cn := range g.Nodes {
// 		cn.f.WriteString(fmt.Sprintf("CNN[0]=%d\tPCH[0]=%d\n", cn.Id, cn.Id))
// 		cn.CNN[0] = cn
// 		cn.PCH[0] = cn
// 	}
// }

func (g *Graph) PlotGraph(filename string) error {
	f, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}

	defer f.Close()

	f.WriteString("graph RandomGraph {\n\tgraph [layout=neato, splines=true, overlap=false];\n\n")

	for _, n := range g.Nodes {
		ch := n.PCH[3]
		color := colors[ch.Id%len(colors)]
		f.WriteString(fmt.Sprintf("\t%d [pos=\"%f,%f!\" fillcolor=\"%s\" style=\"filled\" label=\"%d,%.2f\"];\n", n.Id, n.PosX, n.PosY, color, n.Id, n.Velocity))
	}

	for _, n := range g.Nodes {
		for _, neighbor := range n.Neighbors {
			if neighbor.Id > n.Id {
				f.WriteString(fmt.Sprintf("\t%d -- %d;\n", n.Id, neighbor.Id))
			}
		}
	}

	f.WriteString("}")
	return nil
}

// func (g *Graph) getClusterMembers(n *node.Node, d int) []*node.Node {
// 	cms := make([]*node.Node, 0)
// 	for _, node.Node := range g.Nodes {
// 		if node.Node.PCH[d] == n.PCH[d] && node.Node != n {
// 			cms = append(cms, node.Node)
// 		}
// 	}
// 	return cms
// }
