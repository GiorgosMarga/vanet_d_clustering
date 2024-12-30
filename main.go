package main

import (
	"fmt"
	"log"

	"math/rand"

	"github.com/GiorgosMarga/vanet_d_clustering/graph"
)

func main() {
	r := rand.New(rand.NewSource(100))
	var speed = map[int]float64{
		1:  3,
		2:  3.4,
		3:  2.6,
		4:  2.8,
		5:  3.3,
		6:  3.4,
		7:  3.2,
		8:  3,
		9:  2.9,
		10: 2.8,
		11: 3.2,
		12: 2.9,
		13: 3,
		14: 3.1,
	}
	_ = speed
	var position map[int][2]float64 = map[int][2]float64{
		1:  {5, 3},
		2:  {7, 4},
		3:  {8, 5},
		4:  {6, 2},
		5:  {8, 1},
		6:  {9, 4},
		7:  {10, 5},
		8:  {10, 3},
		9:  {4, 5},
		10: {2, 3},
		11: {2, 0},
		12: {1, 5},
		13: {0, 4},
		14: {0, 1},
	}

	g := graph.NewGraph(14)
	for i := range 14 { //float64(r.Intn(30)+20)
		n := graph.NewNode(i+1, position[i+1][0], position[i+1][1], float64(r.Intn(30)+20), fmt.Sprintf("%d.info", i+1))
		g.AddNode(n)
	}

	g.Nodes[1].AddNeighbor(g.Nodes[2])
	g.Nodes[1].AddNeighbor(g.Nodes[4])
	g.Nodes[1].AddNeighbor(g.Nodes[10])
	g.Nodes[2].AddNeighbor(g.Nodes[3])
	g.Nodes[3].AddNeighbor(g.Nodes[6])
	g.Nodes[6].AddNeighbor(g.Nodes[7])
	g.Nodes[6].AddNeighbor(g.Nodes[8])
	g.Nodes[4].AddNeighbor(g.Nodes[5])
	g.Nodes[9].AddNeighbor(g.Nodes[10])
	g.Nodes[10].AddNeighbor(g.Nodes[11])
	g.Nodes[10].AddNeighbor(g.Nodes[12])
	g.Nodes[11].AddNeighbor(g.Nodes[14])
	g.Nodes[12].AddNeighbor(g.Nodes[13])
	g.Nodes[13].AddNeighbor(g.Nodes[14])

	// for _, n := range g.Nodes {
	// 	fmt.Printf("%d: ", n.Id)
	// 	for _, cn := range n.GetdHopNeighs(2) {
	// 		fmt.Printf("%d ", cn.Id)
	// 	}
	// 	fmt.Println()
	// }
	g.DHCV(3)
	if err := g.PlotGraph("test.dot"); err != nil {
		log.Fatal(err)
	}
}
