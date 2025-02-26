package main

import (
	"flag"
	"log"

	"github.com/GiorgosMarga/vanet_d_clustering/graph"
)

func main() {
	var (
		d                int
		minClusterNumber int
		graphPath        string
	)

	flag.IntVar(&d, "d", 3, "d")
	flag.IntVar(&minClusterNumber, "m", 3, "The minimum number of cluster members a cluster can have.")
	flag.StringVar(&graphPath, "g", "graph1.graph", "The path to the graph file.")
	flag.Parse()

	g := graph.NewGraph(minClusterNumber, d)
	if err := g.ReadFile(graphPath); err != nil {
		log.Fatal(err)
	}
	g.Print()

	g.DHCV()
	if err := g.PlotGraph("test.dot", d); err != nil {
		log.Fatal(err)
	}

}
