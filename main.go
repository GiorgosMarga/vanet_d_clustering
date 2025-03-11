package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/GiorgosMarga/vanet_d_clustering/graph"
	"github.com/GiorgosMarga/vanet_d_clustering/utils"
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

	f, err := os.ReadDir("snapshots")
	if err != nil {
		log.Fatal(err)
	}

	for _, snapshot := range f {

		filename := utils.GetFileName(snapshot.Name())
		if err != nil {
			log.Fatal(err)
		}
		g := graph.NewGraph(minClusterNumber, d)
		if err := g.ReadFile(fmt.Sprintf("snapshots/%s", snapshot.Name()), "\n\n"); err != nil {
			fmt.Println(err)
			break
		}
		g.Print()

		g.DHCV()
		if err := g.PlotGraph(fmt.Sprintf("graphviz/%s.dot", filename), d); err != nil {
			log.Fatal(err)
		}
		if err := g.GenerateSUMOFile(fmt.Sprintf("sumo/%s.sumo", filename)); err != nil {
			log.Fatal(err)
		}
	}

}
