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

	flag.IntVar(&d, "d", 2, "d")
	flag.IntVar(&minClusterNumber, "m", 3, "The minimum number of cluster members a cluster can have.")
	flag.StringVar(&graphPath, "g", "snapshots", "The path to the graph folder that contains the graphs.")
	flag.Parse()

	f, err := os.ReadDir(graphPath)
	if err != nil {
		log.Fatal(err)
	}

	// TODO: add filename
	g, err := graph.NewGraph(minClusterNumber, d)
	if err != nil {
		log.Fatal(err)
	}
	for _, snapshot := range f {
		filename := utils.GetFileName(snapshot.Name())
		if err != nil {
			log.Fatal(err)
		}

		if err := g.ParseGraphFile(fmt.Sprintf("snapshots/%s", snapshot.Name()), "\n\n"); err != nil {
			fmt.Println(err)
			continue
		}

		g.DHCV()
		if err := g.PlotGraph(fmt.Sprintf("graphviz/%s.dot", filename), d); err != nil {
			log.Fatal(err)
		}
		if err := g.GenerateSUMOFile(fmt.Sprintf("sumo/%s.sumo", filename)); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Filename: %s/%s -> Connectivity: %f\n", graphPath, filename, g.CalculateDensity())
		fmt.Printf("Filename: %s/%s -> Clusters: %d\n", graphPath, filename, g.NumOfClusters())
		fmt.Printf("Filename: %s/%s -> Average Cluster Size: %f\n", graphPath, filename, g.AverageClusterSize())
	}

}
