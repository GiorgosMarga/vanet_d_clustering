package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/GiorgosMarga/vanet_d_clustering/graph"
	"github.com/GiorgosMarga/vanet_d_clustering/utils"
)

var requiredFolders = []string{"cars_info", "graphviz", "sumo"}

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

	for _, folder := range requiredFolders {
		if _, err := os.Stat(folder); os.IsNotExist(err) {
			err = os.Mkdir(folder, 0777)
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("Successfully created folder: (%s)\n", folder)
		}
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
		fmt.Printf("Filename: %s -> Connectivity: %f\n", filename, g.CalculateDensity())
		fmt.Printf("Filename: %s -> Clusters: %d\n", filename, g.NumOfClusters())
		fmt.Printf("Filename: %s -> Average Cluster Size: %f\n", filename, g.AverageClusterSize())
	}

}
