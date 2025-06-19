package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/GiorgosMarga/vanet_d_clustering/graph"
	"github.com/GiorgosMarga/vanet_d_clustering/utils"
)

var requiredFolders = []string{"cars_info", "graphviz", "sumo", "graph_images"}

func main() {
	var (
		d                int
		minClusterNumber int
		nodes            int
		graphPath        string
	)

	flag.IntVar(&d, "d", 2, "d")
	flag.IntVar(&minClusterNumber, "m", 3, "The minimum number of cluster members a cluster can have.")
	flag.StringVar(&graphPath, "g", "snapshots", "The path to the graph folder that contains the graphs.")
	flag.IntVar(&nodes, "n", 60, "The total number of nodes. Used to create a pool of nodes.")
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
	g, err := graph.NewGraph(minClusterNumber, d, nodes)
	if err != nil {
		log.Fatal(err)
	}
	for _, snapshot := range f {
		filename := utils.GetFileName(snapshot.Name())
		if err != nil {
			log.Fatal(err)
		}
		g.Log(fmt.Sprintf("------------%s----------\n", filename))
		if err := g.ParseGraphFile(fmt.Sprintf("%s/%s", graphPath, snapshot.Name()), "\n\n"); err != nil {
			fmt.Println(err)
			continue
		}
		g.DHCV()
		if err := g.PlotGraph(fmt.Sprintf("graphviz/%s.dot", filename), d); err != nil {
			log.Fatal("Plot error:", err)
		}
		if err := g.GenerateSUMOFile(fmt.Sprintf("sumo/%s.sumo", filename)); err != nil {
			log.Fatal("Generating SUMO:", err)
		}
		fmt.Printf("Filename: %s -> Connectivity: %d%% | Clusters: %d | AverageClusterSize: %d\n", filename, int(g.CalculateDensity()*100), g.NumOfClusters(), int(g.AverageClusterSize()))
	}

	for _, node := range g.Nodes {
		if node.Id%2 == 0 {
			fmt.Printf("Node [%d] (cluster head for %d rounds, messages %.2f) predict\n", node.Id, node.ClusterHeadRounds, float64(node.MessagesSent)/float64(node.TotalRounds))
			// node.Train()
			node.Predict()
		}
	}
}
