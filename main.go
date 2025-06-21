package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/GiorgosMarga/vanet_d_clustering/graph"
	"github.com/GiorgosMarga/vanet_d_clustering/gru"
	"github.com/GiorgosMarga/vanet_d_clustering/node"
	"github.com/GiorgosMarga/vanet_d_clustering/utils"
)

var requiredFolders = []string{"cars_info", "graphviz", "sumo", "graph_images"}

func main() {
	var (
		d                    int
		minClusterNumber     int
		nodes                int
		graphPath            string
		a                    float64
		b                    float64
		c                    float64
		trainSize            float64
		hiddenSize           int
		inputSize            int
		epochs               int
		batchSize            int
		patience             int
		parsevalValuesToSend int
		learningRate         float64
		sendWeightsPeriod    int
		rnpPercentage        float64
		parsevalError        int
		lossThreshold        float64
	)

	flag.IntVar(&d, "d", 2, "d")
	flag.IntVar(&minClusterNumber, "m", 3, "The minimum number of cluster members a cluster can have.")
	flag.StringVar(&graphPath, "g", "snapshots", "The path to the graph folder that contains the graphs.")
	flag.IntVar(&nodes, "n", 60, "The total number of nodes. Used to create a pool of nodes.")

	flag.Float64Var(&a, "a", 0.1, "a is the weight for distance")

	flag.Float64Var(&b, "b", 0.9, "b is the weight for speed")

	flag.Float64Var(&c, "c", 1.0, "c is the weight for degree")
	flag.Float64Var(&trainSize, "ts", 0.8, "ts is the trainsize for the gru config.")

	flag.IntVar(&hiddenSize, "hs", 16, "hs is the hidden size for gru config.")

	flag.IntVar(&inputSize, "is", 4, "is is the input size for gru config.")

	flag.IntVar(&epochs, "e", 50, "e is the epochs for gru config.")

	flag.IntVar(&batchSize, "bs", 3, "bs is the batch size for gru config.")

	flag.IntVar(&patience, "pt", 20, "pt is the patience for early stopping for gru config.")

	flag.IntVar(&parsevalValuesToSend, "pv2s", 5, "pv2s is the number of parseval values to send.")

	flag.Float64Var(&learningRate, "lr", 0.01, "lr is the learning rate for gru config.")
	flag.IntVar(&sendWeightsPeriod, "swp", 1, "swp is the send weights period.")
	flag.Float64Var(&rnpPercentage, "rnp", 1.0, "rnp is random node partitipation percentage.")
	flag.IntVar(&parsevalError, "pe", 0, "pe is the parseval error.")
	flag.Float64Var(&lossThreshold, "lth", 0.001, "lth is the loss threshold.")
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
	g, err := graph.NewGraph(minClusterNumber, d, nodes, &gru.GRUConfig{
		TrainSizePercentage: trainSize,
		HiddenStateSize:     hiddenSize,
		InputSize:           inputSize,
		Epochs:              epochs,
		BatchSize:           batchSize,
		Patience:            patience,
		LearningRate:        learningRate,
		LossThreshold:       lossThreshold,
	}, &node.AlgoConfig{
		A:                    a,
		B:                    b,
		C:                    c,
		ParsevalValuesToSend: parsevalValuesToSend,
		SendWeightPeriod:     sendWeightsPeriod,
		RnpPercentage:        rnpPercentage, // Random Node Partitipation percentage, if it is set to 1, all nodes participate
		ParsevalError:        float64(parsevalError),
	})
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
		if node.IsCH() {
			fmt.Printf("Node [%d] (cluster head for %d rounds, messages %.2f, total rounds: %d) predict\n", node.Id, node.ClusterHeadRounds, float64(node.MessagesSent)/float64(node.TotalRounds), node.TotalRounds)

			// node.Train()
			node.Predict()
		}
	}

}
