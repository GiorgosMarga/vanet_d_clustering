package graph

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/GiorgosMarga/vanet_d_clustering/node"
	"github.com/GiorgosMarga/vanet_d_clustering/utils"
)

func TestDistributedBFS(t *testing.T) {
	// wg := &sync.WaitGroup{}
	g, err := NewGraph(3, 2, 60)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.ParseGraphFile("../snapshots/cars_15.txt", "\n\n"); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	defer cancel()

	for _, n := range g.Nodes {
		go n.Start(ctx)
	}

	startNode, ok := g.Nodes[3]

	if !ok {
		t.FailNow()
	}

	fmt.Println(node.PrintPath(startNode.FindPath(g.Nodes[5])))
	fmt.Println(startNode.DistributedBFS(5))

}

func TestException1(t *testing.T) {
	g, err := NewGraph(4, 2, 5)
	if err != nil {
		t.Fatal(err)
	}

	node0 := g.PoolOfNodes[0]
	node1 := g.PoolOfNodes[1]
	node2 := g.PoolOfNodes[2]
	node3 := g.PoolOfNodes[3]
	node4 := g.PoolOfNodes[4]

	g.Nodes[0] = node0
	g.Nodes[1] = node1
	g.Nodes[2] = node2
	g.Nodes[3] = node3
	g.Nodes[4] = node4

	node0.AddNeighbor(node2)
	node1.AddNeighbor(node2)
	node2.AddNeighbor(node3)
	node3.AddNeighbor(node4)

	node0.PCH[0] = node0
	node0.CNN[0] = node0
	node1.PCH[0] = node1
	node1.CNN[0] = node1
	node2.PCH[0] = node2
	node2.CNN[0] = node2
	node3.PCH[0] = node3
	node3.CNN[0] = node3
	node4.PCH[0] = node4
	node4.CNN[0] = node4

	node0.PCH[1] = node2
	node0.CNN[1] = node2
	node1.PCH[1] = node2
	node1.CNN[1] = node2
	node2.PCH[1] = node2
	node2.CNN[1] = node2
	node3.PCH[1] = node4
	node3.CNN[1] = node4
	node4.PCH[1] = node4
	node4.CNN[1] = node4

	node0.PCH[2] = node3
	node0.CNN[2] = node3

	node1.PCH[2] = node2
	node1.CNN[2] = node2
	node2.PCH[2] = node2
	node2.CNN[2] = node2
	node3.PCH[2] = node4
	node3.CNN[2] = node4
	node4.PCH[2] = node4
	node4.CNN[2] = node4

	g.formClusters()
	g.Print()

	fmt.Println(g.clusters)

	ctx, cancel := context.WithCancel(context.Background())
	for _, n := range g.Nodes {
		go n.Start(ctx)
	}

	for _, n := range g.Nodes {
		go n.Exception1()
	}

	time.Sleep(5 * time.Second)
	cancel()

	g.formClusters()
	fmt.Println(g.clusters)

}

func TestExceptions(t *testing.T) {
	wg := &sync.WaitGroup{}
	g, err := NewGraph(4, 2, 60)
	if err != nil {
		t.Fatal(err)
	}

	if err := g.ParseGraphFile("../snapshots/cars_10.txt", "\n\n"); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	beaconCTX, beaconCancel := context.WithCancel(context.Background())

	defer cancel()
	defer beaconCancel()
	for _, n := range g.Nodes {
		go n.Beacon(beaconCTX)
		go n.Start(ctx)
		wg.Add(1)
		go func() {
			n.RelativeMax(2)
			wg.Done()
		}()
	}
	wg.Wait()
	fmt.Println("Finished Relative Max")
	for _, n := range g.Nodes {
		wg.Add(1)
		go func() {
			if err := n.Exceptions(); err != nil {
				t.Error(err)
			}
			wg.Done()
		}()
	}
	wg.Wait()

	g.formClusters()
	fmt.Println(g.clusters)
}

func TestDHCVWithExceptions(t *testing.T) {
	f, err := os.ReadDir("../snapshots")
	if err != nil {
		log.Fatal(err)
	}

	g, err := NewGraph(4, 2, 60)
	if err != nil {
		log.Fatal(err)
	}
	d := 2

	graphPath := "../snapshots"
	wg := &sync.WaitGroup{}
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
		ctx, cancel := context.WithCancel(context.Background())
		beaconCTX, beaconCancel := context.WithCancel(context.Background())

		for _, n := range g.Nodes {
			go n.Beacon(beaconCTX)
			go n.Start(ctx)
			wg.Add(1)
			go func() {
				n.RelativeMax(2)
				wg.Done()
			}()
		}
		wg.Wait()
		fmt.Println("Finished Relative Max")
		for _, n := range g.Nodes {
			wg.Add(1)
			go func() {
				if err := n.Exceptions(); err != nil {
					fmt.Println(err)
				}
				wg.Done()
			}()
		}
		wg.Wait()
		g.formClusters()

		g.Log(fmt.Sprintf("Clusters: %v\n", g.clusters))

		if err := g.PlotGraph(fmt.Sprintf("../graphviz/%s.dot", filename), d); err != nil {
			log.Fatal("Plot error:", err)
		}
		if err := g.GenerateSUMOFile(fmt.Sprintf("../sumo/%s.sumo", filename)); err != nil {
			log.Fatal("Generating SUMO:", err)
		}
		fmt.Printf("Filename: %s -> Connectivity: %d%% | Clusters: %d | AverageClusterSize: %d\n", filename, int(g.CalculateDensity()*100), g.NumOfClusters(), int(g.AverageClusterSize()))

		cancel()
		beaconCancel()
	}
}
