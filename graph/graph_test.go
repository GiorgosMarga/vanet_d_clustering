package graph

import (
	"context"
	"fmt"
	"testing"

	"github.com/GiorgosMarga/vanet_d_clustering/node"
)

func TestDistributedBFS(t *testing.T) {
	// wg := &sync.WaitGroup{}
	g, err := NewGraph(3, 2, 60)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.ParseGraphFile("../snapshots/cars_65.txt", "\n\n"); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	defer cancel()

	for _, n := range g.Nodes {
		go n.Start(ctx)
	}

	startNode, ok := g.Nodes[8]

	if !ok {
		t.FailNow()
	}

	fmt.Println(node.PrintPath(startNode.FindPath(g.Nodes[45])))
	fmt.Println(startNode.DistributedBFS(45))

}
