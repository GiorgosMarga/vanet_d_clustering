package graph

import (
	"context"
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	"github.com/GiorgosMarga/vanet_d_clustering/messages"
)

var colors = []string{"red", "blue", "lightblue", "purple", "yellow", "green", "brown", "pink", "black"}

const (
	a = 0.9
	b = 0.1
)

type Node struct {
	Id           int
	Neighbors    map[int]*Node
	Velocity     float64
	PosX         float64
	PosY         float64
	CNN          []*Node
	PCH          []*Node
	msgChan      chan *messages.Message
	neighChans   map[int]chan *messages.Message
	internalChan chan *messages.BeaconMessage
	mtx          sync.Mutex
}

type Graph struct {
	Size       int
	Nodes      map[int]*Node
	NumOfNodes int
	wg         sync.WaitGroup
	clusters   map[int][]int
}

func NewGraph(size int) *Graph {
	return &Graph{
		Size:     size,
		Nodes:    make(map[int]*Node),
		wg:       sync.WaitGroup{},
		clusters: make(map[int][]int),
	}
}
func NewNode(id int, posx, posy, velocity float64) *Node {
	return &Node{
		Id:           id,
		Neighbors:    make(map[int]*Node),
		Velocity:     velocity,
		PosX:         posx,
		PosY:         posy,
		CNN:          make([]*Node, 5),
		PCH:          make([]*Node, 5),
		msgChan:      make(chan *messages.Message, 1000),
		neighChans:   make(map[int]chan *messages.Message),
		internalChan: make(chan *messages.BeaconMessage, 1000),
		mtx:          sync.Mutex{},
	}
}

func (g *Graph) AddNode(n *Node) {
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

func (n *Node) AddNeighbor(neighbor *Node) {
	n.Neighbors[neighbor.Id] = neighbor
	n.neighChans[neighbor.Id] = neighbor.msgChan
	neighbor.Neighbors[n.Id] = n
	neighbor.neighChans[n.Id] = n.msgChan
}

func (n *Node) Degree() int {
	return len(n.Neighbors)
}

func (g *Graph) DHCV(d int) {
	wg := sync.WaitGroup{}
	ctx, cancel := context.WithCancel(context.Background())
	defer func() {
		cancel()
		wg.Wait()
	}()
	for _, n := range g.Nodes {
		wg.Add(1)
		go func() {
			n.start(ctx)
			wg.Done()
		}()
	}
	g.initialization()
	for i := range d {
		for _, n := range g.Nodes {
			g.wg.Add(1)
			go func() {
				n.relativeMax(i)
				g.wg.Done()
				fmt.Printf("[%d]: Finished Round %d\n", n.Id, i+1)
			}()
		}
		g.wg.Wait()
	}

	time.Sleep(500 * time.Millisecond)
	// exception 1
mainLoop:
	for _, n := range g.Nodes {
		ch := n.PCH[d]
		pathToCH := n.findPath(ch)
		for _, node := range pathToCH {
			if node.PCH[d] != ch && ch != node {
				fmt.Printf("[%d]: passing through %d\n", n.Id, node.PCH[d].Id)
				// node passes through another cluster
				for idx := d - 1; idx >= 0; idx-- {
					pch := n.PCH[idx]
					if _, ok := n.Neighbors[pch.Id]; ok {
						n.PCH[d] = pch
						fmt.Printf("[%d]: Setting new CH %d\n", n.Id, node.Id)
						continue mainLoop
					}
				}
				fmt.Printf("[%d]: Didn't find a better CH\n", n.Id)
				n.PCH[d] = node.PCH[d]
				break
			}
		}
	}
	for _, n := range g.Nodes {
		ch := n.PCH[d]
		if _, ok := g.clusters[ch.Id]; !ok {
			g.clusters[ch.Id] = make([]int, 0)
		}
		g.clusters[ch.Id] = append(g.clusters[ch.Id], n.Id)
	}

	fmt.Println(g.clusters)
}
func (n *Node) isCH(d int) bool {
	return n.PCH[d].Id == n.Id
}
func (g *Graph) initialization() {
	for _, cn := range g.Nodes {
		cn.CNN[0] = cn
		cn.PCH[0] = cn
	}
}

func (n *Node) GetdHopNeighs(d int) map[int]*Node {
	visited := make(map[int]struct{})
	q := make([]*Node, 1)
	q[0] = n
	visited[n.Id] = struct{}{}
	for i := 0; i < d; i++ {
		for _, cn := range q {
			q = q[1:]
			for _, nn := range cn.Neighbors {
				if _, ok := visited[nn.Id]; ok {
					continue
				}
				visited[nn.Id] = struct{}{}
				q = append(q, nn)
			}
		}
	}
	neighs := make(map[int]*Node)
	for _, cn := range q {
		neighs[cn.Id] = cn
	}
	return neighs
}

func (n *Node) bcast(msg *messages.Message) {
	for _, cn := range n.Neighbors {
		timer := time.NewTimer(500 * time.Millisecond)
		select {
		case n.neighChans[cn.Id] <- msg:
			continue
		case <-timer.C:
			fmt.Printf("[%d]: Failed to send message (bcast) to: (%d)\n", n.Id, cn.Id)
			timer.Stop()
		}
	}
}
func (n *Node) relativeMax(i int) {
	n.mtx.Lock()
	defer n.mtx.Unlock()
	idx := i + 1
	// fmt.Printf("[%d]: Round %d\n", n.Id, idx)
	neighs := n.GetdHopNeighs(idx)
	for nodeId := range neighs {
		msg := messages.NewMessage(n.Id, nodeId, messages.DefaultTTL, messages.NewBeaconMessage(n.Velocity, n.PosX, n.PosY, n.Id, n.Degree()))
		// check if its neighbor else bcast the message
		if c, ok := n.neighChans[nodeId]; ok {
			c <- msg
		} else {
			n.bcast(msg)
		}
	}
	// fmt.Printf("[%d]: Sent all beacon messages\n", n.Id)
	minMobility := math.MaxFloat64
	var CNN int
	messagesReceived := make(map[int]struct{})
	for len(messagesReceived) < len(neighs) {

		msg := <-n.internalChan
		if _, ok := messagesReceived[msg.SenderId]; ok {
			continue
		}
		relativeSpeed := n.CNN[i].getRelativeSpeedPos(msg.Velocity, msg.PosX, msg.PosY)
		if relativeSpeed < minMobility || (relativeSpeed == minMobility && n.CNN[i].Degree() < msg.Degree) {
			minMobility = relativeSpeed
			CNN = msg.SenderId
		}
		messagesReceived[msg.SenderId] = struct{}{}
	}
	cnnNode := neighs[CNN]
	n.CNN[idx] = cnnNode
	if cnnNode.Degree() > n.PCH[i].Degree() {
		n.PCH[idx] = cnnNode
	} else {
		n.PCH[idx] = n.PCH[i]
	}

	if n.CNN[idx].Degree() == n.CNN[idx-1].Degree() && n.CNN[idx].Velocity < n.CNN[idx-1].Velocity {
		msg := messages.NewMessage(n.Id, n.CNN[idx-1].Id, messages.DefaultTTL, messages.NewPCHMessage(n.CNN[idx], idx))
		n.sendMsg(msg)
	}
}

func (n *Node) sendMsg(msg *messages.Message) {
	if c, ok := n.neighChans[msg.To]; ok {
		timer := time.NewTimer(500 * time.Millisecond)
		select {
		case c <- msg:
			break
		case <-timer.C:
			fmt.Printf("[%d]: Failed to send message to: (%d)\n", n.Id, msg.To)
			timer.Stop()
		}
		return
	}
	n.bcast(msg)
}

func (n *Node) start(ctx context.Context) {
	messagesReceived := make(map[int]struct{})
	for {
		select {
		case m := <-n.msgChan:
			if _, ok := messagesReceived[m.ID]; ok {
				continue
			}
			messagesReceived[m.ID] = struct{}{}
			if m.Ttl <= 0 {
				continue
			}
			if m.To != n.Id {
				newMsg := *m
				newMsg.Ttl -= 1
				n.bcast(&newMsg)
				continue
			}
			switch m.Msg.(type) {
			case *messages.BeaconMessage:
				n.internalChan <- m.Msg.(*messages.BeaconMessage)
			case *messages.PCHMessage:
				go n.handlePCHMessage(m.Msg.(*messages.PCHMessage))
			}
		case <-ctx.Done():
			// fmt.Printf("[%d]: Terminating...\n", n.Id)
			return
		}
	}
}

func (n *Node) handlePCHMessage(msg *messages.PCHMessage) {
	n.mtx.Lock()
	defer n.mtx.Unlock()
	if msg.Node.(*Node).Degree() > n.PCH[msg.Round].Degree() {
		fmt.Printf("[%d]: Setting new pch %d\n", n.Id, msg.Node.(*Node).Id)
		n.PCH[msg.Round] = msg.Node.(*Node)
	}
}

func (n *Node) getRelativeSpeedPos(vel float64, x, y float64) float64 {
	dx := math.Pow(n.PosX-x, 2)
	dy := math.Pow(n.PosY-y, 2)
	dxy := math.Sqrt(dx + dy)
	return a*dxy + b*math.Abs(n.Velocity-vel)

}

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

func (n *Node) findPath(tNode *Node) []*Node {
	type state struct {
		node *Node
		path []*Node
	}
	visited := make(map[int]struct{})
	q := make([]state, 1)
	q[0] = state{
		node: n,
		path: []*Node{n},
	}
	for len(q) > 0 {
		cState := q[0]
		if cState.node.Id == tNode.Id {
			return cState.path
		}
		visited[cState.node.Id] = struct{}{}
		q = q[1:]
		for _, n := range cState.node.Neighbors {
			if _, ok := visited[n.Id]; ok {
				continue
			}
			newPath := make([]*Node, len(cState.path))
			copy(newPath, cState.path)
			newPath = append(newPath, n)
			newState := state{
				node: n,
				path: newPath,
			}
			q = append(q, newState)
		}
	}
	return []*Node{}
}

func (g *Graph) getClusterMembers(n *Node, d int) []*Node {
	cms := make([]*Node, 0)
	for _, node := range g.Nodes {
		if node.PCH[d] == n.PCH[d] && node != n {
			cms = append(cms, node)
		}
	}
	return cms
}

func printPath(path []*Node) {
	for _, n := range path {
		fmt.Printf("%d ", n.Id)
	}
	fmt.Println()
}
