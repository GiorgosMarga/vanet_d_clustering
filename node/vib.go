package node

import (
	"fmt"
	"math"
	"slices"

	"github.com/GiorgosMarga/vanet_d_clustering/messages"
)

const (
	ClusterStatusNone = iota
	ClusterStatusCM
	ClusterStatusCH
	ClusterStatusSE
)

type VIB struct {
	m map[int]*VIBEntry
}

type VIBEntry struct {
	direction        float64
	location         Location
	velocity         float64
	clusteringStatus int
	hopsToCh         int
	idToCh           int
	cosSim           float64
	avgRelSpeed      float64
}

type Location struct {
	x float64
	y float64
}

func NewVIB() *VIB {
	return &VIB{
		m: make(map[int]*VIBEntry),
	}
}

func (v *VIB) add(msg *messages.BeaconMessage) {
	v.m[msg.SenderId] = &VIBEntry{
		direction: msg.Angle,
		location: Location{
			x: msg.PosX,
			y: msg.PosY,
		},
		velocity:         msg.Velocity,
		clusteringStatus: msg.ClusteringStatus,
		hopsToCh:         msg.HopsToCH,
		idToCh:           msg.IdToCH,
		cosSim:           msg.CosSimilarity,
		avgRelSpeed:      msg.AvgNeighRelSpeed,
	}
}

func (v *VIB) getCHs() []int {
	chs := make([]int, 0)

	for k, entry := range v.m {
		if entry.clusteringStatus == ClusterStatusCH {
			chs = append(chs, k)
		}
	}
	return chs
}
func (v *VIB) getCMs() []int {
	cms := make([]int, 0)

	for k, entry := range v.m {
		if entry.clusteringStatus == ClusterStatusCM {
			cms = append(cms, k)
		}
	}
	return cms
}

func (v *VIB) getEntry(id int) (*VIBEntry, error) {
	entry, ok := v.m[id]
	if !ok {
		return nil, fmt.Errorf("cant find entry")
	}
	return entry, nil
}

func (v *VIB) setAvgComSim(id int, avgCoSim float64) {
	e, ok := v.m[id]
	if !ok {
		fmt.Println("Not ok for:", id)
	}
	e.cosSim = avgCoSim
	v.m[id] = e
}

func (v *VIB) getMinCoSim(ids []int) int {
	minCo := math.MaxFloat64
	id := 0
	for k, ve := range v.m {
		if !slices.Contains(ids, k) {
			continue
		}
		if ve.cosSim < minCo {
			minCo = ve.cosSim
			id = k
		}
	}
	return id
}

func (v *VIB) getMinValCoSim(ids []int) float64 {
	minCo := math.MaxFloat64
	for k, ve := range v.m {
		if !slices.Contains(ids, k) {
			continue
		}
		if ve.cosSim < minCo {
			minCo = ve.cosSim
		}
	}
	return minCo
}
