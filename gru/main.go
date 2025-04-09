package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"strconv"
	"strings"
)

func main() {
	weights1 := readFile("weights/layer_0_weight_0.txt")
	weights2 := readFile("weights/layer_0_weight_1.txt")
	weights3 := readFile("weights/layer_0_weight_2.txt")
	weights4 := readFile("weights/layer_1_weight_0.txt")
	weights5 := readFile("weights/layer_1_weight_1.txt")

	weights := [][][]float64{
		{weights1},
		{weights2},
		{weights3},
		{weights4},
		{weights5},
	}

	Train(weights, 1)
}

func readFile(filename string) []float64 {
	data, err := ioutil.ReadFile(filename)
	weights := []float64{}
	if err != nil {
		log.Fatal(err)
	}

	// Convert the byte slice to a string
	fileContent := string(data)

	// Split the string by spaces (assuming space-separated numbers)
	parts := strings.Fields(fileContent)

	// Loop through the parts and convert them to float64
	for _, part := range parts {
		// Convert each string to a float64
		f, err := strconv.ParseFloat(part, 64)
		if err != nil {
			log.Fatal(err)
		}

		weights = append(weights, f)
	}
	return weights
}

func Train(targets [][][]float64, epochs int) {
	// Serialize the data using JSON
	data, err := json.Marshal(targets)
	if err != nil {
		log.Fatal("Error serializing data:", err)
	}

	// Debug print: Check what you're sending
	// fmt.Println("Sending data:", string(data))

	// Send the serialized data over UDP
	serverAddr := "192.168.1.3:4004"
	conn, err := net.Dial("udp", serverAddr)
	if err != nil {
		log.Fatal("Error connecting to server:", err)
	}
	defer conn.Close()

	_, err = conn.Write(data) // Send the byte slice to the server
	if err != nil {
		log.Fatal("Error sending data:", err)
	}

	fmt.Println("Data sent to server.")

	// wait for response
	buffer := make([]byte, 65535)
	n, err := conn.Read(buffer)
	if err != nil {
		fmt.Println("Timeout or error:", err)
		return
	}

	fmt.Printf("Received response from %s: %s\n", string(buffer[:n]))

}
