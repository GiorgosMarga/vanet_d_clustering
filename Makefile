app/run:
	go run main.go

graph:
	dot -Tpng test.dot -o output.png

clean:
	rm -rf *.png
	rm -rf ./sumo/*
	rm -rf ./graphviz/*
	rm -rf ./cars_info/snapshots/*

.PHONY: graph