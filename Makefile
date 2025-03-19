app/run:
	go run main.go

graph:
	dot -Tpng test.dot -o output.png

clean:
	rm -rf *.png
	rm -rf ./sumo/*
	rm -rf ./graphviz/*
	rm -rf ./graph_info/*
	rm -rf ./cars_info/snapshots/*
clean/snapshots:
	rm -rf ./snapshots/*
.PHONY: graph