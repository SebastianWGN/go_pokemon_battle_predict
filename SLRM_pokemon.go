package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
)

func update_weights(radio []float64, sales []float64, weight float64, bias float64, learning_rate float64) (float64, float64) {
	weight_deriv := 0.0
	bias_deriv := 0.0
	companies := len(radio)

	for i := 0; i < companies-1; i++ {
		weight_deriv += -2 * radio[i] * (sales[i] - (weight*radio[i] + bias))
		bias_deriv += -2 * (sales[i] - (weight*radio[i] + bias))
	}

	weight -= (weight_deriv / float64(companies)) * learning_rate
	bias -= (bias_deriv / float64(companies)) * learning_rate

	return weight, bias
}

func cost_function(radio []float64, sales []float64, weight float64, bias float64) float64 {
	companies := len(radio)
	total_error := 0.0
	for i := 0; i < companies-1; i++ {
		total_error += math.Pow((sales[i] - (weight*radio[i] + bias)), 2) //revisar error
	}
	return total_error / float64(companies)
}

func predict_sales(radio float64, weight float64, bias float64) float64 {
	pred := weight*radio + bias
	return pred
}

func train(radio []float64, sales []float64, weight float64, bias float64, learning_rate float64, iters int) (float64, float64, []float64) {
	var cost_history []float64

	for i := 0; i < iters-1; i++ {
		weight, bias = update_weights(radio, sales, weight, bias, learning_rate)

		//calculate cost for auditing purposes
		cost := cost_function(radio, sales, weight, bias)
		cost_history = append(cost_history, cost)

		//log progress
		if i%10 == 0 {
			fmt.Println("Iter =", i, "weight =", weight, "bias =", bias, "cost =", cost)
		}
	}

	return weight, bias, cost_history
}

func main() {
	csvfile, err := os.Open("Pokemon_matchups.csv")
	if err != nil {
		log.Fatalln("No se puede abrir el archivo", err)
	}

	var radio []float64
	var sales []float64

	//Parse the file

	r := csv.NewReader(csvfile)

	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		r, err := strconv.ParseFloat(record[3], 64)
		s, err := strconv.ParseFloat(record[14], 64)
		radio = append(radio, r)
		sales = append(sales, s)
	}

	weight := 0.0
	bias := 0.0
	lr := 0.01
	iters := 10
	a, b, c := train(radio, sales, weight, bias, lr, iters)
	fmt.Println(c)

	p := predict_sales(50, a, b)
	fmt.Println(p)

}
