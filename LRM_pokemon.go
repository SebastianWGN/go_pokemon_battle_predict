package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sync"

	"github.com/go-gota/gota/series"
	"github.com/kniren/gota/dataframe"
	"gonum.org/v1/gonum/mat"
)

var wg sync.WaitGroup

type matrix struct {
	dataframe.DataFrame
}

func (m matrix) At(i, j int) float64 {
	return m.Elem(i, j).Float()
}

func (m matrix) T() mat.Matrix {
	return mat.Transpose{m}
}

func sigmoid_util(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func sigmoid(x mat.Matrix) mat.Matrix {
	eval_matrix := x
	outputs := mat.Col(nil, 0, eval_matrix) //change outputs

	size := len(outputs)
	process_outputs := make([]float64, size)
	for i, value := range outputs {
		process_outputs[i] = sigmoid_util(value)
	}
	return mat.NewDense(size, 1, process_outputs)
}

func sumElements(x mat.Matrix) float64 {
	outputs := mat.Col(nil, 0, x)

	sum := 0.0
	for _, value := range outputs {
		sum += value
	}
	return sum
}

func zeros(n int) []float64 {
	a := make([]float64, n)
	for i := range a {
		a[i] = 0
	}
	return a
}

func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func addScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return add(m, n)
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

func subtractScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return subtract(n, m)
}

func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func multiplyScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return multiply(m, n)
}

func logMatrix(m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	util := mat.Col(nil, 0, m)
	a := make([]float64, r*c)
	for x, value := range util {
		a[x] = math.Log(value)
	}
	n := mat.NewDense(r, c, a)
	return n
}

func train_test_split(filename string, size int, x_train chan mat.Matrix, x_test chan mat.Matrix, y_train chan mat.Matrix, y_test chan mat.Matrix) {

	x_size := (size * 80) / 100

	pokemon_matchups_train, err := os.Open(filename)
	pokemon_matchups_test, err := os.Open(filename)

	if err != nil {
		log.Fatalln("No se puede abrir el archivo", err)
	}

	rand.Seed(42)
	//Read file
	file := dataframe.ReadCSV(pokemon_matchups_train)
	file_data := file.Select([]string{"Index", "Hp_1", "Attack_1", "Hp_2", "Attack_2", "Winner"})

	file_data = file_data.Filter(dataframe.F{"Index", series.LessEq, x_size})
	file_data = file_data.Drop(0)

	//set x train data
	var train mat.Matrix
	train = matrix{file_data}

	//set test
	file2 := dataframe.ReadCSV(pokemon_matchups_test)
	file2_data := file2.Select([]string{"Index", "Hp_1", "Attack_1", "Hp_2", "Attack_2", "Winner"})

	file2_data = file2_data.Filter(dataframe.F{"Index", series.Greater, x_size})
	file2_data = file2_data.Drop(0)

	//set y train data
	var test mat.Matrix
	test = matrix{file2_data}

	file3_data := file_data.Drop(0)
	file3_data = file3_data.Drop(0)
	file3_data = file3_data.Drop(0)
	file3_data = file3_data.Drop(0)

	//set x test data
	var train2 mat.Matrix
	train2 = matrix{file3_data}

	//set y test data
	file4_data := file2.Select([]string{"Index", "Hp_1", "Attack_1", "Hp_2", "Attack_2", "Winner"})
	file4_data = file4_data.Filter(dataframe.F{"Index", series.Greater, x_size})
	file4_data = file4_data.Drop(0)
	file4_data = file4_data.Drop(0)
	file4_data = file4_data.Drop(0)
	file4_data = file4_data.Drop(0)
	file4_data = file4_data.Drop(0)

	var test2 mat.Matrix
	test2 = matrix{file4_data}

	x_train <- train
	x_test <- test
	y_train <- train2
	y_test <- test2

	wg.Done()

}

func approximate(X mat.Matrix, weights mat.Matrix, bias float64, n_row int, n_col int) mat.Matrix {
	m_result := mat.NewDense(n_row, n_col, nil)
	m_result.Product(X, weights)
	linear_model := addScalar(bias, m_result)
	return linear_model
}

func compute_gradients(matrix_util2 mat.Matrix, X mat.Matrix, y mat.Matrix, n_samples int, n_features int) (mat.Matrix, float64) {
	y_predicted := matrix_util2
	y_sub := subtract(y_predicted, y)
	_, y_sub_col := y_sub.Dims()
	m_prod := mat.NewDense(n_features, y_sub_col, nil)
	m_prod.Product(X.T(), y_sub)
	//prueba = m_prod
	mvar := 1.0 / float64(n_samples)
	dw := multiplyScalar(mvar, m_prod)
	db := mvar * sumElements(y_sub)
	return dw, db
}

type LogRegression struct {
	lr      float64
	n_iters int
	weights mat.Matrix
	bias    float64
}

func cost_function(predictions mat.Matrix, y mat.Matrix, cost_result chan float64) {
	observations, _ := y.Dims()
	//For error when 1
	neg_y := multiplyScalar(-1.0, y)
	log_predictions := logMatrix(predictions)
	class1_cost := multiply(neg_y, log_predictions)

	//For error when 0
	comp_y := subtractScalar(1, y)
	log_comp_predictions := logMatrix(subtractScalar(1, predictions))
	class2_cost := multiply(comp_y, log_comp_predictions)

	//Take the sum
	cost_mat := subtract(class1_cost, class2_cost)
	cost := sumElements(cost_mat) / float64(observations)

	cost_result <- cost
}

func count_non_zero(x mat.Matrix) int {
	var non_zeros int
	values := mat.Col(nil, 0, x)
	for _, value := range values {
		if value != 0 {
			non_zeros += 1
		}
	}
	return non_zeros
}

func accuracy_pred(y_predicted mat.Matrix, y mat.Matrix) float64 {
	n_row, _ := y.Dims()
	y_result := subtract(y_predicted, y)
	return (1.0 - float64(count_non_zero(y_result))/float64(n_row)) * 100.0

}

func decision_boundary(y_predicted mat.Matrix) mat.Matrix {
	r, c := y_predicted.Dims()
	values := mat.Col(nil, 0, y_predicted)
	a := make([]float64, r*c)
	for x, value := range values {
		if value < 0.5 {
			a[x] = 0.0
		} else {
			a[x] = 1.0
		}
	}
	y_result := mat.NewDense(len(a), 1, a)
	return y_result
}

func (l *LogRegression) fit(X mat.Matrix, y mat.Matrix) (mat.Matrix, float64) {
	//init parameters
	n_samples, n_features := X.Dims()
	_, l_columns := l.weights.Dims()
	cost_result := make(chan float64)
	var prueba mat.Matrix
	//var cost float64
	var accuracy float64
	//gradient descent
	for i := 0; i < l.n_iters-1; i++ {
		//approximate
		linear_model := approximate(X, l.weights, l.bias, n_samples, l_columns)
		//linear_model := <-matrix_util
		y_predicted := sigmoid(linear_model)
		y_predicted2 := decision_boundary(y_predicted)
		prueba = y_predicted2
		accuracy = accuracy_pred(y_predicted2, y)
		//compute gradients
		dw, db := compute_gradients(y_predicted, X, y, n_samples, n_features)
		//update parameters
		l.weights = subtract(l.weights, multiplyScalar(l.lr, dw))
		l.bias -= l.lr * db
		//v1
		go cost_function(y_predicted, y, cost_result)
		cost := <-cost_result
		if i%1000 == 0 {
			fmt.Println("Iterador: ", i, "cost: ", cost)
		}
	}
	fmt.Println("Predictions: \n")
	matPrint(prueba)
	fmt.Println("Accuracy: ", accuracy, "%")
	return l.weights, l.bias
}

func (l *LogRegression) predict(X mat.Matrix, y mat.Matrix) (mat.Matrix, float64) {
	n_samples, _ := X.Dims()
	_, l_columns := l.weights.Dims()
	//matrix_predict := make(chan mat.Matrix)
	//matrix_result := make(chan mat.Matrix)
	linear_model := mat.NewDense(n_samples, l_columns, nil)
	linear_model.Product(X, l.weights)
	//matrix_predict <- linear_model
	y_predicted := decision_boundary(sigmoid(linear_model))
	accuracy := accuracy_pred(y_predicted, y)
	return y_predicted, accuracy

}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func main() {

	wg.Add(1)

	filename := "./Pokemon_matchups.csv"
	split := 18515
	x_train := make(chan mat.Matrix)
	x_test := make(chan mat.Matrix)
	y_train := make(chan mat.Matrix)
	y_test := make(chan mat.Matrix)

	//set weights
	weights := make([]float64, 5)
	for i := range weights {
		weights[i] = 1
	}

	weights_data := mat.NewDense(5, 1, weights)

	go train_test_split(filename, split, x_train, x_test, y_train, y_test)

	x_train_data := <-x_train
	x_test_data := <-x_test
	y_train_data := <-y_train
	y_test_data := <-y_test

	fmt.Println("X Train: \n")
	fmt.Println(x_train_data)

	fmt.Println("X Test: \n")
	fmt.Println(x_test_data)

	fmt.Println("Y train: \n")
	fmt.Println(y_train_data)

	fmt.Println("Y test: \n")
	fmt.Println(y_test_data)

	data2 := LogRegression{0.0001, 50000, weights_data, 0.0}

	parameters, bias := data2.fit(x_train_data, y_train_data)
	fmt.Println("Paramaters: \n")
	matPrint(parameters)
	fmt.Println("\n Bias: ", bias, "\n")
	predictions, accuracy_predict := data2.predict(x_test_data, y_test_data)
	fmt.Println("Predictions: \n")
	matPrint(predictions)
	fmt.Println("Accuracy predict: ", accuracy_predict, "%")

}
