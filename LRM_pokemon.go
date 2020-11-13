package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/kniren/gota/dataframe"
	"gonum.org/v1/gonum/mat"
)

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

func sigmoid(x chan mat.Matrix, y chan mat.Matrix) {
	eval_matrix := <-x
	outputs := mat.Col(nil, 0, eval_matrix) //change outputs

	size := len(outputs)
	process_outputs := make([]float64, size)
	for i, value := range outputs {
		process_outputs[i] = sigmoid_util(value)
	}
	y <- mat.NewDense(size, 1, process_outputs)
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

func train_test_split(filename string, split float64, train_data chan mat.Matrix, test_data chan mat.Matrix) {
	pokemon_matchups_train, err := os.Open(filename)
	pokemon_matchups_test, err := os.Open("./Pokemon_matchups_test.csv")

	if err != nil {
		log.Fatalln("No se puede abrir el archivo", err)
	}

	rand.Seed(42)
	//Read file
	file := dataframe.ReadCSV(pokemon_matchups_train)
	file_data := file.Select([]string{"Hp_1", "Attack_1", "Hp_2", "Attack_2", "Winner"})

	//set train data
	var train mat.Matrix
	train = matrix{file_data}

	//set test
	file2 := dataframe.ReadCSV(pokemon_matchups_test)
	file2_data := file2.Select([]string{"Hp_1", "Attack_1", "Hp_2", "Attack_2", "Winner"})

	//set test data
	var test mat.Matrix
	test = matrix{file2_data}

	train_data <- train
	test_data <- test

}

func approximate(X mat.Matrix, weights mat.Matrix, bias float64, n_row int, n_col int, matrix_util chan mat.Matrix) {
	m_result := mat.NewDense(n_row, n_col, nil)
	m_result.Product(X, weights)
	linear_model := addScalar(bias, m_result)
	matrix_util <- linear_model
}

func compute_gradients(matrix_util2 chan mat.Matrix, X mat.Matrix, y mat.Matrix, n_samples int, n_features int, db_back chan float64, dw_back chan mat.Matrix) {
	y_predicted := <-matrix_util2
	y_sub := subtract(y_predicted, y)
	_, y_sub_col := y_sub.Dims()
	m_prod := mat.NewDense(n_features, y_sub_col, nil)
	m_prod.Product(X.T(), y_sub)
	//prueba = m_prod
	mvar := 1.0 / float64(n_samples)
	dw := multiplyScalar(mvar, m_prod)
	db := mvar * sumElements(y_sub)
	dw_back <- dw
	db_back <- db
}

type LogRegression struct {
	lr      float64
	n_iters int
	weights mat.Matrix
	bias    float64
}

func (l *LogRegression) fit(X mat.Matrix, y mat.Matrix) (mat.Matrix, float64) {
	//init parameters
	n_samples, n_features := X.Dims()
	_, l_columns := l.weights.Dims()
	matrix_util := make(chan mat.Matrix)
	matrix_util2 := make(chan mat.Matrix)
	dw_back := make(chan mat.Matrix)
	db_back := make(chan float64)
	//var prueba mat.Matrix

	//gradient descent
	for i := 0; i < l.n_iters-1; i++ {
		//approximate
		go approximate(X, l.weights, l.bias, n_samples, l_columns, matrix_util)
		//linear_model := <-matrix_util
		go sigmoid(matrix_util, matrix_util2)
		//compute gradients
		go compute_gradients(matrix_util2, X, y, n_samples, n_features, db_back, dw_back)
		//update parameters
		dw := <-dw_back
		db := <-db_back
		l.weights = subtract(l.weights, multiplyScalar(l.lr, dw))
		l.bias -= l.lr * db
	}

	return l.weights, l.bias
}

func (l *LogRegression) predict(X mat.Matrix) mat.Matrix {
	n_samples, _ := X.Dims()
	_, l_columns := l.weights.Dims()
	matrix_predict := make(chan mat.Matrix)
	matrix_result := make(chan mat.Matrix)
	linear_model := mat.NewDense(n_samples, l_columns, nil)
	linear_model.Product(X, l.weights)
	matrix_predict <- linear_model
	sigmoid(matrix_predict, matrix_result)
	y_predicted := <-matrix_result

	return y_predicted
}

func main() {

	filename := "./Pokemon_matchups.csv"
	split := 0.66
	train_data := make(chan mat.Matrix)
	test_data := make(chan mat.Matrix)

	//set weights
	weights := make([]float64, 5)
	for i := range weights {
		weights[i] = 1
	}

	weights_data := mat.NewDense(5, 1, weights)
	//set y data
	datab := make([]float64, 18515)
	for i := range datab {
		datab[i] = rand.Float64()
	}
	y_data := mat.NewDense(18515, 1, datab)

	go train_test_split(filename, split, train_data, test_data)

	train_data_real := <-train_data
	//test_data_real := <-test_data

	//set waited_data

	time.Sleep(time.Second * 200)

	data2 := LogRegression{0.000005, 100000, weights_data, 0.0}

	fmt.Println(data2.fit(train_data_real, y_data))

	//fmt.Println(data2.predict(test_data_real))

}
