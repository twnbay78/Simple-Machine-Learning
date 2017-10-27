#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
// print arrays
//#define DEBUG
//#define DEBUG2
//#define DEBUG3
/* By. Leo Scarano
 * Rutgers University 
 * 2017
 *
 * This program will take in training data on house prices/attributes, and will use
 * a linear regression model to compute weights. The model is based off of 
 * finding the pseudoinverse, and the inverse method used is Gauss-Jordan 
 * elimination. After the weights are computed, they are applled to a set
 * of hosue attrivutres to estimate the price of the hosue. 
 *
 * Theoretically, the more training  data we have, the better fit we will get 
 *
 */

	// function to free a pointer to a double array
	void freeDouble(double** arr, int rowSize){
		for(rowSize -= 1; rowSize >= 0; rowSize--){
			free(arr[rowSize]);
			arr[rowSize] = NULL;
		}
		free(arr);
		arr = NULL;
	}


// prints a double-typed matrix to STDOUT
void printArray(double** matrix, int rowSize, int colSize){
	int i, j;
	for(i = 0; i < rowSize; ++i){
		for(j = 0; j < colSize; ++j){
			printf("%lf ", matrix[i][j]);
		}
	printf("\n");
	}
}	


/* Transposes a matrix
 * index(i,j) of the input matrix becomes index(j,i) for all values in the input matrix
 */
void matrixTranspose(double** inputArray, double** outputArray, int rowSize, int colSize){
	int i, j;
	for(i = 0; i < rowSize; ++i){
		for(j = 0; j < colSize; ++j){
			outputArray[j][i] = inputArray[i][j];
#if defined(DEBUG3)
	      		 printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
			 printf("inputArr[%d][%d]: %lf, ouptutArr[%d][%d]: %lf\n", i, j, inputArray[i][j], j, i, outputArray[j][i]);
	      		 printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
#endif		
		}
	}
}

// Calculating the inverse using Gauss-Jordan elimination
// Split into 2 parts: FORWARD PASS (upper triangular matrix) and REAR PASS (lower triangular matrix)
double** matrixInverse(double** matrix, int rowSize, int colSize){
	// FORWARD PASS -> getting upper triangular matrrix 
	// 1. make pivots 1
	// 2. zero out pivot columns
	
	int i, j, k, p, dist, check;
	// determining if the pivot value needs to be checked
	int counter = 0;
	// constant = row operation constant needed to make pivot column values 0
	// pivot = row operation constant needed to make pivot value 1
	double constant, pivot; 
	for(i = 0; i < rowSize; ++i){
		// declare pivot col
		if(counter == 0){
			p = i;
			counter++;
			// Step 1
			if(matrix[i][p] != 1){
				pivot = 1/matrix[i][p];
				for(j = 0; j < colSize; ++j){
					matrix[i][j] *= pivot;
				}
			// since the pivot value is checked, the distance from the pivot 
			// on the col is now 1
			dist = 1;	
			// since the pivot is checked, we need to incriment the row
			continue;
			}
		}
		// Step 2
		for(j = i; j < rowSize; ++j){
			// if pivot col value is not 0, we must apply row operation to the row
			if(matrix[j][p] != 0){
				check  = 1;
			}
			constant = matrix[j][p];
			// iterate through the row and apply constant to each value
			for(k = p; k < colSize; ++k){
				if(check == 1){
					matrix[j][k] -= (constant * (matrix[j-dist][k]));
				}
			}
			
			// we are done with all of the rows in the pivot column, distance from pivot increases
			check = 0;
			dist++;
		}		
		// resetting variables, moving onto the next pivot col
		counter = 0;
		i -= 1;
		dist = 0;
	}

#if defined(DEBUG)
	printf("Upper Triangular matrix: \n");
	printArray(matrix, rowSize, colSize);
#endif

	// REAR PASS -> getting lower triangular matrix
	// resetting variables
	dist = 1;
	counter = 0;
	constant = 0;
	check = 0;
	for(i = rowSize-1; i > 0; --i){
		j = i - 1;
		while (j >= 0){
			// check to see if row operation is needed
			if(matrix[j][i] != 0){
				constant = matrix[j][i];
				for(k = 0; k < colSize; ++k){
					matrix[j][k] -= (constant*matrix[j+dist][k]);
				}
			}
			// moving up 1 in the row, distance increses by 1
			check = 0;
			dist++;
			j--;
		}
		// Changing the pivot col we are checking, distance is now 1
		dist = 1;
	}
	
#if defined(DEBUG)
	printf("Lower Triangular Matrix: \n");
	printArray(matrix, rowSize, colSize);
#endif
	
	// Taking the identity matrix out
	// initializing new matrix
	double** outputMatrix = (double**)malloc(sizeof(double*) * rowSize);
	for(i = 0; i < rowSize; ++i){
		outputMatrix[i] = (double*)malloc(sizeof(double*) * rowSize);
	}

	// copying inverse over to the new matrix
	for(i = 0; i < rowSize; ++i){
		for(j = rowSize; j < 2*rowSize; ++j){
			outputMatrix[i][j-rowSize] = matrix[i][j];
		}
	}
	return outputMatrix;
}


// Multiplies 2 matricies, outputs a separate matrix
// This is not an in-place multiplication
double** matrixMultiply(double** matrix1, double** matrix2, int row1, int col1, int row2, int col2){
	int i, j, k;
	// initializing new matrix for multiply 
	// output is a square matrix of (attribute_count+1) x (attribute_count+1)
	int numElem = row1 * col2;
	double** outputMatrix = (double**)calloc(numElem, sizeof(double*) * row1);
	for(i = 0; i < row1; ++i){
		outputMatrix[i] = (double*)calloc(numElem, sizeof(double) * col2);
	}

	for(i = 0; i < row1; ++i){
		for(j = 0; j < col2; ++j){
			for(k = 0; k < col1; ++k){
				outputMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
			}	
		}
	}
	return outputMatrix;
}


int main(int argc, char* argv[]){
	// check for proper argument usage
	if(argc < 3 || argc > 3){
		printf("ERROR: Inproper use of arguments!\n");
		exit(0);
	}

	// Opening the file with a file pointer
	char* train_filename = argv[1];
	int attribute_count = -1;
	int num_of_mat_entries = -1;
	FILE* fpointer;
	fpointer = fopen(train_filename, "r");
	
	// if the file cannot be opened properly
	if(fpointer == NULL){
		printf("ERROR: Could not load file\n");
		exit(0);
	}else{
		// reads first two integer values from testing file
		// first value = # of attributes
		// second value = # of training entries
		fscanf(fpointer, "%d", &attribute_count);
		fscanf(fpointer, "%d", &num_of_mat_entries);
#if defined(DEBUG)
		printf("attr: %d, entries: %d\n", attribute_count, num_of_mat_entries);
#endif

		// initialize matrix X
		double** X = (double**)malloc(sizeof(double*)*num_of_mat_entries);
		int i;
		for(i = 0; i < num_of_mat_entries; ++i){
			X[i] = (double*)malloc(sizeof(double) * (attribute_count+1));
		}

		// initializing space for vector Y
		double** Y = (double**)malloc(sizeof(double*) * num_of_mat_entries);
		for(i = 0; i < num_of_mat_entries; ++i){
			Y[i] = (double*)malloc(sizeof(double));
		}

		// Fill values of the matrix
		int j;
		int counter = 0;
		double temp = 0;
		for(i = 0; i < num_of_mat_entries; ++i){
			for(j = 0; j < (attribute_count+1); ++j){
				// case for first col - values must be 1
				if(j == 0){
					X[i][j] = 1;
					continue;
				}
				while(fscanf(fpointer, "%lf", &X[i][j]) != EOF){
					fscanf(fpointer, " ,");
					counter++;
#if defined(DEBUG)
					printf("Counter: %d\n", counter);
					printf("i: %d, j: %d\n", i, j);
#endif
					// case for the house price - exclude house price in Xa
					// places the house prices in a temp variable and creates
					// the Y vetctor
					if(counter >= attribute_count){
						fscanf(fpointer, "%lf", &temp);
						Y[i][0] = temp;
						counter = 0;
					}
					break;
				}
			}
		}
		fclose(fpointer);
		
		// printing array after data input
#if defined(DEBUG)
		printf("X: \n");
		printArray(X, num_of_mat_entries, (attribute_count+1));
#endif

		// transposing X
		double** transpose = (double**)malloc(sizeof(double*) * (attribute_count+1));
	        for(i = 0; i < (attribute_count+1); ++i){
			transpose[i] = (double*)malloc(sizeof(double) * num_of_mat_entries);
		}	
			
		matrixTranspose(X, transpose,  num_of_mat_entries, (attribute_count+1));

		// printing matrix after transpose
#if defined(DEBUG)
		printf("X^T: \n");
		printArray(transpose, (attribute_count+1), num_of_mat_entries);
#endif

		// multiplying X^T * X
		double** multiply = matrixMultiply(transpose, X, (attribute_count+1), num_of_mat_entries, num_of_mat_entries, (attribute_count+1));

		// printing matrix after transpose
#if defined(DEBUG)
		printf("X^T * X: \n");
		printArray(multiply, (attribute_count+1), (attribute_count+1));
#endif

		// initialize augmented matrix
		// augmented matrix will be of size n x 2n
		// the second half of the matrix columns will be the corresponding
		// identity matrix of the left half of the matrix columns
		int num_of_augment_elements = (attribute_count+1) * (2*(attribute_count+1)); 
		double** augment = (double**)malloc(sizeof(double*) * (attribute_count+1));
		for(i = 0; i < (attribute_count+1); ++i){
			// using callcoc so the identity matrix side will be all 0's
			augment[i] = (double*)calloc(num_of_augment_elements, sizeof(double) * (attribute_count+1)*2);
		}

		// copying the values from (X^T * X) over to the augmented matrix
		for(i = 0; i < (attribute_count+1); ++i){
			for(j = 0; j < 2*(attribute_count+1); ++j){
				if(i < (attribute_count+1) && j < (attribute_count+1)){
					augment[i][j] = multiply[i][j];
				}else if(j == i + (attribute_count+1)){
					augment[i][j] = 1;
				}
			}
		}
		
		// printing the matrix after it has been augmented 
#if defined(DEBUG)
		printf("Augment: \n");
		printArray(augment, (attribute_count+1), 2*(attribute_count+1));
#endif

		// calculating the inverse of (X^T * X)
		double** inverse = matrixInverse(augment, (attribute_count+1), 2*(attribute_count+1));

		// printing the matrix after the inverse is calculated
#if defined(DEBUG)
		printf("Inverse: \n");
		printArray(inverse, (attribute_count+1), (attribute_count+1));
#endif

		// printing Y
#if defined(DEBUG)
		printf("Y: \n");
		printArray(Y, num_of_mat_entries, 1);
#endif
		
		// multiplying inverse and transpose
		double** inverse_transpose = matrixMultiply(inverse, transpose, (attribute_count+1), (attribute_count+1), (attribute_count+1), num_of_mat_entries);

		// printing inverse * transpose 
#if defined(DEBUG)
		printf("(X^D * X)^-1 * X^T:\n");
		printArray(inverse_transpose, (attribute_count+1), (attribute_count+1));
#endif

		// Getting the weight matrix
		// multiplying inverse_transpose by Y
		double** weights = matrixMultiply(inverse_transpose, Y, (attribute_count+1), num_of_mat_entries, num_of_mat_entries, 1);

		// printing the weight matrix
#if defined(DEBUG)
		printf("Weights: \n");
		printArray(weights, (attribute_count+1), 1);
#endif

		// Reading in the test data
		char* testFilename = argv[2];
		FILE* fpointer = fopen(testFilename, "r");
		int test_entries;

		// If the file couldn't be opened, stop
		if(fpointer == NULL){
			printf("ERROR: Could not load file\n");
			exit(0);
		}else {
			fscanf(fpointer, "%d", &test_entries);		
			// initialize space for test matrix
#if defined(DEBUG)
			printf("te: %d, attc: %d\n", test_entries, (attribute_count+1));
#endif
			double** testMatrix = (double**)malloc(sizeof(double*) * test_entries);
			int i;
			for(i = 0; i < test_entries; ++i){
				testMatrix[i] = (double*)malloc(sizeof(double) * (attribute_count+1));
			}		

			// loop through file and fill matrix
			// 1's are inserted into the first col for w0
			int j, k;
			for(j = 0; j < test_entries; ++j){
				for(k = 0; k < (attribute_count+1); ++k){
					if(k == 0){
						testMatrix[j][k] = 1;
						continue;
					}
					while(fscanf(fpointer, "%lf", &testMatrix[j][k]) != EOF){
						fscanf(fpointer, " ,");
#if defined(DEBUG)
						printf("mat[%d][%d]: %lf\n", j, k, testMatrix[j][k]);
#endif
						break;
					}
				}
			}
			
			// printing matrix from the test file
#if defined(DEBUG)
			printArray(testMatrix, test_entries, (attribute_count+1));
#endif

			// appying equation: Y = w0 + w1x1 + w2x2 + ..... + wNxN
			// Y = estimated hosue price
			// w = weight
			// x = attribute
			// N = number of attributes
			double value;
			for(i = 0; i < test_entries; ++i){
				for(j = 0; j < (attribute_count+1); ++j){
#if defined(DEBUG)
					printf("weight: %lf, attribute: %lf\n", weights[j][0], testMatrix[i][j]);
#endif
					value += weights[j][0] * testMatrix[i][j]; 
				}
				printf("%0.0f\n", value);
				value = 0;
			}
			freeDouble(testMatrix, test_entries);
		}
		// freeing all of the memory allocated
		freeDouble(X, num_of_mat_entries);
		freeDouble(Y, num_of_mat_entries);
		freeDouble(transpose, (attribute_count+1));
		freeDouble(multiply, (attribute_count+1));
		freeDouble(inverse, (attribute_count+1));
		freeDouble(inverse_transpose, (attribute_count+1)); 
		freeDouble(augment, (attribute_count+1));
		freeDouble(weights, (attribute_count+1));

	}

	
	
#if defined(DEBUG2)
	// arrays for testing
	int rowSize = 3;
	int colSize = 3;
	double** arr1 = (double**)malloc(sizeof(double*) * rowSize);
	double** arr2 = (double**)malloc(sizeof(double*) * colSize);
	double** arr3 = (double**)malloc(sizeof(double*) * rowSize);

	// initializing matricies
	int i;
	for(i = 0; i < rowSize; ++i){
		arr1[i] = (double*)malloc(sizeof(double) * colSize);
	}

	for(i = 0; i < colSize; ++i){
		arr2[i] = (double*)malloc(sizeof(double) * rowSize);
	}
	for(i = 0; i < rowSize; ++i){
		arr3[i] = (double*)malloc(sizeof(double) * rowSize);
	}
	
	// filling test matrix
	double temp;

	int j;
	for(i = 0; i < rowSize; ++i){
		for(j = 0; j < colSize; ++j){
			printf("\nEnter a value for the matrix: ");
			scanf(" %lf", &temp);
			arr1[i][j] = temp;
		}
	}

	printArray(arr1, rowSize, colSize);
	printArray(arr1, rowSize, colSize);
	// transposing test matricies
	matrixTranspose(arr1, arr2, rowSize, colSize);
	

	printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
	printArray(arr2, colSize, rowSize);

	// multiplying matricies
	matrixMultiply(arr1, arr2, arr3, rowSize, colSize, colSize, rowSize);

	printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
	printArray(arr3, rowSize, colSize);
	printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
	
	matrixInverse(arr3, rowSize, rowSize*2);
	printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
	printArray(arr3, rowSize, rowSize*2);

#endif
	
	return 0;
}
