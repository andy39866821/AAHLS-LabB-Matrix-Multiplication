# NTHU AAHLS Lab B: Matrix Multiplication


## Function Description

Matrix multiplication is a common numerical operation in math, Figure below demonstrate the method of multiplicaiton. And due to it simple computation and high parallelism, it is very suitable to design hardware to accelerate the computing.
![](https://i.imgur.com/2Y7i32Z.png)
![](https://i.imgur.com/SOCJ32t.png)
In this lab, we will use High-level Synthesis to implement matrix multiplication in two different ways, first is brutal force, just implement software matrix multiplication code with different pipeline directive; Second is block matrix mutilication, which used fixed computing unit to fit different kind of matrix.

### Brutal Force

Simple three nested-loop to compute the result, I use directive **#pragame HLS ARRAY_RESHAPE** to reshape the input array, make it has more ports to be read, which indicate that we can compute more result in one cycle. Also I set **#pragma HLS PIPELINE** to differnt loop body to analyze their performance are resource usage.
``` cpp=
void matrixmul(int A[N][M], int B[M][P], int AB[N][P]) {
    #pragma HLS ARRAY_RESHAPE variable=A complete dim=2
    #pragma HLS ARRAY_RESHAPE variable=B complete dim=1
    /* for each row and column of AB */
    row: for(int i = 0; i < N; ++i) {
        col: for(int j = 0; j < P; ++j) {
            #pragma HLS PIPELINE II=1
            /* compute (AB)i,j */
            int ABij = 0;
            product: for(int k = 0; k < M; ++k) {
                ABij += A[i][k] * B[k][j];
            }
            AB[i][j] = ABij;
        }
    }
}
```

### Block Matrix Mutliplication
The concept of block matrix is used fixed size compute kernel to fit different kinds of matrix multiplication. For example, below figure demonstate a 4x4 matrix multiplication by using fixed 2x2 block matrix multiplication four times.

![](https://i.imgur.com/AOZxeIg.png)

In the complex implementation, I use **hls:stream** to implement a FIFO input, and I fixed the compute unit to compute **BLOCK_SIZE** matrix **SIZE/BLOCK_SIZE** times to fit different **SIZE** computation. Also in this part I will analyze **#pragma HLS DATAFLOW** and **#pragma HLS PIPELINE** effect by the synthesis report.

```cpp=

typedef int DTYPE;
const int SIZE = 8;
const int BLOCK_SIZE = 4;

typedef struct {
    DTYPE a[BLOCK_SIZE]; 
} blockvec;

typedef struct {
    DTYPE out[BLOCK_SIZE][BLOCK_SIZE]; 
} blockmat;

void blockmatmul(hls::stream<blockvec> &Arows, hls::stream<blockvec> &Bcols,blockmat &ABpartial, int it) {
    
    DTYPE AB[BLOCK_SIZE][BLOCK_SIZE] = { 0 };

    #pragma HLS DATAFLOW
    int counter = it % (SIZE/BLOCK_SIZE);
    static DTYPE A[BLOCK_SIZE][SIZE];
    if(counter == 0){ //only load the A rows when necessary
        loadA: for(int i = 0; i < SIZE; i++) {
            blockvec tempA = Arows.read();
            for(int j = 0; j < BLOCK_SIZE; j++) {
                #pragma HLS PIPELINE II=1
                A[j][i] = tempA.a[j];
            }
        }
    }

    partialsum: for(int k=0; k < SIZE; k++) {
        blockvec tempB = Bcols.read();
        ps_i:for(int i = 0; i < BLOCK_SIZE; i++) {
            ps_j:for(int j = 0; j < BLOCK_SIZE; j++) {
                #pragma HLS PIPELINE II=1
                
                AB[i][j] = AB[i][j]  + A[i][k] * tempB.a[j];
            }
        }
    }
    
    
    writeoutput: for(int i = 0; i < BLOCK_SIZE; i++) {
        for(int j = 0; j < BLOCK_SIZE; j++) {
            ABpartial.out[i][j] = AB[i][j];
        }
    }
    
}

```


## Build Flow

In this lab, I use Vitis to do the HLS. Here is the step
1. Software C simulation
2. Synthesis C++ code to Verilog code
    a. Clock period: 10ns
4. Co-simulation to verify the functionality


## Result & Analysis

### Brutal Implementation
I test different position of PIPELINE directive, result meet my expection. For the product pipeline, it only consume 3 DSP (two add and one multiply), and consume N(32)xP(32)xM(32) cycles; For the col pipeline, it uses 3xM(32) = 96 DSP and consmes N(32)xP(32) cycles; For row pipeline, it uses 3xP(32)xM(32)=3072 DSP and consumes N(32) cycles.
![](https://i.imgur.com/ZUFhySt.png)

![](https://i.imgur.com/NGEvcG4.png)


I also test **#pragma HLS ARRAY_RESHAPE** in this design, I set different dimension of reshape constraint. We can find that without reshape, all of the array are mapped to single port memory with 32 bits data port and 10 bits address port; In dim 1 or 2, A and B both are mapped to single port memory with 1024 bits data port and 6 bits address port; And for the last one, dim=0, it expands entire array, reshape the A and B array to pure registers with 32768 bits, but it is not synthesizable in Vviado since upper bound of register size is 4096 bits.
![](https://i.imgur.com/28wtB3J.png)


### Block Matrix Multiplication

In the source code, we have two stream input (Arows, Bcols), and one array output. For **block_vec** stream input, the port after synthesisze is sizeof(int)xBLOCK_SIZE=128 bits port with two handshaking signals: empty_n and read to load data with host.
And for array output block_mat, it has sizeof(int)xBLOCK_SIZExBLOCK_SIZE = 32x4x4 = 512 bits port with ap_vld writing handshaking signal.
![](https://i.imgur.com/qkDY4ZO.png)


For three part of design, Loop load consumes 9xSIZE(8) = 72 cycles; Loop compute comsume 24xSIZE(8) = 192 cycles; Loop store consume BLOCK_SIZE(4)xBLOCK_SIZE(4)+loop_entry_time(2) = 18 cycles.
![](https://i.imgur.com/qSgV1HB.png)
