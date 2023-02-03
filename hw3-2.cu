#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <iostream> 
#include <cuda_runtime.h>
#include <chrono>
#define blocking_factor 32
using namespace std;
std::chrono::steady_clock::time_point total_start, total_end,io_start,io_end,com_start,com_end;

const int INF = ((1 << 30) - 1);
int original_vertex_number;//read vertex number from input file
int edge_number;//read edge number from input file
int vertex_number;//set the space to store vertex
int* cpu_dist = NULL;//CPU memory
int* device_dist = NULL;//device memory

//Try to change the index to get memory
// __device__ __host__ size_t setlocation(int i, int j, int size)
// {
// 	return i * size + j;
// }
int ceil(int a, int b){return (a + b -1)/b;}

void input(char *inFileName){

    FILE* file = fopen(inFileName, "rb");
    fread(&vertex_number, sizeof(int), 1, file);
    fread(&edge_number, sizeof(int), 1, file);

    // cudaMallocHost(&cpu_dist,sizeof(int)*(vertex_number*vertex_number));
    cpu_dist = (int*) malloc(sizeof(int)*vertex_number*vertex_number);
     
    for (int i = 0; i < vertex_number; i++) {
        #pragma omp parallel for
        for (int j = 0; j < vertex_number; j++) {
        if (i==j) {
            cpu_dist[i*vertex_number+j] = 0;
        } else {
            cpu_dist[i*vertex_number+j] = INF;
        }
        }
    }

    //有edge_number條邊，每條邊由3個數所組成，1.srcouse vertex 2. device_dist vertex 3.weight
    //ex. vertex_number=3 edge_number=6 cpu_dist最多有9，因為是有向圖，所以有可能會不滿，但前面已經初始化過值了（0跟無限大）
    //      pair[0]=0 pair[1]=0           cpu_dist[0]=0
    //      pair[0]=0 pair[1]=1 pair[2]=3 cpu_dist[1]=3
    //      pair[0]=0 pair[1]=2 pair[2]=4 cpu_dist[2]=4
    //      pair[0]=1 pair[1]=0           cpu_dist[3]=INF
    //      pair[0]=1 pair[1]=1           cpu_dist[4]=0
    //      pair[0]=1 pair[1]=2 pair[2]=5 cpu_dist[5]=5
    int pair[3];
    #pragma omp parallel for
    for (int i=0; i < edge_number; i++) { 
        fread(pair, sizeof(int), 3, file); 
        cpu_dist[pair[0] * vertex_number + pair[1]] = pair[2]; 
    } 
    fclose(file);
}

void output(char *outFileName) {
    FILE *outfile = fopen(outFileName, "w");
    for (int i = 0; i < vertex_number; ++i) {
        fwrite(&cpu_dist[i * vertex_number], sizeof(int), vertex_number, outfile);
    }
    fclose(outfile);
}


__global__ 
void phase1(int round, int vertex_number, int* cpu_dist){

    //Set share memory
    __shared__ int shared_device_dist[blocking_factor][blocking_factor];

    //To set shared memory location
    int x = threadIdx.x;
    int y = threadIdx.y;
    
    //To set shared memory location from CPU location
    int x_offset = round*blocking_factor + x;
    int y_offset = round*blocking_factor + y;
    
    // copy to shared memory from CPU memory
    if((y_offset<vertex_number) && (x_offset<vertex_number)){
        shared_device_dist[x][y]=cpu_dist[y_offset*vertex_number + x_offset]; 
    }else{
        shared_device_dist[x][y]=INF;
    }

    //Try to use Instruction-level parallelism
    // if((round*B + y+32)<vertex_number && (round*B + x+32)<vertex_number){
    //     shared_device_dist[x][y]=cpu_dist[y_offset*vertex_number + x_offset]; 
    //     shared_device_dist[x+][y]=cpu_dist[y_offset*vertex_number + x_offset]; 
    //     shared_device_dist[x][y]=cpu_dist[y_offset*vertex_number + x_offset]; 
    //     shared_device_dist[x][y]=cpu_dist[y_offset*vertex_number + x_offset]; 
    // }else{
    //     shared_device_dist[x][y]=INF;
    // }
    
    __syncthreads();

    #pragma unroll
    for(int k=0; k<blocking_factor; ++k){
        shared_device_dist[x][y]=min(shared_device_dist[x][y],shared_device_dist[x][k]+shared_device_dist[k][y]); 
        __syncthreads();
    }

    //Set global memory from device memory
    if((y_offset<vertex_number) && (x_offset<vertex_number)){
        cpu_dist[y_offset*vertex_number+x_offset] = shared_device_dist[x][y];
    }
   
}

__global__ void phase2(int round, int vertex_number, int* cpu_dist)
{
    //Set share memory
    __shared__ int shared_device_pivot[blocking_factor][blocking_factor];
    __shared__ int shared_device_dist[blocking_factor][blocking_factor];

    //To set shared memory location
    int x = threadIdx.x;
    int y = threadIdx.y;
    int x_block;
    int y_block;

    //To set shared memory location from CPU location
    int y_offset = round*blocking_factor + y;
    int x_offset = round*blocking_factor + x;
    int y_block_offset=blockIdx.x*blocking_factor + y;
    int x_block_offset=blockIdx.x*blocking_factor + x;
    

    // copy to shared memory from CPU memory
     if((y_offset<vertex_number) && (x_offset<vertex_number)){
        shared_device_pivot[x][y]=cpu_dist[y_offset*vertex_number+x_offset]; 
    }else{
        shared_device_pivot[x][y]=INF;
    }
    __syncthreads();

    //To handle same row
    if(blockIdx.y == 0){ 
        y_block = y_offset;
        x_block = x_block_offset; 
        if(y_block >= vertex_number || x_block >= vertex_number){
            return;
        }
        else{
            shared_device_dist[x][y]=cpu_dist[y_block*vertex_number + x_block]; 
        } 
        __syncthreads(); 
        #pragma unroll
        for(int k=0; k<blocking_factor; ++k){
            shared_device_dist[x][y]=min(shared_device_dist[x][y],shared_device_dist[x][k]+shared_device_pivot[k][y]); 
            __syncthreads();
        }
    }else{       
    //To handle same column    
        y_block = y_block_offset;
        x_block = x_offset;  
        if(y_block >= vertex_number || x_block >= vertex_number){
            return;
        }
        else{
            shared_device_dist[x][y]=cpu_dist[y_block*vertex_number + x_block]; 
        }  
        __syncthreads();        
        #pragma unroll
        for(int k=0; k< blocking_factor; ++k){
            shared_device_dist[x][y]=min(shared_device_dist[x][y],shared_device_pivot[x][k]+shared_device_dist[k][y]); 
            __syncthreads();
        }     
    }

    // Set global memory from device memory
    cpu_dist[y_block*vertex_number + x_block] = shared_device_dist[x][y];

}

__global__ 
void phase3(int round, int vertex_number, int* cpu_dist)
{
    //Set the result that update global memory
    int result;

    //Set shared memory
    __shared__ int shared_device_row[blocking_factor][blocking_factor];
    __shared__ int shared_device_column[blocking_factor][blocking_factor];
    
    //To set shared memory location
    int x = threadIdx.x;
    int y = threadIdx.y;
   
    //To set shared memory location from CPU location
    int x_block=blockIdx.x*blocking_factor + x;
    int y_block=blockIdx.y*blocking_factor + y;
    int y_block_column=blockIdx.y*blocking_factor + y;
    int x_block_row=blockIdx.x*blocking_factor + x;
    int y_block_row=round*blocking_factor + y;
    int x_block_column=round*blocking_factor + x;
    
    // copy to shared memory from CPU memory
    //row
    if(y_block_row<vertex_number && x_block_row<vertex_number){
        shared_device_row[x][y]=cpu_dist[y_block_row*vertex_number + x_block_row]; 
    }else{
        shared_device_row[x][y]=INF;
    }
    //column
    if(y_block_column<vertex_number && x_block_column<vertex_number){
        shared_device_column[x][y]=cpu_dist[y_block_column*vertex_number + x_block_column]; 
    }else{
        shared_device_column[x][y]=INF;
    }
    
    __syncthreads();

    if((y_block<vertex_number) && (x_block<vertex_number)){
        result = cpu_dist[y_block*vertex_number + x_block];      
        #pragma unroll
        for(int k=0; k<blocking_factor; ++k){
            result=min(result,shared_device_row[x][k]+shared_device_column[k][y]); 
        }
        cpu_dist[y_block*vertex_number + x_block] = result;
    }else{
        return;
    }
}




void block_FW(int B){
	
    //Set the times to run(vertex_number+blocking_factor-1/blocking_factor)
    int round = ceil(vertex_number, blocking_factor);
    //set share memory size
    int shareMemory_size=vertex_number*vertex_number*sizeof(int);
    //set each share memory array size
    int shareMemory_arraySize=blocking_factor*blocking_factor*sizeof(int);
    //ensure it can handle even if it's not divisible
    const int block_number=ceil(vertex_number,blocking_factor);

    // cudaHostRegister(cpu_dist, vertex_number*vertex_number*sizeof(int), cudaHostRegisterDefault);
    // cudaMallocHost(&cpu_dist,sizeof(int)*(vertex_number*vertex_number));
    //allocate memory of deivce
	cudaMalloc(&device_dist, shareMemory_size);
    //copy mamory from CPU to device
    cudaMemcpy(device_dist, cpu_dist, shareMemory_size, cudaMemcpyHostToDevice);
	
	for(int r = 0; r < round; r++){
		phase1<<< dim3(1,1), dim3(blocking_factor, blocking_factor) >>>(r, vertex_number,device_dist);
		phase2<<< dim3(block_number, 2), dim3(blocking_factor, blocking_factor) >>>(r, vertex_number,device_dist);
		phase3<<< dim3(block_number,block_number), dim3(blocking_factor, blocking_factor)>>>(r, vertex_number,device_dist);
    }

    //device to host
	cudaMemcpy(cpu_dist, device_dist, shareMemory_size, cudaMemcpyDeviceToHost);
  
}



int main(int argc, char* argv[])
{
    //total_start = std::chrono::steady_clock::now();
    //io_start = std::chrono::steady_clock::now();
    input(argv[1]);
    //io_end = std::chrono::steady_clock::now();
    // std::cout << "[Input_TIME] " << std::chrono::duration_cast<std::chrono::microseconds>(io_end - io_start).count() << "\n";
    //com_start = std::chrono::steady_clock::now();
    block_FW(blocking_factor);
    //com_end = std::chrono::steady_clock::now();
    // std::cout << "[COMPUTATION_TIME] " << std::chrono::duration_cast<std::chrono::microseconds>(com_end - com_start).count() << "\n";
    //io_start = std::chrono::steady_clock::now();
    output(argv[2]);
    //io_end = std::chrono::steady_clock::now();
    // std::cout << "[Output_TIME] " << std::chrono::duration_cast<std::chrono::microseconds>(io_end - io_start).count() << "\n";
    cudaFreeHost(cpu_dist);
    cudaFree(device_dist);
    // total_end = std::chrono::steady_clock::now();
    // std::cout << "[TOTAL_TIME] " << std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count() << "\n";

	return 0;
}