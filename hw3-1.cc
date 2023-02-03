#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <pthread.h>

const int INF = ((1 << 30) - 1);
const int V = 50010;
inline void input(char* inFileName);
inline void output(char* outFileName);

int vertex_number, edges_number; 
static int Dist[V][V];


void* floyd_algorithm(void* args){
    int k,i,j;
    for (int k = 0; k < vertex_number; ++k)
	{
        #pragma omp parallel for schedule(guided, 1) collapse(2)
		for (int i = 0; i < vertex_number; ++i)
		{
            // #pragma clang loop vectorize(enable) interleave(enable)
			for (int j = 0; j < vertex_number; ++j)
			{
				if (Dist[i][j] > Dist[i][k] + Dist[k][j] && Dist[i][k] != INF)
					Dist[i][j] = Dist[i][k] + Dist[k][j];
			}
		}
	}
    return NULL;
}


void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&vertex_number, sizeof(int), 1, file); 
    fread(&edges_number, sizeof(int), 1, file); 

    for (int i = 0; i < vertex_number; ++i) {
        #pragma omp parallel for
        for (int j = 0; j < vertex_number; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    #pragma omp parallel for
    for (int i = 0; i < edges_number; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    #pragma omp parallel for
    for (int i = 0; i < vertex_number; ++i) {
        for (int j = 0; j < vertex_number; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), vertex_number, outfile);
    
    }
    fclose(outfile);
}

int main(int argc, char* argv[]) {

    //detect how many CPUs 
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int cpu_num=CPU_COUNT(&cpu_set);
    pthread_t threads[cpu_num];
    int ID[cpu_num];

    // Read data
    input(argv[1]);
   
    for (int i = 0; i < cpu_num; i++) {
      pthread_create(&threads[i], NULL,floyd_algorithm, NULL);
      // ID[i]=i;
      // pthread_create(&threads[i], NULL,worker, (void*)&ID[i]);
    }

    for (int i = 0; i < cpu_num; i++) {
      pthread_join(threads[i], NULL);
    }

    output(argv[2]);
    return 0;
}