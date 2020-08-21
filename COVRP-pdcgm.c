#include <stdio.h>


#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h> 
#include<assert.h> 

/* Bring in the PDCGM/HOPDM library files */
#include "../../pdcgm/pdcgm.h"  

/* Bring in the elementary shortest path problem solver (RCESPP) */
#include "include/rcespp.h" 

/* Flag: Save to a file when active */ 
#define SAVE_TO_FILE

/* Optimality tolerance */
#define VRPTW_DELTA 1.E-6

/* Tolerance to zero */
#define VRPTW_TOL_ZERO 1.E-6
 
#define PI                      3.1415926
#define EARTH_RADIUS            6378.137      
#define maxSize 10000

/* Maximum number of columns that can be generated at a time */
int MAX_NUMBER_COLS_IN_A_CALL = 1;


int visit[maxSize];
typedef struct {
    int src;
    int dest; // dest Node
    int w; // edge weight 
}G_Edge; 
 
typedef struct{
    int id;
    int edge_num;
    G_Edge** edge_list; // edges
}G_Node; 

typedef struct{
    G_Node** node_list; // Nodes
    int n; // Node number 
    int e; // Edge number
}G_Graph;

//G_Graph* graph;

typedef struct {
int nodes[maxSize];
int n;//the number of nodes
}Path;

typedef struct
{
    double **distance;     //两个点之间距�?
    int *demand;        //需�?
    int q;   // 每辆车最大容�?
    int nC;  //customer的数�?
    int nV;  //车数�?
    int nN;  //total number of nodes in the problem (包括起点终点) 
    int nr;
}T_instance_data;

int insertNode(G_Graph* graph, int i) {
    // Judge if graph has been initialize.
    if (graph->node_list == NULL) {
        printf("graph hasn't been initialized!\n");
        return -1;
    }

    // create a G_Node named i.
    G_Node* p_node = (G_Node*) malloc(sizeof(G_Node));
    p_node->id = i;
    p_node->edge_num = 0;
    p_node->edge_list = (G_Edge**) malloc(maxSize * sizeof(G_Edge*));

    // add node into graph.
    graph->node_list[graph->n++] = p_node;
    return 0;
}

int insertEdge(G_Graph* graph, int src_id, int dest_id, int weight) {
    // Judge if graph has been initialize.
    if (graph->node_list == NULL) {
        printf("graph hasn't been initialized!\n");
        return -1;
    }

    int src_exist = 0;
    int dest_exist = 0;
    // check src/dest node exists
    for (int i = 0; i < graph->n; i++) {
        if (graph->node_list[i]->id == src_id) {
            src_exist = 1;
        } else if (graph->node_list[i]->id == dest_id) {
            dest_exist = 1;
        } else if (src_exist && dest_exist) {
            break;
        }
    }

    if (!src_exist) {
        insertNode(graph, src_id);
    }
    if (!dest_exist) {
        insertNode(graph, dest_id);
    }

    // find the src node and insert edge to it.
    for (int i = 0; i < graph->n; i++) {
        if (graph->node_list[i]->id == src_id) {
            G_Edge* p_edge = (G_Edge*) malloc(sizeof(G_Edge));
            p_edge->src = src_id;
            p_edge->dest = dest_id;
            p_edge->w = weight;
            graph->node_list[i]->edge_list[graph->node_list[i]->edge_num++] = p_edge;
        }
    }
    graph->e++;
    return 0;
}

Path* copyPath(Path* p_path) {
    Path* p_new_path = (Path*) malloc(sizeof(Path));
    p_new_path->n = p_path->n;
    for (int i = 0; i < p_path->n; i++) {
        p_new_path->nodes[i] = p_path->nodes[i];
    }
    return p_new_path;
}
int result_path_num = 0;
Path** bfs(G_Graph* graph, int start, int end) {
    Path** path_list = (Path**) malloc(maxSize * sizeof(Path*));
    Path** result_path_list = (Path**) malloc(maxSize * sizeof(Path*));
    int path_num = 0;
    for (int i = 0; i < graph->n; i++) {
        if (graph->node_list[i]->id == start) {
            // 找到起始�?
            G_Node* p_node = graph->node_list[i];
            for (int j = 0; j < p_node->edge_num; j++) {
                // 对于start节点，利用start引出的edge生成相同数量的path并加入path_list
                Path* p_path = (Path*) malloc(sizeof(Path));
                p_path->nodes[p_path->n++] = p_node->edge_list[j]->dest;
                path_list[path_num++] = p_path;
            }
        }
    }
    while(path_num > 0) {
        Path* p_path = path_list[--path_num];

        for (int i = 0; i < graph->n; i++) {
            if (graph->node_list[i]->id == p_path->nodes[p_path->n-1]) {
                G_Node* p_node = graph->node_list[i];
                if (p_node->edge_num == 0) {
                    break;
                } else {
                    for (int j = 0; j < p_node->edge_num; j++) {
                        Path* p_new_path = copyPath(p_path);
                        p_new_path->nodes[p_new_path->n++] = p_node->edge_list[j]->dest;
                        if (p_node->edge_list[j]->dest == end) {
                            result_path_list[result_path_num++] = p_new_path; 
                        } else {
                            path_list[path_num++] = p_new_path;
                        }
                    } 
                }
            }
        }
    }
    return result_path_list;
}






double radian(double d)
{
    return d * PI / 180.0;   //角度1˚ = π / 180
}
 
//计算距离
double get_distance(double lat1,double lng1, double lat2, double lng2)
{
    double radLat1 = radian(lat1);
    double radLat2 = radian(lat2);
    
    double a = radLat1 - radLat2;
    double b = radian(lng1) - radian(lng2);
 
    double dst = 20 * asin((sqrt(pow(sin(a / 2),2) + cos(radLat1) * cos(radLat2) * pow(sin(b / 2),2) )));
 
    dst = dst * EARTH_RADIUS;
    dst= round(dst * 10000) / 10000;
    return dst;
}
//Edge
typedef struct{
    int src;
    int dest;
    double weight;
}Edge;
//Graph
typedef struct{
    int V;
    int E;
    Edge* edge;
}Graph;
// create a graph
Graph* CreateGraph(int v,int e){
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->E = e;
    graph->V = v;
    graph->edge = (Edge*)malloc(e*sizeof(Edge));
    return graph;
}
//get solution
void print_path(int root,int *pre, int *solution) 
{  
    while(root!=pre[root]) //前驱  
    {  
        // printf("%d-->", root);
	if(root != 0) solution[root-1] = 1;
        root = pre[root];
    }  
    //if(root == pre[root])  
        // printf("%d\n", root);  
}  

int BellmanFord(Graph* graph,int *demand, int *solution, double *sub_opt, int*p, T_instance_data *instance)
{
    int v = graph->V;
    int e = graph->E;
    double dist[v];      //距离
    double shortest=10000;
    int capacity[v];  //各点容量
    int pre[v];        //谦虚节点
    int index[v];
    int i,j,src=0;
    int INT_MAX = 100000;
    // 初始�?
    for(i=0;i<v;++i){
        dist[i] = INT_MAX;
        pre[i] = 0;
        capacity[i] = 0;
    }//for
    dist[src] = 0;
    printf("BF_initial\n");
    // v-1次操�?
    Edge edge;
    int a,b;
    int shortest_node;
    double weight;
    
    G_Graph* graph2;
    Path** result_path = (Path**) malloc(maxSize * sizeof(Path*));
    graph2 = (G_Graph*)malloc(sizeof(G_Graph));
    graph2->n = 0;
    graph2->e = 0;
    graph2->node_list = (G_Node**) malloc(maxSize * sizeof(G_Node*));

    for(i=1;i<v;i++){
        // 对e条边进行松弛
        for(j=0;j<e;j++){
            edge = graph->edge[j];
            a = edge.src;
            b = edge.dest;
            weight = edge.weight;
            if(dist[a]!= INT_MAX && dist[a] + weight < dist[b]){
	        // printf("the edge %d to %d relax\n",a,b);
                if(capacity[a] + demand[b] > 45) continue;
                if(b == pre[a]) continue;   //前置节点的前置是本身（循环了�?
                dist[b] = dist[a]+weight;
                capacity[b] = capacity[a] + demand[b];
                pre[b] = a;
                insertEdge(graph2,a,b,weight);
		//printf("the edge %d to %d finish relax\n",a,b);
            }//if
        }//for
    }//for

    double t;
    int m;
    //for(i=1;i<v;i++) printf("the shortest distance for %d is %f\n",i,dist[i]);

    for (i=0;i<v;i++) index[i] = i;
    // bubble order
    for (i=0; i<v-1; ++i) {
		for (j=1; j<v-1-i; ++j) {
			if (dist[j] > dist[j+1]) {
			    t = dist[j];
			    dist[j] = dist[j+1];
			    dist[j+1] = t;
			    m = index[j];
			    index[j] = index[j+1];
			    index[j+1] = m;
			}
		}		
    }
    int root,flag_while,flag_for;
    double total_distance;
    flag_for = 0;
    for(i=1;i<v;i++){

    	result_path_num = 0;

	   	if(flag_for == 1) break;
	   	shortest_node = index[i]; // the end node of the shortest path
		*sub_opt = dist[i];       // the shortest distance
	   	printf("the shortest node is %d\n",shortest_node);
		printf("the shortest distance is %f\n",dist[i]);
	   	root = index[i];
	   	flag_while = 0;
		
	   	result_path= bfs(graph2,0,shortest_node);
		//printf("the result path number is %d\n",result_path_num);
		if(result_path_num != 1){
	   		for (int m = 0; m < result_path_num; m++) {
	   			total_distance = 0;
        			total_distance = instance->distance[0][result_path[m]->nodes[0]];
				printf("Node in each path %d--",result_path[m]->nodes[0]);
        			for(int k = 1; k < result_path[m]->n; k++){
            				total_distance = total_distance + instance->distance[result_path[m]->nodes[k-1]][result_path[m]->nodes[k]];
					printf("%d--",result_path[m]->nodes[k]);
   				}
				printf("\n");
				//printf("distance in %d  path %f\n",m,total_distance);

        			if(total_distance == dist[i]){
				printf("FIND!\n");
        				for(int k = 0; k < result_path[m]->n; k++){
						printf("%d--",result_path[m]->nodes[k]);
        					int tmp = result_path[m]->nodes[k];
        					if(k == 0){
        						pre[tmp] = 0;
        						continue;
        					}
        					pre[tmp] = result_path[m]->nodes[k-1];
        				}
					printf("\n");
					break;
        			}
   			}
		}
	   	 //判断生成的路里有负圈 有的话跳到第二短
	   	 while(root!=pre[root]) {
	    	flag_while += 1;
			root = pre[root];
	 	    if(flag_while>v-1)  break;
			if(root == 0) {
				flag_for = 1;
				break;
			 }
	   	 }//while
    }//for


    //for(i=1;i<v;i++){
    //printf("the shortest distance for %d is %f\n",i,dist[i]);
   // if(dist[i]<shortest) {
    //	shortest = dist[i];
    //	shortest_node = i;
    //	}
    //}
    //printf("the short_node 1 id %d\n",shortest_node);
    for(i=0;i<v;i++) p[i] = pre[i];
    // 检测负权回�?
    int isBack = 0;
    for(i=0;i<e;i++){
        edge = graph->edge[i];
        a = edge.src;
        b = edge.dest;
        weight = edge.weight;
        if(dist[a] != INT_MAX && dist[a]+weight < dist[b]){
            isBack = 1;
            break;
        }//if
    }//for
    // 打印�?
   // *sub_opt = dist[1];
    //printf("the prenode in Bellman are %d,%d,%d,%d,%d,%d,%d,%d,%d\n",p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9]);

    print_path(shortest_node,p,solution);
    return shortest_node;
}

int VRPTW_read_instance(   
    char file_name[],  
    T_instance_data *instance)  
{

    /* Statement of variables */
    FILE *myfile;
    char *ret, current_line[101];
    int i, j, aux, x, y;
    double  *xcoord, *ycoord; /* coordinates of the location of each customer */
    
    /* Open the file */
    myfile = fopen(file_name, "r");
    if(myfile == NULL) return 0;
    
    /* Read (and discard) the first four lines of the file (labels) */
    for(i=0;i<4;i++) ret = fgets(current_line, 100, myfile);
    
    /* Read the values nV and q (max number of vehicles and capacity) */
    aux = fscanf(myfile, "%d %d", &(instance->nV), &(instance->q));
    
    /* Guess the number of nodes (it is not explictly provided in the input file) */
    instance->nN = 102;

    //每个点的横纵坐标 需�?
    xcoord = (double *) malloc(sizeof(double) * (instance->nN));
    if(xcoord == NULL) return -1;
    ycoord = (double *) malloc(sizeof(double) * (instance->nN));
    if(ycoord == NULL) return -1;
    instance->demand = (int *) malloc(sizeof(int) * (instance->nN));
    if(instance->demand == NULL) return -1;

    /* Read (and discard) the next four lines of the file (labels) */
    for(i=0;i<5;i++) ret = fgets(current_line, 100, myfile);
    
    /* Read the length of the items */
    i=0;
    while( !feof(myfile) )
    {
        /* Read a line from the file */
        if( fscanf(myfile, "%d", &j) > 0 )
        {
            aux = fscanf(myfile, "%lf %lf %d", &(xcoord[i]), 
                &(ycoord[i]), &(instance->demand[i]));
            
            /* Increment the number of read customers */
            i++;
        
            /* Check if the allocated memory is enough */
            if( i >= instance->nN )
            {
                /* Reallocated arrays */
                instance->nN *= 2;
                xcoord = (double *) realloc(xcoord, sizeof(double) * instance->nN);
                if(xcoord == NULL) return -1;
                ycoord = (double *) realloc(ycoord, sizeof(double) * instance->nN);
                if(ycoord == NULL) return -1;
                instance->demand = (int *) realloc(instance->demand, sizeof(int) * instance->nN);
                if(instance->demand == NULL) return -1;
            }
        }
    }
    
    //�?算起 前面�?条记�?第一条是depot 那么i=9 就是代表具体数�?是几个就是几�?

    instance->nN = i+1; //仓库那个点相当于两个 最有一个的所有和第一个相�?见下
    instance->nC = i-1;

    /* Create a dummy copy of the depot in the last position */
    xcoord[instance->nN-1] = xcoord[0];
    ycoord[instance->nN-1] = ycoord[0];
    instance->demand[instance->nN-1] = instance->demand[0];
    
    /* Reallocate arrays in order to free unused memory */
    xcoord = (double *) realloc(xcoord, sizeof(double) * instance->nN);
    ycoord = (double *) realloc(ycoord, sizeof(double) * instance->nN);
    instance->demand = (int *) realloc(instance->demand, sizeof(int) * instance->nN);
    
    /* Close the file */
    fclose(myfile);
    
    /* Create the matrix of distances */
    instance->distance = (double **) malloc(sizeof(double *) * instance->nN);
    if(instance->distance == NULL) return -1;
    for(i=0; i<instance->nN; i++) instance->distance[i] = (double *) malloc(sizeof(double) * instance->nN);

    /* Fill the matrix of distances */
    for(i=0; i<instance->nN; i++) //nN*nN matrix
    {
        for(j=0; j<instance->nN; j++) 
        {
            instance->distance[i][j] = get_distance(xcoord[i],ycoord[i],xcoord[j],ycoord[j]);            
        }
    }

    /* Free auxiliar arrays */
    free(xcoord);
    free(ycoord);
    
    /* Return successfully */
    return 1;
}

int VRPTW_free_data(
    T_instance_data *instance)   /* pointer to the instance data */
{
    int i;
   
    if(instance == NULL) return 0;
    if(instance->demand != NULL) free(instance->demand);
    if(instance->distance != NULL) 
    {
        for(i=0; i<instance->nN; i++) free(instance->distance[i]);
        free(instance->distance);
    }
    
    /* Return successfully */
    return 1;
}

int VRPTW_set_initial_columns(
    PDCGM  *PDCGM_env,         /* PDCGM environment */
    T_instance_data *instance) /* Pointer to the instance data */
{
    int i, j, k, t, nc, max_m, max_n, max_nz, ret, 
        count, route_len, *solution;
    PDCGM_SMatrix_CW *M;  /* Matrix with the generated column(s) */
    double obj_value, route_cost, route_red_cost;

    /* Allocate the auxiliary matrix in memory */
    M = PDCGM_ALLOC(PDCGM_SMatrix_CW, 1);
    if(M == NULL) return -1;
    max_m = instance->nC + 2; /* number of rows in the RMP:
            (number of customers + convexity + obj function) */ 
    max_n = instance->nC;
    max_nz = max_n * max_m;
    PDCGM_set_SMatrix_CW(M, max_m, max_n, max_nz);
    
    
    /* Set initial dimensions */
    k=0;
    nc = 0;
    M->m = instance->nC + 2; /* number of rows in the RMP (number of customers + convexity + obj function) */
    M->n = 0; /* number of columns in the RMP (number of customers) */
    M->nz= 0; /* number of nonzeros */
    
    /* Set the initial columns */
    for(i=0; i<instance->nC; i++)
    {
        //每条路都派一辆车 每条路线的长度都是起点到客户*2
        /* Set a new column */
        M->clpnts[nc] = k + 1; /* -------> Fortran index */

        //顺序就是先道�?然后 convexity约束 然后是cost
        
        /* Set the entries of the column */
        M->coeff[k] = 1.0;
        M->rwnmbs[k] = i + 1; /* -------> Fortran index i是到哪个客户*/
        k++;
        
        /* Set the coefficient in the convexity constraints */
        M->coeff[k] = -1.0;
        M->rwnmbs[k] = instance->nC + 1; /* -------> Fortran index */
        k++;
        
        /* Set the cost corresponding to the generated column */
        /* Cost: distance from the source to the customer in {1, ..., nC} */
        M->coeff[k] =  instance->distance[0][i+1];
        M->rwnmbs[k] = instance->nC + 2; /* -------> Fortran index */
        k++;
        
        /* Set the homogeneous route */
        /*
            route[0] = 0      route_indices[0] = 1
            route[1] = 1      route_indices[1] = 2
            route[2] = 2      route_indices[2] = 3
        */
        
        /* Associate the route to the column */
        nc++;
    }
        
    /* End of columns! */
    M->clpnts[nc] = k + 1; /* -------> Fortran index */
    instance->nr = nc;    //the number of routes now

    /* Set the number of columns and nonzeros */
    M->n = nc;
    M->nz = k;
    
    /* Call the PDCGM function that add the new columns to the current RMP */
    nc = PDCGM_add_columns(PDCGM_env, M, NULL);

    /* Clean the memory */
    PDCGM_free_SMatrix_CW(M);
    PDCGM_FREE(M);
    
    /* If the answer is negative, report the ERROR */
    if(nc < 0) return -1;
    
    /* Return successfully */
    return 0;
}

int Slover(T_instance_data *instance, int *solution, double *sub_opt, int*p)
{
    printf("solver\n");
    int v = instance->nN-1;
    int e = (instance->nN - 3) * (instance->nN - 2) + (instance->nN -2);  //待定
    int demand[v];
    int ne = 0;
    int *de;
    int i,j;
    int short_node;
    Graph* graph = CreateGraph(v,e);
    //构造图 写入demand
    instance->distance[0][instance->nN-1] = 10000;
    for(j=0;j<instance->nN-1;j++)
    {
        demand[j] = instance->demand[j];
        for(i=1;i<instance->nN-1;i++){
            if(i == j) continue;
            graph->edge[ne].src = j;
            graph->edge[ne].dest = i;
            graph->edge[ne].weight = instance->distance[j][i];
	    //printf("create the graph: from %d to %d is %f\n",j,i,instance->distance[j][i]);
            ne++;
        }//for i
    }//for j
    //终点节点
    de = demand;
    short_node  = BellmanFord(graph,de,solution,sub_opt,p,instance);
    return short_node;
}
static short VRPTW_oracle(
    double *primal_violation,   /* total violation of the generated constraints, if any */
    double *dual_violation,     /* total violation of the generated columns , if any */
    PDCGM  *PDCGM_env,          /* PDCGM environment */
    void   *instance_data)      /* instance data */
{    
	/* Convert the instance_data to the local type */
    T_instance_data *instance_original = (T_instance_data *) instance_data;


    int i, j, k, t, q, ret, nc, *solution,*p, 
    route_len, count, max_m, max_n, max_nz,
    short_node = 0;
        
    double *u,route_cost=0,*sub_opt, opt_initial;
        
    PDCGM_SMatrix_CW *M; 
    
    PDCGM_set_change_bounds(PDCGM_env, 0);
    PDCGM_set_reliable_answer(PDCGM_env, 1);


    //new struct
    T_instance_data tmp;
    tmp.demand = (int *) malloc(sizeof(int) * instance_original->nN+1);
    if(tmp.demand == NULL) return -1;
    tmp.distance = (double **) malloc(sizeof(double *) * instance_original->nN);
    if(tmp.distance == NULL) return -1;
    for(i=0; i<instance_original->nN; i++) tmp.distance[i] = (double *) malloc(sizeof(double) * instance_original->nN);

    for(i=0;i<instance_original->nN;i++){
    	tmp.demand[i] = instance_original->demand[i];
    	for(j=0;j<instance_original->nN;j++){
    		tmp.distance[i][j] = instance_original->distance[i][j];
		//printf("the original distance from %d to %d is %f\n",i,j,tmp.distance[i][j]);
    	}
    }
    tmp.q = instance_original->q;
    tmp.nC = instance_original->nC;
    tmp.nN = instance_original->nN;
    tmp.nV = instance_original->nV;
    tmp.nr = instance_original->nr;


    /* Set initial values */
    PDCGM_env->ncols_last = 0;
    *dual_violation = 0.0;
    nc = 0;
    ret = 0;
    sub_opt = &opt_initial;
    *sub_opt = 0.0;
    
   // sub_opt = 0;
    
    /* Get the current dual solution */
    u = PDCGM_get_dual_solution(PDCGM_env);
    //printf("u is %f, %f, %f,%f,%f,%f,%f,%f\n",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7]);
    /* Allocate the auxiliary matrix in memory */
    M = PDCGM_ALLOC(PDCGM_SMatrix_CW, 1);
    if(M == NULL) return -1;
    max_m = instance_original->nC + 2; /* number of rows in the RMP:
            (number of customers + number of cuts (SRC) + convexity + obj function) */ 
    max_n = 1;
    max_nz = max_n * max_m;
    PDCGM_set_SMatrix_CW(M, max_m, max_n, max_nz);
    M->m = max_m;
    M->n = M->nz = 0;
    //update the distance function (instance,u)  
    for(i=0;i<instance_original->nN-1;i++){
    	for(j=1;j<instance_original->nN-1;j++){
    		if(j==i) continue;
    		tmp.distance[i][j] = tmp.distance[i][j] - u[j-1];
		//printf("from %d to %d is %f\n",i,j,tmp.distance[i][j]);
    	}
    }
    /* Solve the subproblem  */      
    k=0;
    nc = 0;

    /* Clean the solution array */
    solution = (int*) malloc(sizeof(int) * 100);
    for(i=0; i < instance_original->nC; i++) solution[i] = 0;
    p = (int*) malloc(sizeof(int) * 100);
    for(i=0; i < instance_original->nC; i++) p[i] = 0;


    /* Associate the route to the column */
//    columns_links[nc] = instance_original->nr;
    /* Get the column corresponding to the next stored route */
    short_node = Slover(&tmp,solution,sub_opt,p);
    //printf("the solution is %d,%d,%d,%d,%d,%d,%d\n",solution[0],solution[1],solution[2],solution[3],solution[4],solution[5],solution[6]);
    *dual_violation = *sub_opt - u[instance_original->nC]; 
    printf("up the dual violation is %f\n",*dual_violation); 
    //printf("the prenode are %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10]);
    //printf("the short node is %d\n",short_node);
    //calculate the route cost
    int h;
    h = 0;
    while(short_node!=p[short_node]) //前驱  
    {  
        // printf("%d-->", root);
        if(short_node == 0) break;
        h = p[short_node];
        route_cost = route_cost + instance_original->distance[short_node][h];
        short_node = p[short_node];
    }  
    printf("the cost is %f\n",route_cost);

    //route_cost = route_cost + instance_original->distance[h][0];


    /* Set in the sparse representation */
    if(*dual_violation < 0)
    {
        M->clpnts[nc] = k + 1; /* -------> Fortran index */

        /* Set the entries that correspond to the routes */
        for(i=0; i < instance_original->nC; i++)
        {
        	//printf("solution node %d is %d \n",i,solution[i]);
            /* Check if it is a nonzero entry */
            if(solution[i] > 0.5) 
            {
	        printf("customer %d solution %d\n",i+1,solution[i]);
                /* Set a new entry in the column */
                M->coeff[k] = (double) solution[i];
                M->rwnmbs[k] = i + 1; /* -------> Fortran index */
                k++;
                /* Reset the solution vector */             
                solution[i] = 0;
            }
        }

        /* Set the coefficient of the convexity constraint */
        M->coeff[k] = -1.;
        M->rwnmbs[k] = instance_original->nC + 1; /* -------> Fortran index */
        k++;

        /* Set the cost corresponding to the generated column */
        M->coeff[k] = route_cost;
	printf("the route cost is %f\n",route_cost);
        M->rwnmbs[k] = instance_original->nC + 2; /* -------> Fortran index */
        k++;
        
        /* Increase the number of columns */
        nc = 1;
    }
    
    //else
    //{
        /* Set a column with only one entry (in the convexity constraint) */
      //  M->coeff[k] = -1.0;
        //M->rwnmbs[k] = instance_original->nC + 1; /* -------> Fortran index */
       // k++;
        
        /* Set the number of generated columns */
       // nc = 1;
   // }
    /* End of columns! */
    M->clpnts[nc] = k + 1; /* -------> Fortran index */

    /* Set the number of columns and nonzeros */
    M->n = nc;
    M->nz = k; 

    /* Update the objective value of the oracle */
    if(*dual_violation < 1.0e-6 ) *dual_violation = *dual_violation;
    // *dual_violation = PDCGM_env->rhs[PDCGM_env->dim] * (*dual_violation);
    else *dual_violation = 0.0;


    /* Add the generated columns to the coefficient matrix  */
    if(M->n > 0)
    {
        /* Call the PDCGM function that add the new columns to the current RMP */
        nc = PDCGM_add_columns(PDCGM_env, M, NULL);

        /* If the answer is negative, report the ERROR */
        if( nc < 0 ) return -1;     
    }
       

    /* If nc > 0, then new columns were generated. */
    /* Otherwise, check if a new run is possible */
    if(nc == 0) *dual_violation = 0.0;
    printf("the dual violation is %f\n",*dual_violation);
    
    /* Clean the memory */
    PDCGM_free_SMatrix_CW(M);
    PDCGM_FREE(M);
    
    /* Return 1: COLUMN GENERATION */
    return 1;
}

int main(int arg_c, char *arg_v[])
{
    FILE *out_file;

    char file_name[1000];
    int i, j, k, p;
    short ret; 

    /* Variable to store the instance data */
    T_instance_data VRPTW_instance;
    
    int dim,                 /* number of kept constraints in the RMP */
        nb_cvxty_constr;    /* number of convexity constraints in the RMP */

    PDCGM *VRPTW;
    PDCGM_SMatrix_CW M;
    
    double *u_initial, *b, *lo_box, *up_box, aux_double;
    short  *rw_type;
    
    int    max_outer = 1000;

    /* Check if the file was specified */
    if (arg_c > 1) 
    {
        /* Copy the parameters */
        strcpy(file_name, arg_v[1]);
        if (arg_c > 2 && atoi(arg_v[2]) > 0) MAX_NUMBER_COLS_IN_A_CALL = atoi(arg_v[2]);
    }
    else 
    {
        printf("\n-- ERROR: File name not specified --\n");
        return 0;
    } 
    
    /* Read the problem */
    if(VRPTW_read_instance(file_name, &VRPTW_instance) < 1)
    {
        printf("\nThe file '%s' could not be read.\n", file_name);
        return 0;
    }
    
    #ifdef SAVE_TO_FILE
        /* Save to an output file */
        out_file = fopen("output-vrptw-pdcgm.txt", "a");
        fprintf(out_file, "\n%s ", file_name);
        fflush(out_file); 
        fclose(out_file);
    #endif
    
    /* Set the dimensions of the instance */
    dim = VRPTW_instance.nC;
    nb_cvxty_constr = 1;
    
    
    /* initial guess u, b, rw_type and boxes: */
    b = PDCGM_ALLOC(double, dim + nb_cvxty_constr + 1);
    rw_type = PDCGM_ALLOC(short, dim + nb_cvxty_constr + 1);
    lo_box = PDCGM_ALLOC(double, dim);
    up_box = PDCGM_ALLOC(double, dim);
    u_initial = PDCGM_ALLOC(double, dim + nb_cvxty_constr + 1);
    for (i=0; i<dim; i++) 
    {
        /* Type of the constraint */
        rw_type[i] = GREATER_EQUAL;
        
        /* RHS of the constraint */
        b[i] = 1.0;
        
        /* Lower and upper bound of the associated dual variable */
        lo_box[i] = 0;
        up_box[i] = 30; // VRPTW_instance.distance[0][i+1];
	printf("upbox is %f\n",up_box[i]);
        
        /* An initial guess for the associated dual variable */
//	u_initial[i] = 1.0;
    }
    //up_box[4] = up_box[4]+5;
    printf("the run numbers of main \n");
    //up_box[6] = VRPTW_instance.distance[0][7];
    for (i=dim; i<dim+nb_cvxty_constr; i++) 
    {
        /* Type of the constraint */
        rw_type[i] = GREATER_EQUAL;
        
        /* RHS of the constraint */
        b[i] = -dim;
        /* An initial guess for the associated dual variable */
        u_initial[i] = 0;
	up_box[i] = 1;
	lo_box[i] = 0;
    }
    u_initial[dim+nb_cvxty_constr] = 0.0;
    b[dim+nb_cvxty_constr] = 0.0;
    rw_type[dim+nb_cvxty_constr] = OBJECTIVE;

    /* POPULATE THE INTERNAL DATA STRUCTURE OF PDCGM */ 
    VRPTW = PDCGM_set_data( 
                dim,  /* number of kept constraints in the RMP */
                nb_cvxty_constr,  /* number of convexity constraints in the RMP */
                (dim + nb_cvxty_constr + 1),  /* maximum number of nonzeros in a column */
                MAX_NUMBER_COLS_IN_A_CALL,    /* maximum number of columns generated by the oracle */
                max_outer,     /* maximum number of outer iterations */
                NULL, //u_initial, /* initial guess of the dual solution (may be NULL) */
                b,    /* RHS of each constraint in the RMP */
                rw_type,  /* type of each constraint (row) in the RMP */
                lo_box,   /* lower bound vector of the DUAL variables in the RMP */
                up_box,   /* upper bound vector of the DUAL variables in the RMP */
                &VRPTW_instance);   /* instance data */
    
    if(VRPTW == NULL)   
    {
        printf("-- ERROR: PDCGM_set_data() returned NULL --\n");
        fflush(stdout);
        return 0;
    }
    
    /*--------------------------------------------------------*/
    /* SOLVE THE PROBLEM USING COLUMN GENERATION              */
    /*--------------------------------------------------------*/
     
    /* Set the optimality tolerance for the column generation algorithm */
    PDCGM_set_delta(VRPTW, VRPTW_DELTA);   
     
    /* Set the degree of optimality (parameter D > 1) */
    PDCGM_set_degree_of_optimality(VRPTW, 10.0);
    
    /* Set the verbose mode (how much information is printed) */
    PDCGM_set_verbosity(VRPTW, 1);
    
    /* Disable cut elimination on insertion */
    PDCGM_set_column_elimination(VRPTW, 1);
    
    /* The columns corresponding to homogeneous routes are never removed */
    VRPTW->start_from_reduce_matrix = VRPTW_instance.nN-2;
    
    /* Set epsilon_max */
    PDCGM_set_max_opt_tol(VRPTW, 0.5);
    
    /* Preloaded columns */
    VRPTW_set_initial_columns(VRPTW, &VRPTW_instance);
    /* CALL THE COLUMN GENERATION PROCEDURE */
    ret = PDCGM_solve_MP(VRPTW, VRPTW_oracle);
    
    /* Analyse the answer obtained from PDCGM */
    printf("PDCGM has finished with code = %d\n\n", ret);
    fflush(stdout); 
    double *lambda;
    int lambda_n;
    PDCGM_get_pointer_to_master_solution(VRPTW, &lambda, &lambda_n);
    for(i=0;i<lambda_n;i++)
    {
    printf("%d variable is %f\n",i,lambda[i]);
    }
    #ifdef SAVE_TO_FILE 
        /* Save to an output file */
        out_file = fopen("output-vrptw-pdcgm.txt", "a");
        //fprintf(out_file, "\t%d \t%d \t%d \t%d ", VRPTW_instance.nC, VRPTW_instance.q, (VRPTW->G)->n, (VRPTW->G)->m-dim-2);
        fprintf(out_file, "\t%lf \t%lf \t%1.6E \t%d \t%lf \t%lf \t%lf ",  
            VRPTW->lowerBnd * 0.1, VRPTW->upperBnd * 0.1, VRPTW->rel_gap, VRPTW->outer,
            VRPTW->cputime_RMP, VRPTW->cputime_oracle, VRPTW->cputime_CG);
        fflush(out_file);
        fclose(out_file);
    #endif
    
    /* Clean the memory before returning */
    PDCGM_free(VRPTW);
    free(VRPTW);
    free(u_initial);
    free(b);
    free(rw_type);
    free(lo_box);
    free(up_box);
    VRPTW_free_data(&VRPTW_instance);
    
    return 0; 
}


