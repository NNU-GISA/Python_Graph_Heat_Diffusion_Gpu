__kernel void laplacian_filter(__global float *gray, __global float *new_gray, __global int *ngbrs , __global float *wgts, const int k, const float dt)

{ 
	int n, i, j;
	n = get_global_size(0);
	i = get_global_id(0);
	float tmp;
	float tmp2;	
	float tmp3;	
 	/*<for harcoded k the code is very easy> the forward euler method comes here */
	
	tmp = 0.0;
	tmp2 = 0.0;
	tmp3 = 0.0;
	for (j = 0; j < k; )
	{
		tmp += wgts[i*k + j] * gray[ngbrs[i*k+j]];
		tmp2 += wgts[i*k + j];	

		j = j + 1;
	} 

/*	you can think with pen and paper or you can look at the github,  check the barrier time DONE , check the dt thing DONE, check if can yousymmetrize the graph NO, check on the image file */
	tmp3 = gray[i]*(1.0f - dt* tmp2) + dt*tmp ;
	new_gray[i] = tmp3;
	barrier(CLK_GLOBAL_MEM_FENCE);

 }
