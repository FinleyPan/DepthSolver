/**
 * definitions for census kernel
 */
#define SMEM_SIZE (CENSUS_WINDOW_HEIGHT + 1)

/**
 * definitions for cost volume kernels
 */
#define PROJ_ARRAY_SIZE 12
#define SIZE_DISP_SUBGROUP (DISP_RANGE / NUM_DISP_SUBGROUPS)
#define HALF_W CENSUS_WINDOW_WIDTH/2
#define HALF_H CENSUS_WINDOW_HEIGHT/2

#define CLR_DISP_STEP (DISP_RANGE / 16)

 /**
 * definitions for cost aggregation kernel
 */
#define COST_BUFF_WIDTH (DISP_RANGE/2)
#define NUM_AGG_BUFF_COLS (DISP_RANGE/4)
#define NUM_OBL_PATHS (IMG_H + IMG_W - 1)

__kernel void census_transform_kernel(global ushort* dest, 
									  global const uchar* src){
const int half_w = CENSUS_WINDOW_WIDTH / 2;
const int half_h = CENSUS_WINDOW_HEIGHT / 2;

__local uchar shared_mem[SMEM_SIZE][CENSUS_BLOCK_SIZE];
const int tid = get_local_id(0);
const int x0 = get_group_id(0)*(CENSUS_BLOCK_SIZE - CENSUS_WINDOW_WIDTH + 1) - half_w;
const int y0 = get_group_id(1)*LINES_PER_BLOCK;
//initialize shared memory
for(int i = 0; i < CENSUS_WINDOW_HEIGHT; i++){
	const int x = x0 + tid; 
	const int y = y0 - half_h + i;
	uchar val = 0;
	if(0 <= x && x < IMG_W && 0 <= y && y < IMG_H)
		val = src[x + y*IMG_W];
	shared_mem[i][tid] = val;
}
barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
for(int i=0; i < LINES_PER_BLOCK; ++i){
	if( i+1 < LINES_PER_BLOCK){
		const int x = x0 + tid;
		const int y = y0 + half_h + i + 1;
		uchar val = 0;
		if(0 <= x && x < IMG_W && 0 <= y && y < IMG_H)
			val = src[x + y*IMG_W];
		const int smem_x = tid;
		const int smem_y = (CENSUS_WINDOW_HEIGHT + i) % SMEM_SIZE;
		shared_mem[smem_y][smem_x] = val;
	}
	if(half_w <= tid && tid < CENSUS_BLOCK_SIZE - half_w){
		const int x = x0 + tid;
		const int y = y0 + i;
		if(half_w <= x && x < IMG_W - half_w && 
		   half_h <= y && y < IMG_H - half_h){
			const int smem_x = tid;
			const int smem_y = (half_h + i) % SMEM_SIZE;
			ushort f = 0;
			for(int dy = -half_h; dy < 0; ++dy){
				const int smem_y1 = (smem_y + dy + SMEM_SIZE) % SMEM_SIZE;
				const int smem_y2 = (smem_y - dy + SMEM_SIZE) % SMEM_SIZE;
				for(int dx = -half_w; dx <= half_w; ++dx){
					const int smem_x1 = smem_x + dx;
					const int smem_x2 = smem_x - dx;
					const uchar a = shared_mem[smem_y1][smem_x1];
					const uchar b = shared_mem[smem_y2][smem_x2];					
					f = (f << 1)|(a > b);
				}
			}
			for(int dx = -half_w; dx < 0; ++dx){
				const int smem_x1 = smem_x + dx;
				const int smem_x2 = smem_x - dx;
				const uchar a = shared_mem[smem_y][smem_x1];
				const uchar b = shared_mem[smem_y][smem_x2];
				f = (f << 1)|(a > b);				
			}
			dest[x + y*IMG_W] = f;	
		}	 
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}
}

__kernel void clear_cost_volume_kernel(__global ushort16* d_cost_volume,
									   __global ushort16* d_aggre_cost){
	int x = get_global_id(0) / CLR_DISP_STEP;	
	int y = get_global_id(1);
	int base = (y * IMG_W * DISP_RANGE + x * DISP_RANGE);

	if(x > IMG_W - 1 || y > IMG_H - 1)
		return;
#pragma unroll
	for(int i = 0; i < CLR_DISP_STEP; i++){
		d_cost_volume[base / 16 + i] = (ushort16)(CLR_RAW_COST);
		d_aggre_cost[base / 16 + i] = (ushort16)(CLR_AGG_COST);
	}
}

__kernel void make_cost_volume_kernel(__global ushort* cost_volume,
									  __global const ushort* curr_census,
									  __global const ushort* ref_census,									  
									  __global const float* lookup_depth,
									  __global const float* proj_params){
	__local float proj_array[PROJ_ARRAY_SIZE];
	__local float depth_samples[DISP_RANGE];
	int loc_x = get_local_id(0);
	int base_disp_level = (loc_x % NUM_DISP_SUBGROUPS) * SIZE_DISP_SUBGROUP;
	int x = get_global_id(0) / NUM_DISP_SUBGROUPS;
	int y = get_global_id(1);	
	if(get_local_id(1) == 0 ){
		if(loc_x < PROJ_ARRAY_SIZE)
			proj_array[loc_x] = proj_params[loc_x];
		if(loc_x < DISP_RANGE)				
			depth_samples[loc_x] = lookup_depth[loc_x];		
	}		
	barrier(CLK_LOCAL_MEM_FENCE);	

	ushort left_val = curr_census[y * IMG_W + x];
#pragma unroll
	for(int l = 0; l < SIZE_DISP_SUBGROUP; l++){
		int level = l + base_disp_level;
		int idx = y*IMG_W*DISP_RANGE + x*DISP_RANGE + level;
		if(HALF_W <= x && x < IMG_W  - HALF_W &&
		   HALF_H <= y && y < IMG_H - HALF_W) {
			float px_f = depth_samples[level]*(proj_array[0]*x + proj_array[3]*y + 
			proj_array[6]) + proj_array[9];
			float py_f = depth_samples[level]*(proj_array[1]*x + proj_array[4]*y + 
			proj_array[7]) + proj_array[10];
			float pz_f = depth_samples[level]*(proj_array[2]*x + proj_array[5]*y + 
			proj_array[8]) + proj_array[11];		
			int px = (int)(px_f / pz_f + 0.5f);
			int py = (int)(py_f / pz_f + 0.5f);
			if(0 <= px && px < IMG_W && 
		   	   0 <= py && py < IMG_H){
					ushort right_val = ref_census[py * IMG_W + px];
					cost_volume[idx] = popcount(left_val ^ right_val);			   
				}				
		}
	}
}

inline void min_warp_dp(__local ushort* min_cost_buff,
					    int local_index, int head_index) {    
#pragma unroll
    for (int offset = NUM_AGG_BUFF_COLS / 2;
        offset > 0;
        offset = offset / 2) {
        if (get_local_id(0) < offset) {
            ushort right = min_cost_buff[local_index + offset];
            ushort left = min_cost_buff[local_index];
            min_cost_buff[local_index] = min(left, right);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

//perform dynamic programming for one step
inline void dp_one_step(__global ushort4 *d_aggre_cost,
						__global const ushort4 *d_cost_volume,
						__local ushort2 *agg_cost_buff,
    					__local ushort *min_cost_buff,
						int idx, int var_P2){	
    int k = get_local_id(0); // (32 disp) k in [0..7]
	//linear index in cost_buffs
	int local_idx = COST_BUFF_WIDTH * get_local_id(1) + 2 * k;
	//linear index of row header element in the min_cost_buff
	int head_index = get_local_id(1) * NUM_AGG_BUFF_COLS;
	int local_index = get_local_id(0) + head_index;	

	ushort4 cost_curr = d_cost_volume[idx * DISP_RANGE / 4 + k];

    ushort2 cost_curr_L = (ushort2)(cost_curr.y, cost_curr.x); 
    ushort2 cost_curr_H = (ushort2)(cost_curr.w, cost_curr.z); 

	ushort2 cost_buff_curr_L = agg_cost_buff[local_idx + 0];
	ushort2 cost_buff_curr_H = agg_cost_buff[local_idx + 1];
    ushort2 cost_buff_prev, cost_buff_next;
    
	if((local_idx + 2) % COST_BUFF_WIDTH ==0 )
		cost_buff_next = (ushort2)(MAX_COST);
	else
		cost_buff_next = agg_cost_buff[local_idx + 2];
    
	if(local_idx % COST_BUFF_WIDTH == 0)
		cost_buff_prev = (ushort2)(MAX_COST);
	else
		cost_buff_prev = agg_cost_buff[local_idx - 1];
    
	ushort2 v_cost0_L = cost_buff_curr_L;
	ushort2 v_cost0_H = cost_buff_curr_H;
    ushort2 v_cost1_L = (ushort2)(cost_buff_curr_L.y, cost_buff_prev.x);
    ushort2 v_cost1_H = (ushort2)(cost_buff_curr_H.y, cost_buff_curr_L.x);    
    ushort2 v_cost2_L = (ushort2)(cost_buff_curr_H.y, cost_buff_curr_L.x);
    ushort2 v_cost2_H = (ushort2)(cost_buff_next.y, cost_buff_curr_H.x);

    ushort2 v_min_cost = (ushort2)(min_cost_buff[head_index],
								   min_cost_buff[head_index]);
	ushort2 v_cost3 = add_sat(v_min_cost, (ushort2)(var_P2, var_P2));
    
	v_cost1_L = add_sat(v_cost1_L, (ushort2)(P1));
	v_cost2_L = add_sat(v_cost2_L, (ushort2)(P1));
	v_cost1_H = add_sat(v_cost1_H, (ushort2)(P1));
	v_cost2_H = add_sat(v_cost2_H, (ushort2)(P1));
    
	ushort2 v_tmp_a_L = min(v_cost0_L, v_cost1_L);
    ushort2 v_tmp_a_H = min(v_cost0_H, v_cost1_H);    
	ushort2 v_tmp_b_L = min(v_cost2_L, v_cost3);
	ushort2 v_tmp_b_H = min(v_cost2_H, v_cost3);
    
	ushort2 cost_tmp_L = cost_curr_L + min(v_tmp_a_L, v_tmp_b_L) - v_min_cost;
	ushort2 cost_tmp_H = cost_curr_H + min(v_tmp_a_H, v_tmp_b_H) - v_min_cost;
        
	d_aggre_cost[DISP_RANGE * idx / 4 + k] = add_sat(d_aggre_cost[DISP_RANGE * idx / 4 + k],
						 (ushort4)(cost_tmp_L.y, cost_tmp_L.x, cost_tmp_H.y, cost_tmp_H.x));
    agg_cost_buff[local_idx + 0] = cost_tmp_L;
    agg_cost_buff[local_idx + 1] = cost_tmp_H;
	ushort2 cost_tmp = min(cost_tmp_L, cost_tmp_H);

	min_cost_buff[local_index] = min(cost_tmp.x, cost_tmp.y);
	//wait until min_cost_buff filled
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	min_warp_dp(min_cost_buff, local_index, head_index);
}

inline void clear_local_cost_buff(__local ushort2* agg_cost_buff,
								  __local ushort* min_cost_buff,
								  ushort agg_val, ushort min_val){

    agg_cost_buff[get_local_id(1)*COST_BUFF_WIDTH + get_local_id(0)*2 + 0]=(ushort2)(agg_val);
 	agg_cost_buff[get_local_id(1)*COST_BUFF_WIDTH + get_local_id(0)*2 + 1]=(ushort2)(agg_val);
	min_cost_buff[get_local_id(1)*NUM_AGG_BUFF_COLS + get_local_id(0)] = min_val;									  
}


 __kernel void aggre_cost_horizontal_kernel(__global ushort4 *d_aggre_cost,
	 								        __global const ushort4* d_cost_volume){												
	__local ushort2 agg_cost_buff[COST_BUFF_WIDTH * AGG_STRIDE];
	__local ushort min_cost_buff[NUM_AGG_BUFF_COLS * AGG_STRIDE];
	int i = get_group_id(0) * AGG_STRIDE + get_local_id(1);

	//clear shared memory
	clear_local_cost_buff(agg_cost_buff, min_cost_buff, CLR_AGG_COST, CLR_AGG_COST);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(i < IMG_H){
		//from left to right
		for(int j=0; j < IMG_W; j++){
			int idx = i * IMG_W + j; // linear index in image			
			dp_one_step(d_aggre_cost, d_cost_volume, agg_cost_buff, min_cost_buff, idx, P2);	
		}		
	
		clear_local_cost_buff(agg_cost_buff, min_cost_buff, CLR_AGG_COST, CLR_AGG_COST);
		barrier(CLK_LOCAL_MEM_FENCE);

		//from right to left	
		for(int j=0; j < IMG_W; j++){
			int idx = i * IMG_W + (IMG_W - j - 1);			
			dp_one_step(d_aggre_cost, d_cost_volume, agg_cost_buff, min_cost_buff, idx, P2);
		}
	}		
}

__kernel void aggre_cost_vertical_kernel(__global ushort4 *d_aggre_cost,
	 								     __global const ushort4* d_cost_volume){
	__local ushort2 agg_cost_buff[COST_BUFF_WIDTH * AGG_STRIDE];
	__local ushort min_cost_buff[NUM_AGG_BUFF_COLS * AGG_STRIDE];
	int j = get_group_id(0) *  AGG_STRIDE + get_local_id(1);

	//clear shared memory
	clear_local_cost_buff(agg_cost_buff, min_cost_buff, CLR_AGG_COST, CLR_AGG_COST);
	barrier(CLK_LOCAL_MEM_FENCE);

	//from top to down
	if(j < IMG_W - 1){
		for(int i=1; i < IMG_H; i++){
			int idx = i * IMG_W + j;		
			dp_one_step(d_aggre_cost, d_cost_volume, agg_cost_buff, min_cost_buff, idx, P2);
		}

		clear_local_cost_buff(agg_cost_buff, min_cost_buff, CLR_AGG_COST, CLR_AGG_COST);
		barrier(CLK_LOCAL_MEM_FENCE);

		//from down to top
		for(int i=1; i < IMG_H; i++){
			int idx = (IMG_H - i -1) * IMG_W + j;		
			dp_one_step(d_aggre_cost, d_cost_volume, agg_cost_buff, min_cost_buff, idx, P2);
		}
	}
}

__kernel void aggre_cost_oblique0_kernel(__global ushort4 *d_aggre_cost,
	 								     __global const ushort4* d_cost_volume){
	__local ushort2 agg_cost_buff[COST_BUFF_WIDTH * AGG_STRIDE];
	__local ushort min_cost_buff[NUM_AGG_BUFF_COLS * AGG_STRIDE];											 
	int path_idx = get_group_id(0) * AGG_STRIDE + get_local_id(1);
	if(path_idx >= NUM_OBL_PATHS) 
		return;	

	//from topleft to downright
	int i = max(0, -(IMG_W - 1) + path_idx);
	int j = max(0, IMG_W - 1 - path_idx);
	
	//clear shared memory		
	clear_local_cost_buff(agg_cost_buff, min_cost_buff, INVALID_COST, INVALID_COST);	
	barrier(CLK_LOCAL_MEM_FENCE);		

	while(i < IMG_H && j < IMG_W){		
		int idx = i * IMG_W + j;		 		
		dp_one_step(d_aggre_cost, d_cost_volume, agg_cost_buff, min_cost_buff, idx, P2);		
		++i; ++j;			 
	}
		
	//from downright to topleft	
	clear_local_cost_buff(agg_cost_buff, min_cost_buff, INVALID_COST, INVALID_COST);
	barrier(CLK_LOCAL_MEM_FENCE);

	while(--i >= 0 && --j >= 0){		
		int idx = i * IMG_W + j;			
		dp_one_step(d_aggre_cost, d_cost_volume, agg_cost_buff, min_cost_buff, idx, P2);		
	}
}

__kernel void aggre_cost_oblique1_kernel(__global ushort4 *d_aggre_cost,
	 								     __global const ushort4* d_cost_volume){
	__local ushort2 agg_cost_buff[COST_BUFF_WIDTH * AGG_STRIDE];
	__local ushort min_cost_buff[NUM_AGG_BUFF_COLS * AGG_STRIDE];											 
	int path_idx = get_group_id(0) * AGG_STRIDE + get_local_id(1);
	if(path_idx >= NUM_OBL_PATHS) 
		return;	

	//from topright to downleft
	int i = max(0, -(IMG_W - 1) + path_idx);
	int j = IMG_W - 1 - max(0, IMG_W - 1 - path_idx);
	
	//clear shared memory		
	clear_local_cost_buff(agg_cost_buff, min_cost_buff, INVALID_COST, INVALID_COST);
	barrier(CLK_LOCAL_MEM_FENCE);		

	while(i < IMG_H && j >= 0){		
		int idx = i * IMG_W + j;	
		dp_one_step(d_aggre_cost, d_cost_volume, agg_cost_buff, min_cost_buff, idx, P2);		
		++i; --j;			 
	}
		
	//from downleft to topright	
	clear_local_cost_buff(agg_cost_buff, min_cost_buff, INVALID_COST, INVALID_COST);	
	barrier(CLK_LOCAL_MEM_FENCE);

	while(--i >= 0 && ++j < IMG_W){		
		int idx = i * IMG_W + j;
		dp_one_step(d_aggre_cost, d_cost_volume, agg_cost_buff, min_cost_buff, idx, P2);			
	}
}

inline float sub_pixel(int x, int y, int c, int cost_c,
					   __global const ushort* d_aggre_cost){
	if(x > 0 && x < IMG_W - 1){		
		if(c == 0)
			return (float)(INVALID_DISP);
		int cost_h = d_aggre_cost[(y * IMG_W + x) * DISP_RANGE + c + 1];
		int cost_l = d_aggre_cost[(y * IMG_W + x) * DISP_RANGE + c - 1];
		float numer = 0.5f * (cost_l - cost_h);
		int denom = cost_l + cost_h - 2*cost_c;
		return c + numer / denom ; 
	}else
		return c;
}

inline float sub_pixel_h(int x, int y, int c, int cost_c, int cost_h,
						 __global const ushort* d_aggre_cost){
	if(x > 0 && x < IMG_W - 1){		
		if(c == 0)
			return (float)(INVALID_DISP);
		int cost_l = d_aggre_cost[(y * IMG_W + x) * DISP_RANGE + c - 1];
		float numer = 0.5f * (cost_l - cost_h);
		int denom = cost_l + cost_h - 2*cost_c;
		return c + numer / denom ; 
	}else
		return c;
}

inline float sub_pixel_l(int x, int y, int c, int cost_c, int cost_l,
						__global const ushort* d_aggre_cost){
	if(x > 0 && x < IMG_W - 1){
		int cost_h = d_aggre_cost[(y * IMG_W + x) * DISP_RANGE + c + 1];
	    float numer = 0.5f * (cost_l - cost_h);
		int denom = cost_l + cost_h - 2*cost_c;
		return c + numer / denom ; 
	}else
		return c;
}

__kernel void winner_take_all_kernel1(__global float* disparity,
									  __global const ushort* d_aggre_cost){
	int x = get_global_id(0);
	int y = get_global_id(1);

	if(x > IMG_W - 1 || y > IMG_H - 1)
		return;
		
	ushort cost_min = MAX_COST;
	ushort cost_sec = MAX_COST;
	int index_min = 0;
	//search the minimum and the second minimum value in one pass 		
	for(int i=0; i<DISP_RANGE; i++){
		ushort cost_tmp = d_aggre_cost[(y * IMG_W + x) * DISP_RANGE + i];
		if(cost_tmp < cost_min){
			cost_sec = cost_min;
			cost_min = cost_tmp;			
			index_min = i;		
		}else if(cost_tmp < cost_sec){
			cost_sec = cost_tmp;			
		}		
	}
	
	if(cost_min <= cost_sec * SCALED_THRESH){
		disparity[y * IMG_W + x] = sub_pixel(x, y, index_min, cost_min,
											 d_aggre_cost);
	}else{
		int cost_h, cost_l;
		if(index_min < (DISP_RANGE - 2) && (cost_h = d_aggre_cost[
			(y * IMG_W + x) * DISP_RANGE + index_min + 1]) == cost_sec){
			disparity[y * IMG_W + x] = sub_pixel_h(x, y, index_min, cost_min,
												   cost_h, d_aggre_cost);			
		}else if(index_min > 0 && (cost_l = d_aggre_cost[
			(y * IMG_W + x) * DISP_RANGE + index_min - 1]) == cost_sec){
			disparity[y * IMG_W + x] = sub_pixel_l(x, y, index_min, cost_min,
												   cost_l, d_aggre_cost);
		}else{
			disparity[y * IMG_W + x] = INVALID_DISP;	
		}
	}
}