#ifdef __ARM_FEATURE_SVE

#include <arm_sve.h>

#define IMPLEMENTING_OPTIMIZATION

#include "spatial_index.h"

    template<>
    Cluster<std::int32_t> SpatialIndex<float>::region_query_optimized(
            const int32_t point_index,
            const std::vector<int32_t>& neighboring_points,
            const float EPS2,
            const Clusters& clusters,
            std::vector<int32_t>& min_points_area,
            int32_t& count) const {
        
	const uint32_t dimensions = static_cast<uint32_t>(m_data.m_chunk[1]);
        
	const float* point = static_cast<float*>(m_data.m_p) + point_index * dimensions;
        
	Cluster cluster_label = m_global_point_offset + point_index + 1;

	size_t n = neighboring_points.size();

	min_points_area = std::vector<int32_t>(n, NOT_VISITED<int32_t>);

	const float* neighbouring_points_ptr = static_cast<float*>(m_data.m_p);

	for (size_t i = 0; i < n; i += svcntw()) {

            svbool_t pg = svwhilelt_b32(i, n);

            svuint32_t sv_indices = svld1_u32(pg, &neighboring_points[i]);

	    svuint32_t sv_indices_scaled = svmul_n_u32_z(pg, sv_indices, dimensions);

	    svfloat32_t results_v = svdup_n_f32(0.0f);

	    for(size_t d = 0; d < dimensions; d++) {
           
		svfloat32_t point_coordinate_v = svdup_n_f32(point[d]); 
	        
		svuint32_t  other_point_index = svadd_n_u32_z(pg, sv_indices_scaled, d);
		
		svfloat32_t other_point_coordinate_v = svld1_gather_u32index_f32(pg, &neighbouring_points_ptr[0], other_point_index);
		
		svfloat32_t diff_v = svsub_f32_x(pg, other_point_coordinate_v, point_coordinate_v);

		svfloat32_t diff_square = svmul_f32_x(pg, diff_v, diff_v);

                results_v = svadd_x(pg, results_v, diff_square);

	    }

            svbool_t mask = svcmple_n_f32(pg, results_v, EPS2);

	    count += svcntp_b32(pg, mask);

	    svint32_t cluster_labels_of_neighbours = svld1_gather_u32index_s32(mask, &clusters[0], sv_indices); //load only cluster labels of distances less than ESP2

	    svbool_t not_visited = svcmpne_n_s32(mask, cluster_labels_of_neighbours, NOT_VISITED); //NOT_VISITED_s32 is equal to INT_MAX

	    svbool_t less_than_zero = svcmplt_n_s32(mask, cluster_labels_of_neighbours, 0);

	    cluster_labels_of_neighbours = svabs_s32_z(less_than_zero, cluster_labels_of_neighbours);

	    cluster_label = std::min(cluster_label, svminv_s32(less_than_zero, cluster_labels_of_neighbours));

	    svst1_u32(mask, &min_points_area[i], sv_indices);

        }
        
	return cluster_label;

    }

#endif

