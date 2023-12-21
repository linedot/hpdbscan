#ifdef __ARM_FEATURE_SVE

#include <arm_sve.h>

#define IMPLEMENTING_OPTIMIZATION

#include "spatial_index.h"


typedef std::int32_t index_type;

template<>
template<>
Cluster<index_type> SpatialIndex<float>::template region_query_optimized<index_type>(
        const index_type point_index,
        const std::vector<index_type>& neighboring_points,
        const float EPS2,
        const Clusters<index_type>& clusters,
        std::vector<index_type>& min_points_area,
        index_type& count) const {

    const std::int32_t dimensions = static_cast<std::int32_t>(m_data.m_chunk[1]);

    const float* point = static_cast<float*>(m_data.m_p) + point_index * dimensions;

    Cluster<index_type> cluster_label = m_global_point_offset + point_index + 1;

    size_t n = neighboring_points.size();

    min_points_area = std::vector<index_type>(n, NOT_VISITED<index_type>);

    const float* neighbouring_points_ptr = static_cast<float*>(m_data.m_p);

    for (size_t i = 0; i < n; i += svcntw()) {

        svbool_t pg = svwhilelt_b32(i, n);

        svint32_t sv_indices = svld1_s32(pg, &neighboring_points[i]);

        svint32_t sv_indices_scaled = svmul_n_s32_z(pg, sv_indices, dimensions);

        svfloat32_t results_v = svdup_n_f32(0.0f);

        for(size_t d = 0; d < dimensions; d++) {

            svfloat32_t point_coordinate_v = svdup_n_f32(point[d]); 

            svint32_t  other_point_index = svadd_n_s32_z(pg, sv_indices_scaled, d);

            svfloat32_t other_point_coordinate_v = svld1_gather_s32index_f32(pg, &neighbouring_points_ptr[0], other_point_index);

            svfloat32_t diff_v = svsub_f32_x(pg, other_point_coordinate_v, point_coordinate_v);

            svfloat32_t diff_square = svmul_f32_x(pg, diff_v, diff_v);

            results_v = svadd_x(pg, results_v, diff_square);

        }

        svbool_t mask = svcmple_n_f32(pg, results_v, EPS2);

        count += svcntp_b32(pg, mask);

        svint32_t cluster_labels_of_neighbours = svld1_gather_s32index_s32(mask, &clusters[0], sv_indices); //load only cluster labels of distances less than ESP2

        svbool_t not_visited = svcmpne_n_s32(mask, cluster_labels_of_neighbours, NOT_VISITED<index_type>); //NOT_VISITED_s32 is equal to INT_MAX

        svbool_t less_than_zero = svcmplt_n_s32(mask, cluster_labels_of_neighbours, 0);

        cluster_labels_of_neighbours = svabs_s32_z(less_than_zero, cluster_labels_of_neighbours);

        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero, cluster_labels_of_neighbours));

        svst1_s32(mask, &min_points_area[i], sv_indices);

    }

    return cluster_label;

}

#endif

