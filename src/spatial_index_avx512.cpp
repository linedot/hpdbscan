#include <immintrin.h>

#define IMPLEMENTING_OPTIMIZATION

#include "spatial_index.h"


#if defined(USE_ND_OPTIMIZATIONS)
template<>
template<>
Cluster<std::int32_t> SpatialIndex<float>::template region_query_optimized_nd<int32_t, 3>(
        const int32_t point_index,
        const std::vector<int32_t>& neighboring_points,
        const float EPS2,
        const Clusters<int32_t>& clusters,
        std::vector<int32_t>& min_points_area,
        int32_t& count) const {
    const size_t dimensions = static_cast<size_t>(m_data.m_chunk[1]);
    const float* point = static_cast<float*>(m_data.m_p) + point_index * dimensions;
    const size_t precision = sizeof(float);
    Cluster<int32_t> cluster_label = m_global_point_offset + point_index + 1;
    const size_t ElementsPerAVX = sizeof(__m512) / precision;

    //  Align memory (This has to be done while data loading, not here. )
    // Initialize array pointers and masks
    size_t npoints = neighboring_points.size();
    min_points_area = std::vector<int32_t>(npoints, NOT_VISITED<int32_t>);
    const float* np_ptr = static_cast<float*>(m_data.m_p);

    __m512 v_eps = _mm512_set1_ps(EPS2);
    // std::cout << "V_EPS: ";
    // print_m512(v_eps);

    __m512 v_zero_ps = _mm512_setzero_ps();
    // std::cout << "V_ZERO_PS: ";
    // print_m512(v_zero_ps);

    __m512i v_zero_epi32 = _mm512_setzero_epi32();
    // std::cout << "V_ZERO_EPI: ";
    // print_m512i(v_zero_epi32);

    __m512i v_one = _mm512_set1_epi32(1);
    // std::cout << "V_ONE: ";
    // print_m512i(v_one);

    __m512i v_dims = _mm512_set1_epi32(dimensions);
    // std::cout << "V_DIMS: ";
    // print_m512i(v_dims);

    for (size_t i = 0; i < npoints; i += ElementsPerAVX){
        // Mask
        __mmask16 mask = ( npoints - i > ElementsPerAVX ) ? (1 << 16) - 1 : ((1 << ( npoints - i )) - 1);
        // Load index into mm512
        __m512i v_indices = _mm512_maskz_loadu_epi32(mask, &neighboring_points[i]);
        // std::cout << "V_INDICES: ";
        // print_m512i(v_indices);
        __m512i v_indices_scaled = _mm512_mullo_epi32(v_indices, v_dims);
        // std::cout << "V_INDICES_SCALED: ";
        // print_m512i(v_indices_scaled);

        // Create result
        __m512 v_results = _mm512_setzero_ps();

        // Loop over dimensions
        //for (size_t d = 0; d < dimensions; d++){
        //    // Load dimension d of current point
        //    __m512 v_current_point_dim = _mm512_set1_ps(point[d]);
        //    // std::cout << "V_CURRENT_POINT_DIM: ";
        //    // print_m512(v_current_point_dim);
        //    // Scale indices
        //    // Load dimension d of other points (shifting base address to match current dimension, and scaling the indices by number of dimensions)
        //    __m512 v_other_points_dim = _mm512_mask_i32gather_ps(v_zero_ps, mask, v_indices_scaled, np_ptr + d, 4);

        //    // Get the diff
        //    __m512 v_diff = _mm512_sub_ps(v_current_point_dim, v_other_points_dim);
        //    // std::cout << "V_DIFF: ";
        //    // print_m512(v_diff);

        //    // Square that diff
        //    __m512 v_sqrd = _mm512_mul_ps(v_diff, v_diff);
        //    // std::cout << "V_SQRD: ";
        //    // print_m512(v_sqrd);

        //    // Add to results
        //    v_results = _mm512_add_ps(v_results, v_sqrd);   
        //}
        // ******* UNROLLED DIMENSION LOOP ************
        __m512 v_current_point_x = _mm512_set1_ps(point[0]);
        __m512 v_current_point_y = _mm512_set1_ps(point[1]);
        __m512 v_current_point_z = _mm512_set1_ps(point[2]);

        __m512 v_other_points_x = _mm512_mask_i32gather_ps(v_zero_ps, mask, v_indices_scaled, np_ptr+0, 4);
        __m512 v_other_points_y = _mm512_mask_i32gather_ps(v_zero_ps, mask, v_indices_scaled, np_ptr+1, 4);
        __m512 v_other_points_z = _mm512_mask_i32gather_ps(v_zero_ps, mask, v_indices_scaled, np_ptr+2, 4);

        __m512 v_diff_x = _mm512_sub_ps(v_current_point_x, v_other_points_x);
        v_results = _mm512_fmadd_ps(v_diff_x, v_diff_x, v_results);
        __m512 v_diff_y = _mm512_sub_ps(v_current_point_y, v_other_points_y);
        v_results = _mm512_fmadd_ps(v_diff_y, v_diff_y, v_results);
        __m512 v_diff_z = _mm512_sub_ps(v_current_point_z, v_other_points_z);
        v_results = _mm512_fmadd_ps(v_diff_z, v_diff_z, v_results);
        // ********************************************


        // Mark entries where the result is less than epsilon^2
        __mmask16 v_eps_mask = _mm512_mask_cmple_ps_mask(mask, v_results, v_eps);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask, v_one);
        // std::cout << "Count: " << count << std::endl;

        // Load cluster labels of close entries
        __m512i v_cluster_labels = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask, v_indices, &clusters[0], 4);

        // filter labels that are not visited and that are less than zero
        __mmask16 v_less_than_zero = _mm512_cmplt_epi32_mask(v_cluster_labels, v_zero_epi32);

        v_cluster_labels = _mm512_abs_epi32(v_cluster_labels);

        // set min cluster label
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero, v_cluster_labels));
        // std::cout << "cluster label: " << cluster_label << std::endl;

        // store min_points_area
        _mm512_mask_storeu_epi32(&min_points_area[i], v_eps_mask, v_indices);
    }
    return cluster_label;
}

#endif

template<>
template<>
Cluster<std::int32_t> SpatialIndex<float>::template region_query_optimized<int32_t>(
        const int32_t point_index,
        const std::vector<int32_t>& neighboring_points,
        const float EPS2,
        const Clusters<int32_t>& clusters,
        std::vector<int32_t>& min_points_area,
        int32_t& count) const {
    const size_t dimensions = static_cast<size_t>(m_data.m_chunk[1]);
#if defined(USE_ND_OPTIMIZATIONS)
    if (3 == dimensions)
    {
        return region_query_optimized_nd<int32_t, 3>(
                point_index,
                neighboring_points,
                EPS2,
                clusters,
                min_points_area,
                count
                );
    }
#endif
    const float* point = static_cast<float*>(m_data.m_p) + point_index * dimensions;
    const size_t precision = sizeof(float);
    Cluster<int32_t> cluster_label = m_global_point_offset + point_index + 1;
    const size_t ElementsPerAVX = sizeof(__m512) / precision;

    //  Align memory (This has to be done while data loading, not here. )
    // Initialize array pointers and masks
    size_t npoints = neighboring_points.size();
    min_points_area = std::vector<int32_t>(npoints, NOT_VISITED<int32_t>);
    const float* np_ptr = static_cast<float*>(m_data.m_p);

    __m512 v_eps = _mm512_set1_ps(EPS2);
    // std::cout << "V_EPS: ";
    // print_m512(v_eps);

    __m512 v_zero_ps = _mm512_setzero_ps();
    // std::cout << "V_ZERO_PS: ";
    // print_m512(v_zero_ps);

    __m512i v_zero_epi32 = _mm512_setzero_epi32();
    // std::cout << "V_ZERO_EPI: ";
    // print_m512i(v_zero_epi32);

    __m512i v_one = _mm512_set1_epi32(1);
    // std::cout << "V_ONE: ";
    // print_m512i(v_one);

    __m512i v_dims = _mm512_set1_epi32(dimensions);
    // std::cout << "V_DIMS: ";
    // print_m512i(v_dims);

    for (size_t i = 0; i < npoints; i += ElementsPerAVX){
        // Mask
        __mmask16 mask = ( npoints - i > ElementsPerAVX ) ? (1 << 16) - 1 : ((1 << ( npoints - i )) - 1);
        // Load index into mm512
        __m512i v_indices = _mm512_maskz_loadu_epi32(mask, &neighboring_points[i]);
        // std::cout << "V_INDICES: ";
        // print_m512i(v_indices);
        __m512i v_indices_scaled = _mm512_mullo_epi32(v_indices, v_dims);
        // std::cout << "V_INDICES_SCALED: ";
        // print_m512i(v_indices_scaled);

        // Create result
        __m512 v_results = _mm512_setzero_ps();

        // Loop over dimensions
        for (size_t d = 0; d < dimensions; d++){
            // Load dimension d of current point
            __m512 v_current_point_dim = _mm512_set1_ps(point[d]);
            // std::cout << "V_CURRENT_POINT_DIM: ";
            // print_m512(v_current_point_dim);
            // Scale indices
            // Load dimension d of other points (shifting base address to match current dimension, and scaling the indices by number of dimensions)
            __m512 v_other_points_dim = _mm512_mask_i32gather_ps(v_zero_ps, mask, v_indices_scaled, np_ptr + d, 4);

            // Get the diff
            __m512 v_diff = _mm512_sub_ps(v_current_point_dim, v_other_points_dim);
            // std::cout << "V_DIFF: ";
            // print_m512(v_diff);

            // Square that diff
            __m512 v_sqrd = _mm512_mul_ps(v_diff, v_diff);
            // std::cout << "V_SQRD: ";
            // print_m512(v_sqrd);

            // Add to results
            v_results = _mm512_add_ps(v_results, v_sqrd);   
        }

        // Mark entries where the result is less than epsilon^2
        __mmask16 v_eps_mask = _mm512_mask_cmple_ps_mask(mask, v_results, v_eps);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask, v_one);
        // std::cout << "Count: " << count << std::endl;

        // Load cluster labels of close entries
        __m512i v_cluster_labels = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask, v_indices, &clusters[0], 4);

        // filter labels that are not visited and that are less than zero
        __mmask16 v_less_than_zero = _mm512_cmplt_epi32_mask(v_cluster_labels, v_zero_epi32);

        v_cluster_labels = _mm512_abs_epi32(v_cluster_labels);

        // set min cluster label
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero, v_cluster_labels));
        // std::cout << "cluster label: " << cluster_label << std::endl;

        // store min_points_area
        _mm512_mask_storeu_epi32(&min_points_area[i], v_eps_mask, v_indices);
    }
    return cluster_label;
}
