#include <arm_sve.h>

template<>
template<>
Cluster<std::int32_t> SpatialIndex<float>::template region_query<int32_t>(
        const int32_t point_index,
        const std::vector<int32_t>& neighboring_points,
        const float EPS2,
        const Clusters& clusters,
        std::vector<int32_t>& min_points_area,
        int32_t& count) const {

    return region_query_optimized<int32_t>(
            point_index,
            neighboring_points,
            EPS2,
            clusters,
            min_points_area,
            count
            );
}

