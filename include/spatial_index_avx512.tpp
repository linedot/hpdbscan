#include <immintrin.h>

template<>
template<>
Cluster<std::int32_t> SpatialIndex<float>::region_query<std::int32_t>(
        const std::int32_t point_index,
        const std::vector<std::int32_t>& neighboring_points,
        const float EPS2,
        const Clusters<std::int32_t>& clusters,
        std::vector<std::int32_t>& min_points_area,
        std::int32_t& count) const {

    return region_query_optimized<std::int32_t>(
            point_index,
            neighboring_points,
            EPS2,
            clusters,
            min_points_area,
            count
            );
}

template<>
template<>
Cluster<std::int64_t> SpatialIndex<float>::region_query<std::int64_t>(
        const std::int64_t point_index,
        const std::vector<std::int64_t>& neighboring_points,
        const float EPS2,
        const Clusters<std::int64_t>& clusters,
        std::vector<std::int64_t>& min_points_area,
        std::int64_t& count) const {

    return region_query_optimized<std::int64_t>(
            point_index,
            neighboring_points,
            EPS2,
            clusters,
            min_points_area,
            count
            );
}
