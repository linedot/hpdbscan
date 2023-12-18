/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: Highly parallel DBSCAN algorithm implementation
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#include <cstdint>

#include "hpdbscan.h"

// explicit template instantiation
template Clusters<std::int16_t> HPDBSCAN::cluster<uint8_t ,std::int16_t>(Dataset&, int);
template Clusters<std::int16_t> HPDBSCAN::cluster<uint16_t,std::int16_t>(Dataset&, int);
template Clusters<std::int16_t> HPDBSCAN::cluster<uint32_t,std::int16_t>(Dataset&, int);
template Clusters<std::int16_t> HPDBSCAN::cluster<uint64_t,std::int16_t>(Dataset&, int);

template Clusters<std::int16_t> HPDBSCAN::cluster<int8_t ,std::int16_t>(Dataset&, int);
template Clusters<std::int16_t> HPDBSCAN::cluster<int16_t,std::int16_t>(Dataset&, int);
template Clusters<std::int16_t> HPDBSCAN::cluster<int32_t,std::int16_t>(Dataset&, int);
template Clusters<std::int16_t> HPDBSCAN::cluster<int64_t,std::int16_t>(Dataset&, int);

template Clusters<std::int16_t> HPDBSCAN::cluster<float ,std::int16_t>(Dataset&, int);
template Clusters<std::int16_t> HPDBSCAN::cluster<double,std::int16_t>(Dataset&, int);

template Clusters<std::int16_t> HPDBSCAN::cluster(const std::string& path, const std::string& dataset);
template Clusters<std::int16_t> HPDBSCAN::cluster(const std::string& path, const std::string& dataset, int threads);

template Clusters<std::int32_t> HPDBSCAN::cluster<uint8_t ,std::int32_t>(Dataset&, int);
template Clusters<std::int32_t> HPDBSCAN::cluster<uint16_t,std::int32_t>(Dataset&, int);
template Clusters<std::int32_t> HPDBSCAN::cluster<uint32_t,std::int32_t>(Dataset&, int);
template Clusters<std::int32_t> HPDBSCAN::cluster<uint64_t,std::int32_t>(Dataset&, int);

template Clusters<std::int32_t> HPDBSCAN::cluster<int8_t ,std::int32_t>(Dataset&, int);
template Clusters<std::int32_t> HPDBSCAN::cluster<int16_t,std::int32_t>(Dataset&, int);
template Clusters<std::int32_t> HPDBSCAN::cluster<int32_t,std::int32_t>(Dataset&, int);
template Clusters<std::int32_t> HPDBSCAN::cluster<int64_t,std::int32_t>(Dataset&, int);

template Clusters<std::int32_t> HPDBSCAN::cluster<float ,std::int32_t>(Dataset&, int);
template Clusters<std::int32_t> HPDBSCAN::cluster<double,std::int32_t>(Dataset&, int);

template Clusters<std::int32_t> HPDBSCAN::cluster(const std::string& path, const std::string& dataset);
template Clusters<std::int32_t> HPDBSCAN::cluster(const std::string& path, const std::string& dataset, int threads);

template Clusters<std::int64_t> HPDBSCAN::cluster<uint8_t ,std::int64_t>(Dataset&, int);
template Clusters<std::int64_t> HPDBSCAN::cluster<uint16_t,std::int64_t>(Dataset&, int);
template Clusters<std::int64_t> HPDBSCAN::cluster<uint32_t,std::int64_t>(Dataset&, int);
template Clusters<std::int64_t> HPDBSCAN::cluster<uint64_t,std::int64_t>(Dataset&, int);

template Clusters<std::int64_t> HPDBSCAN::cluster<int8_t ,std::int64_t>(Dataset&, int);
template Clusters<std::int64_t> HPDBSCAN::cluster<int16_t,std::int64_t>(Dataset&, int);
template Clusters<std::int64_t> HPDBSCAN::cluster<int32_t,std::int64_t>(Dataset&, int);
template Clusters<std::int64_t> HPDBSCAN::cluster<int64_t,std::int64_t>(Dataset&, int);

template Clusters<std::int64_t> HPDBSCAN::cluster<float ,std::int64_t>(Dataset&, int);
template Clusters<std::int64_t> HPDBSCAN::cluster<double,std::int64_t>(Dataset&, int);

template Clusters<std::int64_t> HPDBSCAN::cluster(const std::string& path, const std::string& dataset);
template Clusters<std::int64_t> HPDBSCAN::cluster(const std::string& path, const std::string& dataset, int threads);
