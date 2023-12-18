/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: MPI type selection utility
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#ifndef MPI_UTIL_H
#define MPI_UTIL_H

#include <cstdint>
#include <type_traits>

#include <mpi.h>

// Mpi Type Case
#define MTCex(type, mpi_type, extra) \
    if(std::is_same_v<type, index_type>) \
    { \
        return mpi_type; \
    } \
    extra
#define MTC(type, mpi_type) MTCex(type, mpi_type,)
#define MTCe(type, mpi_type) MTCex(type, mpi_type, else)

    template<typename index_type>
    inline constexpr MPI_Datatype get_mpi_type()
    {
        MTCe(int8_t, MPI_INT8_T)
        MTCe(int16_t, MPI_INT16_T)
        MTCe(int32_t, MPI_INT32_T)
        MTCe(int64_t, MPI_INT64_T)
        MTCe(uint8_t, MPI_UINT8_T)
        MTCe(uint16_t, MPI_UINT16_T)
        MTCe(uint32_t, MPI_UINT32_T)
        MTCe(uint64_t, MPI_UINT64_T)
        MTCe(float, MPI_FLOAT)
        MTC(double, MPI_DOUBLE)

        // gotta noexcept for constexpr, so just return int
        return MPI_INT;
    }

// Don't pollute compiler with macros
#undef MTCe
#undef MTC
#undef MTCex

#endif // MPI_UTIL_H
