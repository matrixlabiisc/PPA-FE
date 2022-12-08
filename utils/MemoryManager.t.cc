//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------


/*
 * @author Ian C. Lin, Sambit Das.
 */

#include <DeviceAPICalls.h>
#include <algorithm>


namespace dftfe
{
  namespace utils
  {
    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST>::allocate(size_type   size,
                                                          ValueType **ptr)
    {
      *ptr = new ValueType[size];
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST>::deallocate(ValueType *ptr)
    {
      delete[] ptr;
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST>::set(size_type  size,
                                                     ValueType *ptr,
                                                     ValueType  val)
    {
      std::fill(ptr, ptr + size, val);
    }

#ifdef DFTFE_WITH_DEVICE
    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST_PINNED>::allocate(
      size_type   size,
      ValueType **ptr)
    {
      deviceHostMalloc((void **)ptr, size * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST_PINNED>::deallocate(
      ValueType *ptr)
    {
      if (ptr != nullptr)
        deviceHostFree(ptr);
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST_PINNED>::set(size_type  size,
                                                            ValueType *ptr,
                                                            ValueType  val)
    {
      std::fill(ptr, ptr + size, val);
    }


    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::DEVICE>::allocate(size_type   size,
                                                            ValueType **ptr)
    {
      deviceMalloc((void **)ptr, size * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::DEVICE>::deallocate(ValueType *ptr)
    {
      deviceFree(ptr);
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::DEVICE>::set(size_type  size,
                                                       ValueType *ptr,
                                                       ValueType  val)
    {
      deviceSetValue(ptr, val, size);
    }

#endif // DFTFE_WITH_DEVICE
  }    // namespace utils

} // namespace dftfe
