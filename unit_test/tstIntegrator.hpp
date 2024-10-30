/****************************************************************************
 * Copyright (c) 2022 by Oak Ridge National Laboratory                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

// #include <CabanaNewPkg_Integrate.hpp>
#include <CabanaNewPkg_Particles.hpp>

namespace Test
{
//---------------------------------------------------------------------------//
void testIntegratorReversibility( int steps )
{
    using exec_space = TEST_EXECSPACE;

    // std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    // std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    // std::array<int, 3> num_cells = { 10, 10, 10 };

    // CabanaNewPkg::Particles<TEST_MEMSPACE>
    //     particles( exec_space(), box_min, box_max, num_cells, 0 );
    // auto x = particles.sliceReferencePosition();
    // std::size_t num_particle = x.size();
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, test_integrate_reversibility )
{
    testIntegratorReversibility( 100 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
